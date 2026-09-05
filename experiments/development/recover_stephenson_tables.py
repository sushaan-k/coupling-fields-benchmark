"""Recover the original selected cells through bounded HTTP range reads.

Only the 92 previously analyzed donors enter the output. The original raw file
is not copied to disk; the output retains the selected 18-feature count matrix.
"""

import argparse
import hashlib
import json
from pathlib import Path

import fsspec
import h5py
import numpy as np
from scipy.sparse import csr_matrix

from experiments import confirm_stephenson_citeseq as original


ROOT = Path(__file__).resolve().parents[2]
URL = ("https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/026/E-MTAB-10026/Files/"
       "covid_portal_210320_with_raw.h5ad")


def recover(output, cache=None):
    manifest = json.loads(original.DEFAULT_SOURCE.read_text())
    records = [r for r in manifest["samples"]
               if r["role"] in {"calibration", "pilot", "held_site"}]
    assert len(records) == 92
    predictions = json.loads((ROOT / "results/stephenson_citeseq_predictions.json").read_text())
    expected_hashes = {s["sample"]: s["selected_barcode_sha256"]
                       for s in predictions["samples"]}
    if cache is None:
        stream = fsspec.open(URL, block_size=4 * 1024 * 1024, cache_type="blockcache",
                             cache_options={"maxblocks":32}).open()
    else:
        from experiments.development.sparse_http_cache import SparseHTTPFile
        stream = SparseHTTPFile(URL, manifest["h5ad"]["bytes"], cache)
    if stream.size != manifest["h5ad"]["bytes"]:
        raise ValueError("remote raw object size differs from original manifest")
    with stream, h5py.File(stream, "r") as handle:
        obs = handle["obs"]
        barcodes = original._dataframe_index(handle, obs)
        samples = original._encoded_column(handle, obs, "sample_id")
        sites = original._encoded_column(handle, obs, "Site")
        clusters = original._encoded_column(handle, obs, "initial_clustering")
        cell_types = np.array([original.CLUSTER_TO_CELL_TYPE[x] for x in clusters])
        selections = []
        for record in records:
            pool = np.flatnonzero(samples == record["sample"])
            assert len(pool) == record["eligible_pool_cells"]
            assert set(sites[pool]) == {record["site"]}
            selected = sorted(sorted(pool, key=lambda i: (
                original._cell_hash(record["donor"], record["sample"], barcodes[i]),
                barcodes[i]))[:512])
            digest = hashlib.sha256(("\n".join(sorted(barcodes[selected])) + "\n").encode()).hexdigest()
            if record["sample"] in expected_hashes:
                assert digest == expected_hashes[record["sample"]]
            selections.append(selected)
        selections = np.asarray(selections)
        ordered_rows = np.sort(selections.ravel())
        assert len(np.unique(ordered_rows)) == 92 * 512
        features = original._feature_columns(handle)
        columns = np.array(features["rna"] + features["adt"])
        matrix = handle["layers/raw"]
        pointers = np.asarray(matrix["indptr"][:], dtype=np.int64)
        values = np.zeros((len(ordered_rows), len(columns)), dtype=np.int32)
        print(f"Selected {len(ordered_rows)} cells; reading 18 features", flush=True)
        for offset in range(0, len(ordered_rows), 256):
            rows = ordered_rows[offset:offset + 256]
            start, stop = int(rows[0]), int(rows[-1]) + 1
            left, right = int(pointers[start]), int(pointers[stop])
            indices = np.asarray(matrix["indices"][left:right])
            counts = np.asarray(matrix["data"][left:right])
            slab = csr_matrix((counts, indices, pointers[start:stop + 1] - left),
                              shape=(stop - start, 24929))
            selected_counts = slab[rows - start][:, columns].toarray()
            if np.any(selected_counts < 0) or not np.all(selected_counts == np.rint(selected_counts)):
                raise ValueError("selected raw counts are not nonnegative integers")
            values[offset:offset + len(rows)] = selected_counts
            if offset % 2560 == 0:
                print(f"{offset + len(rows)}/{len(ordered_rows)} cells", flush=True)
        lookup = np.searchsorted(ordered_rows, selections)
        counts = values[lookup].transpose(0, 2, 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, rna_counts=counts[:, :9], adt_counts=counts[:, 9:],
                            cell_types=cell_types[selections], barcodes=barcodes[selections],
                            donor_ids=np.array([r["donor"] for r in records]),
                            sample_ids=np.array([r["sample"] for r in records]),
                            roles=np.array([r["role"] for r in records]),
                            markers=np.array(original.MARKERS))
    print(f"Recovered counts: {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    recover(args.output, args.cache)
