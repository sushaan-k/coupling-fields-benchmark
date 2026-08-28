from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import preflight_gse239452_citeseq as preflight


def _strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype()))


def _raw_matrix(handle: h5py.File, rows: int, columns: int) -> h5py.Group:
    raw = handle.create_group("raw")
    matrix = raw.create_group("X")
    matrix.attrs["encoding-type"] = "csr_matrix"
    matrix.attrs["shape"] = np.asarray([rows, columns], dtype=np.int64)
    matrix.create_dataset("data", data=np.zeros(0, dtype=np.float32))
    matrix.create_dataset("indices", data=np.zeros(0, dtype=np.int32))
    matrix.create_dataset("indptr", data=np.zeros(rows + 1, dtype=np.int32))
    return raw.create_group("var")


def _write_gex(path: Path, donor: str, common: int = 520) -> None:
    barcodes = [f"batch|{donor}-{index:04d}-1-0-1" for index in range(common)]
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "barcodekey"
        _strings(obs, "barcodekey", barcodes)
        var = _raw_matrix(handle, common, len(preflight.EXPECTED_MARKERS))
        _strings(var, "featurekey", list(preflight.EXPECTED_MARKERS))


def _write_adt(
    path: Path, donor: str, rows: int = 520, common: int | None = None
) -> None:
    matched = rows if common is None else common
    barcodes = [f"batch|{donor}-{index:04d}-1" for index in range(matched)]
    barcodes.extend(f"other|{donor}-{index:04d}-1" for index in range(rows - matched))
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        _strings(obs, "_index", barcodes)
        var = _raw_matrix(handle, rows, len(preflight.EXPECTED_MARKERS))
        ids = ["0072", "0066", "0081", "0050", "0052", "0389", "0073", "0026", "0033"]
        _strings(var, "Featurekey", [f"ADT_C{value}" for value in ids])
        _strings(
            var, "NameInData", [f"prot_{value}" for value in preflight.EXPECTED_MARKERS]
        )
        _strings(
            var,
            "target-1",
            [
                f"{identifier} anti-human {marker}"
                for identifier, marker in zip(ids, preflight.EXPECTED_MARKERS)
            ],
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    manifest = copy.deepcopy(json.loads(preflight.DEFAULT_MANIFEST.read_text()))
    panel = source / manifest["official_archives"]["panel"]["filename"]
    panel.write_bytes(b"official panel fixture")
    manifest["official_archives"]["panel"]["bytes"] = panel.stat().st_size
    manifest["official_archives"]["panel"]["sha256"] = hashlib.sha256(
        panel.read_bytes()
    ).hexdigest()
    for sample in manifest["samples"]:
        donor = sample["donor"]
        gex = sample["gex"]
        _write_gex(source / gex["h5ad"], donor, 520 if donor != "100" else 520)
        archive = source / gex["filename"]
        archive.write_bytes(f"gex:{donor}".encode())
        gex["bytes"] = archive.stat().st_size
        adt = sample["adt"]
        if adt is None:
            continue
        if donor == "100":
            _write_adt(source / adt["h5ad"], donor, rows=378, common=378)
        else:
            _write_adt(source / adt["h5ad"], donor)
        archive = source / adt["filename"]
        archive.write_bytes(f"adt:{donor}".encode())
        adt["bytes"] = archive.stat().st_size
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, source, panel


def test_public_manifest_has_frozen_7_8_9_design_and_two_exclusions():
    manifest = json.loads(preflight.DEFAULT_MANIFEST.read_text())
    samples, _ = preflight._validate_manifest(manifest)
    assert {
        role: sum(row["role"] == role for row in samples)
        for role in preflight.EXPECTED_ROLES
    } == {"calibration": 7, "pilot": 8, "held": 9, "excluded_metadata": 2}
    excluded = {
        row["donor"]: row["exclusion"]
        for row in samples
        if row["role"] == "excluded_metadata"
    }
    assert set(excluded) == {"78", "100"}
    assert "378 rows" in excluded["100"]
    assert "no paired ADT" in excluded["78"]


def test_public_metadata_preflight_is_payload_closed_and_all_eligible_donors_pass():
    path = preflight.DEFAULT_OUTPUT
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS"
    assert payload["source_manifest_sha256"] == preflight._sha256(
        preflight.DEFAULT_MANIFEST
    )
    assert payload["access_audit"]["numeric_matrix_payload_values_read"] == 0
    eligible = [
        row
        for row in payload["samples"]
        if row["role"] in {"calibration", "pilot", "held"}
    ]
    assert len(eligible) == 24
    assert min(row["common_barcode_count"] for row in eligible) == 987
    assert all(row["metadata_eligible"] for row in eligible)


def test_metadata_only_preflight_passes_and_records_no_matrix_payload_reads(tmp_path):
    manifest, source, panel = _fixture(tmp_path)
    output = tmp_path / "preflight.json"
    payload = preflight.run_preflight(manifest, source, panel, output)
    assert payload["status"] == "PASS"
    assert payload["role_counts"] == preflight.EXPECTED_ROLES
    assert payload["access_audit"]["numeric_matrix_payload_values_read"] == 0
    assert payload["access_audit"]["raw_X_data_values_read"] == 0
    assert payload["access_audit"]["raw_X_indices_values_read"] == 0
    assert payload["access_audit"]["raw_X_indptr_values_read"] == 0
    donor100 = next(row for row in payload["samples"] if row["donor"] == "100")
    assert donor100["adt_raw_shape"][0] == 378
    assert donor100["common_barcode_count"] == 378
    assert donor100["metadata_eligible"] is False
    assert output.is_file()


def test_preflight_refuses_an_eligible_donor_below_512_common_barcodes(tmp_path):
    manifest, source, panel = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    record = next(row for row in payload["samples"] if row["donor"] == "47")
    _write_adt(source / record["adt"]["h5ad"], "47", rows=520, common=511)
    with pytest.raises(ValueError, match="47.*fewer than 512"):
        preflight.run_preflight(manifest, source, panel, tmp_path / "result.json")


def test_preflight_has_no_fuzzy_adt_feature_fallback(tmp_path):
    manifest, source, panel = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    record = next(row for row in payload["samples"] if row["donor"] == "47")
    path = source / record["adt"]["h5ad"]
    with h5py.File(path, "r+") as handle:
        axis = handle["raw/var/Featurekey"]
        values = axis.asstr()[:].tolist()
        del handle["raw/var/Featurekey"]
        values[6] = "ADT_C0073_extra"
        _strings(handle["raw/var"], "Featurekey", values)
    with pytest.raises(ValueError, match="misses exact frozen features"):
        preflight.run_preflight(manifest, source, panel, tmp_path / "result.json")


def test_gex_canonicalization_refuses_unfrozen_suffix():
    with pytest.raises(ValueError, match="frozen deposited encoding"):
        preflight._canonical_gex_barcode("batch|AAAC-1")
    assert preflight._canonical_gex_barcode("batch|AAAC-1-0-1") == "batch|AAAC-1"
    assert preflight._canonical_gex_barcode("batch|AAAC-1-2-0") == "batch|AAAC-1"
    assert preflight._canonical_gex_barcode("AAAC-Batch_1A-Pregnant") == "AAAC-Batch_1A"
