"""Donor-held confirmation of conditional RNA-protein fields in GSE278572."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapreg.coupling_fields import (  # noqa: E402
    conditional_association_coordinates,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)


DATA_DIR = ROOT / "data/development/arce_gse278572"
CACHE = DATA_DIR / "matched_rna_protein_counts_v1.npz"
SOURCE_MANIFEST = DATA_DIR / "source_manifest_v1.json"
DONOR_A_CACHE = DATA_DIR / "donor_a_matched_rna_protein_counts_v1.npz"
DONOR_A_MANIFEST = DATA_DIR / "donor_a_preprocessing_manifest_v1.json"
DONOR_A_LOCK = DATA_DIR / "donor_a_preprocessing_lock_v1.json"
METADATA_AUDIT = DATA_DIR / "metadata_audit_v1.json"
PREANALYSIS_LOCK = DATA_DIR / "preanalysis_lock_v1.json"
ABORTED_PREANALYSIS_LOCK = (
    DATA_DIR / "preanalysis_lock_aborted_estimator_3823bcfa.json"
)
OUTPUT = ROOT / "results/arce_gse278572_conditional_field_confirmation.json"
PROTOCOL = (
    ROOT
    / "docs/ARCE_GSE278572_CONDITIONAL_FIELD_CONFIRMATION_FREEZE_2026-08-26.md"
)
DEVELOPMENT_RESULT = (
    ROOT / "results/public_coupling_atlas_benchmark_v4_final_estimator.json"
)
DEVELOPMENT_PROTOCOL = (
    ROOT / "docs/PUBLIC_COUPLING_ATLAS_BENCHMARK_FREEZE_2026-08-26.md"
)
SCGPT_EMBEDDING = ROOT / "data/scgpt_gene_embeddings.npz"

SEED = 20260826
STATE_COUNT = 3
PSEUDOCOUNT = 0.5
NULL_PERMUTATIONS = 64
BOOTSTRAPS = 2_000
MINIMUM_ARM_CELLS = 30
CONTROL = "Non-Targeting"
DONORS = ("A", "B")
LINEAGES = ("Treg", "Teff")
CONDITIONS = ("Resting", "Stimulated")
EXCLUDED_CELL = "ACATACGGTTCTGGTA-7"
STRUCTURED_NUCLEAR_FRACTION = 0.1
STRUCTURED_GRAPH_PENALTY = 5.0
ENDPOINT_RIDGE_PENALTY = 0.1
HYPERGRAPH_NEIGHBORS = 5

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl"
MATRIX_URL = f"{GEO_BASE}/GSE278572_matrix.mtx.gz"
MATRIX_COMPRESSED_BYTES = 3_857_480_145
MATRIX_ROWS = 36_807
MATRIX_COLUMNS = 249_799
MATRIX_NONZEROS = 1_027_798_128
ZENODO_URL = "https://zenodo.org/api/records/13924126/files/data_tables.zip/content"
ZENODO_MEMBER = "data_tables/S14_metadata_Treg_Teff_perturbseq.xlsx"

SOURCE_SPECS = {
    "barcodes": {
        "path": DATA_DIR / "GSE278572_barcodes.tsv.gz",
        "url": f"{GEO_BASE}/GSE278572_barcodes.tsv.gz",
        "bytes": 1_121_892,
        "sha256": "0c26723009bb4e507437c49ba92e382506f676c8e3c1ba19245db1208a2514b7",
    },
    "features": {
        "path": DATA_DIR / "GSE278572_features.tsv.gz",
        "url": f"{GEO_BASE}/GSE278572_features.tsv.gz",
        "bytes": 335_300,
        "sha256": "411d0aef6d630eee905f67ea79d1ca0608ff7bc293591aa45ca8019f234f00ba",
    },
    "guide_calls": {
        "path": DATA_DIR / "GSE278572_protospacer_calls_per_cell.csv.gz",
        "url": f"{GEO_BASE}/GSE278572_protospacer_calls_per_cell.csv.gz",
        "bytes": 1_952_405,
        "sha256": "1c556df41f441987033bdfc002bc6a25f5194ec39ebbe2ace66baedb2ec3636e",
    },
}
S14_PATH = DATA_DIR / "S14_metadata_Treg_Teff_perturbseq.xlsx"
S14_BYTES = 29_140_658
S14_SHA256 = "ee33fc765234e073aaac9bf394ea067df63f196c22bb6c7b6dbbead09ee2343b"
ZENODO_ZIP_BYTES = 57_967_623
ZENODO_ZIP_SHA256 = "dc9e2efb04d24f1a6d4b8db6a8b1d5cd01c935777c3740088be339de5b5062b4"

# Antibody names are matched only when the antigen has one unambiguous cognate
# transcript in the deposited gene feature table. The order is part of the
# frozen analysis contract.
ADT_TO_GENE = {
    "Hu.CD101": "CD101",
    "Hu.CD103": "ITGAE",
    "Hu.CD105_43A3": "ENG",
    "Hu.CD107a": "LAMP1",
    "Hu.CD112": "NECTIN2",
    "Hu.CD119": "IFNGR1",
    "Hu.CD11a": "ITGAL",
    "Hu.CD11b": "ITGAM",
    "Hu.CD11c": "ITGAX",
    "Hu.CD122": "IL2RB",
    "Hu.CD123": "IL3RA",
    "Hu.CD124": "IL4R",
    "Hu.CD127": "IL7R",
    "Hu.CD13": "ANPEP",
    "Hu.CD134": "TNFRSF4",
    "Hu.CD137": "TNFRSF9",
    "Hu.CD141": "THBD",
    "Hu.CD146": "MCAM",
    "Hu.CD14_M5E2": "CD14",
    "Hu.CD152": "CTLA4",
    "Hu.CD154": "CD40LG",
    "Hu.CD155": "PVR",
    "Hu.CD161": "KLRB1",
    "Hu.CD163": "CD163",
    "Hu.CD169": "SIGLEC1",
    "Hu.CD18": "ITGB2",
    "Hu.CD183": "CXCR3",
    "Hu.CD185": "CXCR5",
    "Hu.CD19": "CD19",
    "Hu.CD194": "CCR4",
    "Hu.CD195": "CCR5",
    "Hu.CD196": "CCR6",
    "Hu.CD1c": "CD1C",
    "Hu.CD1d": "CD1D",
    "Hu.CD2": "CD2",
    "Hu.CD20_2H7": "MS4A1",
    "Hu.CD21": "CR2",
    "Hu.CD22": "CD22",
    "Hu.CD223": "LAG3",
    "Hu.CD226_11A8": "CD226",
    "Hu.CD23": "FCER2",
    "Hu.CD24": "CD24",
    "Hu.CD244": "CD244",
    "Hu.CD25": "IL2RA",
    "Hu.CD26": "DPP4",
    "Hu.CD267": "TNFRSF13B",
    "Hu.CD268": "TNFRSF13C",
    "Hu.CD27": "CD27",
    "Hu.CD270": "TNFRSF14",
    "Hu.CD272": "BTLA",
    "Hu.CD274": "CD274",
    "Hu.CD279": "PDCD1",
    "Hu.CD28": "CD28",
    "Hu.CD29": "ITGB1",
    "Hu.CD303": "CLEC4C",
    "Hu.CD31": "PECAM1",
    "Hu.CD314": "KLRK1",
    "Hu.CD319": "SLAMF7",
    "Hu.CD328": "SIGLEC7",
    "Hu.CD33": "CD33",
    "Hu.CD335": "NCR1",
    "Hu.CD35": "CR1",
    "Hu.CD352": "SLAMF6",
    "Hu.CD36": "CD36",
    "Hu.CD38_HIT2": "CD38",
    "Hu.CD39": "ENTPD1",
    "Hu.CD40": "CD40",
    "Hu.CD41": "ITGA2B",
    "Hu.CD42b": "GP1BA",
    "Hu.CD45_HI30": "PTPRC",
    "Hu.CD47": "CD47",
    "Hu.CD48": "CD48",
    "Hu.CD49a": "ITGA1",
    "Hu.CD49b": "ITGA2",
    "Hu.CD49d": "ITGA4",
    "Hu.CD4_RPA.T4": "CD4",
    "Hu.CD5": "CD5",
    "Hu.CD52": "CD52",
    "Hu.CD54": "ICAM1",
    "Hu.CD56": "NCAM1",
    "Hu.CD57": "B3GAT1",
    "Hu.CD58": "CD58",
    "Hu.CD62L": "SELL",
    "Hu.CD62P": "SELP",
    "Hu.CD64": "FCGR1A",
    "Hu.CD69": "CD69",
    "Hu.CD7": "CD7",
    "Hu.CD71": "TFRC",
    "Hu.CD73": "NT5E",
    "Hu.CD79b": "CD79B",
    "Hu.CD8": "CD8A",
    "Hu.CD81": "CD81",
    "Hu.CD82": "CD82",
    "Hu.CD83": "CD83",
    "Hu.CD85j": "LILRB1",
    "Hu.CD86": "CD86",
    "Hu.CD88": "C5AR1",
    "Hu.CD94": "KLRD1",
    "Hu.CD95": "FAS",
    "Hu.CD99": "CD99",
    "Hu.CLEC12A": "CLEC12A",
    "Hu.CX3CR1": "CX3CR1",
    "Hu.FceRIa": "FCER1A",
    "Hu.GPR56": "ADGRG1",
    "Hu.HLA.E": "HLA-E",
    "Hu.KLRG1": "KLRG1",
    "Hu.LOX.1": "OLR1",
    "Hu.TIGIT": "TIGIT",
    "HuMs.CD44": "CD44",
    "HuMs.CD49f": "ITGA6",
    "HuMs.integrin.b7": "ITGB7",
    "HuMsRt.CD278": "ICOS",
}

EXTRACTOR_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class FastInput {
 public:
  static constexpr size_t kSize = 1u << 20;
  char buffer[kSize];
  size_t position = 0;
  size_t length = 0;

  int get() {
    if (position == length) {
      length = std::fread(buffer, 1, kSize, stdin);
      position = 0;
      if (length == 0) return EOF;
    }
    return static_cast<unsigned char>(buffer[position++]);
  }

  bool skip_line() {
    int value;
    while ((value = get()) != EOF) {
      if (value == '\n') return true;
    }
    return false;
  }

  bool next_uint(uint64_t& output) {
    int value;
    do {
      value = get();
      if (value == EOF) return false;
    } while (value < '0' || value > '9');
    output = 0;
    do {
      output = output * 10 + static_cast<uint64_t>(value - '0');
      value = get();
    } while (value >= '0' && value <= '9');
    return true;
  }
};

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: extractor mapping output cells features\n";
    return 2;
  }
  const std::string mapping_path = argv[1];
  const std::string output_path = argv[2];
  const size_t selected_cells = std::stoull(argv[3]);
  const size_t selected_features = std::stoull(argv[4]);

  std::vector<int32_t> row_map(36808, -1);
  std::vector<int32_t> column_map(249800, -1);
  std::ifstream mapping(mapping_path);
  char kind;
  size_t source;
  size_t destination;
  while (mapping >> kind >> source >> destination) {
    if (kind == 'R') row_map.at(source) = static_cast<int32_t>(destination);
    if (kind == 'C') column_map.at(source) = static_cast<int32_t>(destination);
  }
  if (!mapping.eof()) {
    std::cerr << "invalid mapping file\n";
    return 3;
  }

  std::vector<uint32_t> counts(selected_cells * selected_features, 0);
  FastInput input;
  if (!input.skip_line() || !input.skip_line()) {
    std::cerr << "truncated Matrix Market header\n";
    return 4;
  }
  uint64_t rows, columns, nonzeros;
  if (!input.next_uint(rows) || !input.next_uint(columns) ||
      !input.next_uint(nonzeros)) {
    std::cerr << "missing Matrix Market dimensions\n";
    return 5;
  }
  if (rows != 36807 || columns != 249799 || nonzeros != 1027798128ULL) {
    std::cerr << "unexpected Matrix Market dimensions\n";
    return 6;
  }
  uint64_t retained = 0;
  for (uint64_t index = 0; index < nonzeros; ++index) {
    uint64_t row, column, value;
    if (!input.next_uint(row) || !input.next_uint(column) ||
        !input.next_uint(value)) {
      std::cerr << "truncated Matrix Market body at entry " << index << "\n";
      return 7;
    }
    const int32_t target_row = row_map.at(row);
    const int32_t target_column = column_map.at(column);
    if (target_row >= 0 && target_column >= 0) {
      const size_t output = static_cast<size_t>(target_column) * selected_features +
                            static_cast<size_t>(target_row);
      const uint64_t total = static_cast<uint64_t>(counts[output]) + value;
      if (total > UINT32_MAX) {
        std::cerr << "count overflow\n";
        return 8;
      }
      counts[output] = static_cast<uint32_t>(total);
      ++retained;
    }
  }
  std::ofstream output(output_path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(counts.data()),
               static_cast<std::streamsize>(counts.size() * sizeof(uint32_t)));
  if (!output) {
    std::cerr << "failed to write selected count matrix\n";
    return 9;
  }
  std::cerr << "retained_entries=" << retained << "\n";
  return 0;
}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 << 20)
    temporary.replace(path)


def _validate_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path.name} has the wrong byte count")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{path.name} failed its SHA-256 check")


def _ensure_small_sources() -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, object] = {}
    for name, specification in SOURCE_SPECS.items():
        path = specification["path"]
        if not path.exists():
            _download(str(specification["url"]), path)
        _validate_file(
            path,
            expected_bytes=int(specification["bytes"]),
            expected_sha256=str(specification["sha256"]),
        )
        inventory[name] = {
            "path": str(path.relative_to(ROOT)),
            "url": specification["url"],
            "bytes": specification["bytes"],
            "sha256": specification["sha256"],
        }

    if not S14_PATH.exists():
        archive = DATA_DIR / "data_tables.zip"
        _download(ZENODO_URL, archive)
        _validate_file(
            archive,
            expected_bytes=ZENODO_ZIP_BYTES,
            expected_sha256=ZENODO_ZIP_SHA256,
        )
        with zipfile.ZipFile(archive) as handle:
            with handle.open(ZENODO_MEMBER) as source, S14_PATH.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 << 20)
        archive.unlink()
    _validate_file(S14_PATH, expected_bytes=S14_BYTES, expected_sha256=S14_SHA256)
    inventory["cell_metadata"] = {
        "path": str(S14_PATH.relative_to(ROOT)),
        "url": ZENODO_URL,
        "archive_member": ZENODO_MEMBER,
        "archive_bytes": ZENODO_ZIP_BYTES,
        "archive_sha256": ZENODO_ZIP_SHA256,
        "bytes": S14_BYTES,
        "sha256": S14_SHA256,
    }
    return inventory


def _write_or_validate_lock() -> dict[str, object]:
    if not ABORTED_PREANALYSIS_LOCK.exists():
        raise FileNotFoundError("the preserved aborted-run lock is missing")
    expected = {
        "schema": "v2p2r.arce-gse278572-preanalysis-lock/1",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "estimator": "mapreg/coupling_fields.py",
        "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
        "development_result": str(DEVELOPMENT_RESULT.relative_to(ROOT)),
        "development_result_sha256": _sha256(DEVELOPMENT_RESULT),
        "development_protocol": str(DEVELOPMENT_PROTOCOL.relative_to(ROOT)),
        "development_protocol_sha256": _sha256(DEVELOPMENT_PROTOCOL),
        "outcome_matrix_opened_at_lock": False,
        "analytical_holdout_disclosure": (
            "donor B supplied no statistic, state, field, score, or human-visible "
            "value before this corrected lock; raw counts entered only the "
            "transient buffer of an interrupted prior stream"
        ),
        "donor_b_literally_unopened_before_corrected_lock": False,
        "donor_b_analytical_summary_before_corrected_lock": False,
        "aborted_prior_lock": str(ABORTED_PREANALYSIS_LOCK.relative_to(ROOT)),
        "aborted_prior_lock_sha256": _sha256(ABORTED_PREANALYSIS_LOCK),
    }
    if PREANALYSIS_LOCK.exists():
        observed = json.loads(PREANALYSIS_LOCK.read_text())
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(f"preanalysis lock mismatch for {key}")
        return observed
    expected["locked_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(PREANALYSIS_LOCK, expected)
    return expected


def _read_metadata() -> pd.DataFrame:
    columns = [
        "cell",
        "nCount_RNA",
        "sgrna",
        "sg_target",
        "nCount_ADT",
        "donor",
        "hash.ID",
    ]
    metadata = pd.read_excel(S14_PATH, usecols=columns)
    metadata = metadata[metadata["cell"] != EXCLUDED_CELL].copy()
    metadata["cell"] = metadata["cell"].astype(str)
    metadata["sgrna"] = metadata["sgrna"].astype(str)
    metadata["sg_target"] = metadata["sg_target"].astype(str)
    metadata["donor"] = metadata["donor"].astype(str)
    metadata["hash.ID"] = metadata["hash.ID"].astype(str)
    if len(metadata) != 100_086 or not metadata["cell"].is_unique:
        raise ValueError("unexpected final-cell metadata cardinality")
    if set(metadata["donor"]) != set(DONORS):
        raise ValueError("unexpected donor labels")
    expected_states = {
        f"{condition}-{lineage}"
        for condition in CONDITIONS
        for lineage in LINEAGES
    }
    if set(metadata["hash.ID"]) != expected_states:
        raise ValueError("unexpected HTO state labels")
    return metadata


def _feature_specification() -> dict[str, object]:
    rows = []
    with gzip.open(SOURCE_SPECS["features"]["path"], "rt") as handle:
        rows = [tuple(row) for row in csv.reader(handle, delimiter="\t")]
    if len(rows) != MATRIX_ROWS:
        raise ValueError("unexpected feature count")
    genes: dict[str, list[int]] = {}
    antibody_rows: dict[str, int] = {}
    biological_adts = []
    for index, (identifier, name, feature_type) in enumerate(rows, start=1):
        if feature_type == "Gene Expression":
            genes.setdefault(name, []).append(index)
        elif feature_type == "Antibody Capture":
            antibody_rows[name] = index
            if not identifier.startswith("HTO_") and not name.startswith("Isotype_"):
                biological_adts.append(name)
    if len(biological_adts) != 130:
        raise ValueError("unexpected biological ADT count")
    missing_adts = sorted(set(ADT_TO_GENE) - set(antibody_rows))
    missing_genes = sorted(set(ADT_TO_GENE.values()) - set(genes))
    duplicated_genes = sorted(
        gene for gene in set(ADT_TO_GENE.values()) if len(genes.get(gene, [])) != 1
    )
    if missing_adts or missing_genes or duplicated_genes:
        raise ValueError(
            f"feature-map mismatch: ADT={missing_adts}, gene={missing_genes}, "
            f"duplicated={duplicated_genes}"
        )
    matched_adts = list(ADT_TO_GENE)
    matched_genes = [ADT_TO_GENE[name] for name in matched_adts]
    row_mapping = []
    for output, gene in enumerate(matched_genes):
        row_mapping.append((genes[gene][0], output))
    offset = len(matched_genes)
    for output, adt in enumerate(biological_adts, start=offset):
        row_mapping.append((antibody_rows[adt], output))
    protein_positions = [biological_adts.index(name) for name in matched_adts]
    return {
        "matched_adts": matched_adts,
        "matched_genes": matched_genes,
        "biological_adts": biological_adts,
        "protein_positions": protein_positions,
        "row_mapping": row_mapping,
        "selected_features": len(row_mapping),
    }


def _metadata_audit(sources: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    metadata = _read_metadata()
    with gzip.open(SOURCE_SPECS["barcodes"]["path"], "rt") as handle:
        barcodes = [line.rstrip("\n") for line in handle]
    if len(barcodes) != MATRIX_COLUMNS or len(set(barcodes)) != len(barcodes):
        raise ValueError("unexpected or duplicated matrix barcodes")
    missing = sorted(set(metadata["cell"]) - set(barcodes))
    if missing:
        raise ValueError(f"metadata cells absent from matrix: {missing[:5]}")

    guide_calls = pd.read_csv(SOURCE_SPECS["guide_calls"]["path"])
    if not guide_calls["cell_barcode"].is_unique:
        raise ValueError("guide-call barcodes are duplicated")
    guide_lookup = guide_calls.set_index("cell_barcode")["feature_call"]
    called = metadata["cell"].map(guide_lookup)
    mismatch = metadata.loc[called.to_numpy() != metadata["sgrna"].to_numpy(), "cell"]
    if len(mismatch):
        raise ValueError(f"S14 and GEO guide calls disagree for {len(mismatch)} cells")

    calibration = metadata["donor"].eq("A") & metadata["sg_target"].eq(CONTROL)
    calibration &= metadata["cell"].map(
        lambda cell: zlib.crc32(f"arce-encoder-calibration|{cell}".encode()) % 5 == 0
    )
    scoring = ~calibration
    target_names = sorted(set(metadata["sg_target"]) - {CONTROL})
    support_rows = []
    for donor in DONORS:
        for lineage in LINEAGES:
            for condition in CONDITIONS:
                state = f"{condition}-{lineage}"
                base = scoring & metadata["donor"].eq(donor) & metadata["hash.ID"].eq(state)
                for target in target_names:
                    support_rows.append(
                        {
                            "donor": donor,
                            "lineage": lineage,
                            "condition": condition,
                            "target": target,
                            "cells": int((base & metadata["sg_target"].eq(target)).sum()),
                        }
                    )
    minimum = min(row["cells"] for row in support_rows)
    eligible = [
        target
        for target in target_names
        if min(row["cells"] for row in support_rows if row["target"] == target)
        >= MINIMUM_ARM_CELLS
    ]
    if len(target_names) != 28 or len(eligible) != 28 or minimum != 36:
        raise ValueError("metadata-only target eligibility changed")
    control_support = {}
    for donor in DONORS:
        for lineage in LINEAGES:
            for condition in CONDITIONS:
                state = f"{condition}-{lineage}"
                mask = (
                    scoring
                    & metadata["donor"].eq(donor)
                    & metadata["hash.ID"].eq(state)
                    & metadata["sg_target"].eq(CONTROL)
                )
                control_support[f"{donor}|{lineage}|{condition}"] = int(mask.sum())

    features = _feature_specification()
    audit = {
        "schema": "v2p2r.arce-gse278572-metadata-audit/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata_only": True,
        "cells_after_declared_exclusion": len(metadata),
        "excluded_unassigned_cell": EXCLUDED_CELL,
        "donor_cells": {
            donor: int(metadata["donor"].eq(donor).sum()) for donor in DONORS
        },
        "calibration_cells_donor_a_only": int(calibration.sum()),
        "targets": target_names,
        "eligible_targets": eligible,
        "minimum_target_arm_cells": minimum,
        "target_arm_support": support_rows,
        "control_arm_support_after_calibration_removal": control_support,
        "matched_rna_protein_features": len(features["matched_adts"]),
        "biological_adt_features": len(features["biological_adts"]),
        "sources": sources,
    }
    _write_json(METADATA_AUDIT, audit)
    return audit, metadata


def _compile_extractor(directory: Path) -> Path:
    source = directory / "extract_selected_matrix.cpp"
    executable = directory / "extract_selected_matrix"
    source.write_text(EXTRACTOR_SOURCE)
    subprocess.run(
        ["clang++", "-O3", "-std=c++17", str(source), "-o", str(executable)],
        check=True,
    )
    return executable


def _stream_matrix(
    mapping: Path,
    raw_output: Path,
    extractor: Path,
    *,
    selected_cells: int,
    selected_features: int,
    directory: Path,
) -> dict[str, object]:
    sha_path = directory / "matrix.sha256"
    bytes_path = directory / "matrix.bytes"
    command = f"""
set -euo pipefail
curl --fail --location --retry 5 --retry-all-errors --silent --show-error \
  {shlex.quote(MATRIX_URL)} \
  | tee >(shasum -a 256 > {shlex.quote(str(sha_path))}) \
        >(wc -c > {shlex.quote(str(bytes_path))}) \
  | gzip -dc \
  | {shlex.quote(str(extractor))} {shlex.quote(str(mapping))} \
      {shlex.quote(str(raw_output))} {selected_cells} {selected_features}
wait
"""
    subprocess.run(["/bin/bash", "-c", command], check=True)
    streamed_sha = sha_path.read_text().split()[0]
    streamed_bytes = int(bytes_path.read_text().strip())
    if streamed_bytes != MATRIX_COMPRESSED_BYTES:
        raise ValueError("streamed matrix byte count changed")
    expected_raw = selected_cells * selected_features * np.dtype(np.uint32).itemsize
    if raw_output.stat().st_size != expected_raw:
        raise ValueError("selected raw count matrix has the wrong size")
    return {
        "url": MATRIX_URL,
        "compressed_bytes": streamed_bytes,
        "sha256": streamed_sha,
        "matrix_market": {
            "rows": MATRIX_ROWS,
            "columns": MATRIX_COLUMNS,
            "nonzeros": MATRIX_NONZEROS,
        },
        "retained_cells": selected_cells,
        "retained_features": selected_features,
    }


def _extract_count_cache(
    metadata: pd.DataFrame,
    features: dict[str, object],
    cache_path: Path,
) -> dict[str, object]:
    with gzip.open(SOURCE_SPECS["barcodes"]["path"], "rt") as handle:
        barcodes = [line.rstrip("\n") for line in handle]
    column_lookup = {barcode: index for index, barcode in enumerate(barcodes, start=1)}
    selected_columns = [column_lookup[cell] for cell in metadata["cell"]]

    with tempfile.TemporaryDirectory(prefix="arce_extract_", dir=DATA_DIR) as temporary:
        directory = Path(temporary)
        mapping = directory / "selection.tsv"
        with mapping.open("w") as handle:
            for source, destination in features["row_mapping"]:
                handle.write(f"R\t{source}\t{destination}\n")
            for destination, source in enumerate(selected_columns):
                handle.write(f"C\t{source}\t{destination}\n")
        extractor = _compile_extractor(directory)
        raw = directory / "selected_counts.uint32"
        matrix = _stream_matrix(
            mapping,
            raw,
            extractor,
            selected_cells=len(metadata),
            selected_features=int(features["selected_features"]),
            directory=directory,
        )
        values = np.memmap(
            raw,
            dtype=np.uint32,
            mode="r",
            shape=(len(metadata), int(features["selected_features"])),
        )
        matched = len(features["matched_genes"])
        temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary_cache.open("wb") as handle:
            np.savez_compressed(
                handle,
                cell=metadata["cell"].to_numpy(dtype=str),
                donor=metadata["donor"].to_numpy(dtype=str),
                hash_id=metadata["hash.ID"].to_numpy(dtype=str),
                sgrna=metadata["sgrna"].to_numpy(dtype=str),
                target=metadata["sg_target"].to_numpy(dtype=str),
                ncount_rna=metadata["nCount_RNA"].to_numpy(dtype=np.float64),
                ncount_adt=metadata["nCount_ADT"].to_numpy(dtype=np.float64),
                rna_counts=np.asarray(values[:, :matched]),
                adt_counts=np.asarray(values[:, matched:]),
                matched_genes=np.asarray(features["matched_genes"], dtype=str),
                matched_adts=np.asarray(features["matched_adts"], dtype=str),
                biological_adts=np.asarray(features["biological_adts"], dtype=str),
                protein_positions=np.asarray(features["protein_positions"], dtype=np.int64),
                matrix_sha256=np.asarray(matrix["sha256"]),
            )
        temporary_cache.replace(cache_path)
    return matrix


def _write_donor_a_lock() -> dict[str, object]:
    if DONOR_A_LOCK.exists():
        return json.loads(DONOR_A_LOCK.read_text())
    lock = {
        "schema": "v2p2r.arce-gse278572-donor-a-preprocessing-lock/1",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "donor": "A",
        "donor_b_matrix_values_accessed": False,
        "cell_selection": "deposited final cells assigned donor A",
        "matched_feature_pairs": len(ADT_TO_GENE),
        "normalization": {
            "rna": "log1p(10000 * matched-gene count / deposited nCount_RNA)",
            "protein": "per-cell CLR over 130 biological ADTs, then matched subset",
        },
        "encoder": {
            "calibration": (
                "donor-A non-targeting CRC32 remainder 0 of 5 pooled across "
                "lineage and stimulation"
            ),
            "pca_components": 6,
            "states": STATE_COUNT,
            "kmeans_n_init": 50,
            "seed": SEED,
        },
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256_at_lock": _sha256(Path(__file__).resolve()),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256_at_lock": _sha256(PROTOCOL),
    }
    _write_json(DONOR_A_LOCK, lock)
    return lock


def prepare_donor_a_cache() -> dict[str, object]:
    _write_donor_a_lock()
    sources = _ensure_small_sources()
    audit, metadata = _metadata_audit(sources)
    if DONOR_A_CACHE.exists():
        if not DONOR_A_MANIFEST.exists():
            raise ValueError("donor-A cache exists without its preprocessing manifest")
        return json.loads(DONOR_A_MANIFEST.read_text())
    donor_a = metadata[metadata["donor"] == "A"].copy()
    if len(donor_a) != 48_386:
        raise ValueError("donor-A cell count changed")
    features = _feature_specification()
    matrix = _extract_count_cache(donor_a, features, DONOR_A_CACHE)
    manifest = {
        "schema": "v2p2r.arce-gse278572-donor-a-preprocessing/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "donor": "A",
        "donor_b_matrix_values_accessed": False,
        "cells": len(donor_a),
        "matrix": matrix,
        "cache": str(DONOR_A_CACHE.relative_to(ROOT)),
        "cache_bytes": DONOR_A_CACHE.stat().st_size,
        "cache_sha256": _sha256(DONOR_A_CACHE),
        "preprocessing_lock": str(DONOR_A_LOCK.relative_to(ROOT)),
        "preprocessing_lock_sha256": _sha256(DONOR_A_LOCK),
        "metadata_audit": str(METADATA_AUDIT.relative_to(ROOT)),
        "metadata_audit_sha256": _sha256(METADATA_AUDIT),
        "small_sources": sources,
        "eligible_targets": audit["eligible_targets"],
    }
    _write_json(DONOR_A_MANIFEST, manifest)
    return manifest


def prepare_cache() -> dict[str, object]:
    lock = _write_or_validate_lock()
    sources = _ensure_small_sources()
    audit, metadata = _metadata_audit(sources)
    features = _feature_specification()
    if CACHE.exists():
        if not SOURCE_MANIFEST.exists():
            raise ValueError("cache exists without a source manifest")
        return json.loads(SOURCE_MANIFEST.read_text())

    matrix = _extract_count_cache(metadata, features, CACHE)

    manifest = {
        "schema": "v2p2r.arce-gse278572-source-manifest/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preanalysis_lock": str(PREANALYSIS_LOCK.relative_to(ROOT)),
        "preanalysis_lock_sha256": _sha256(PREANALYSIS_LOCK),
        "small_sources": sources,
        "matrix": matrix,
        "cache": str(CACHE.relative_to(ROOT)),
        "cache_bytes": CACHE.stat().st_size,
        "cache_sha256": _sha256(CACHE),
        "metadata_audit": str(METADATA_AUDIT.relative_to(ROOT)),
        "metadata_audit_sha256": _sha256(METADATA_AUDIT),
        "matched_feature_pairs": dict(
            zip(features["matched_adts"], features["matched_genes"], strict=True)
        ),
        "biological_adt_features": features["biological_adts"],
        "lock": lock,
    }
    _write_json(SOURCE_MANIFEST, manifest)
    return manifest


def _stable_calibration(cells: np.ndarray, donor: np.ndarray, target: np.ndarray) -> np.ndarray:
    selected = np.asarray(
        [
            zlib.crc32(f"arce-encoder-calibration|{cell}".encode()) % 5 == 0
            for cell in cells
        ]
    )
    return selected & (donor == "A") & (target == CONTROL)


def _fit_state_encoder(
    rna: np.ndarray,
    protein: np.ndarray,
    calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if int(calibration.sum()) != 1_408:
        raise ValueError("calibration-set cardinality changed")

    def encode(values: np.ndarray, offset: int) -> tuple[np.ndarray, dict[str, object]]:
        center = values[calibration].mean(axis=0)
        scale = values[calibration].std(axis=0)
        scale[scale < 1e-6] = 1.0
        standardized = (values - center) / scale
        pca = PCA(n_components=6, svd_solver="full")
        pca.fit(standardized[calibration])
        coordinates = pca.transform(standardized)
        cluster = KMeans(
            n_clusters=STATE_COUNT,
            n_init=50,
            random_state=SEED + offset,
        )
        cluster.fit(coordinates[calibration])
        states = cluster.predict(coordinates).astype(np.int64)
        return states, {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "calibration_state_counts": np.bincount(
                states[calibration], minlength=STATE_COUNT
            ).tolist(),
        }

    first_state, first = encode(rna, 101)
    second_state, second = encode(protein, 102)
    return first_state, second_state, {
        "calibration_cells": int(calibration.sum()),
        "calibration_definition": (
            "donor-A non-targeting CRC32 remainder 0 of 5, pooled across lineage "
            "and stimulation"
        ),
        "rna": first,
        "protein": second,
    }


def _arm_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    indices = np.flatnonzero(mask)
    if len(indices) < MINIMUM_ARM_CELLS:
        raise ValueError("arm fell below its frozen cell threshold")
    first_values = first[indices]
    second_values = second[indices]
    table = np.bincount(
        first_values * STATE_COUNT + second_values,
        minlength=STATE_COUNT**2,
    ).reshape(STATE_COUNT, STATE_COUNT)
    probability = table / table.sum()
    basis = helmert_contrast(STATE_COUNT)
    covariance_table = probability - np.outer(
        probability.sum(axis=1), probability.sum(axis=0)
    )
    covariance = (basis.T @ covariance_table @ basis).ravel()
    endpoint = np.concatenate(
        (basis.T @ probability.sum(axis=1), basis.T @ probability.sum(axis=0))
    )
    estimate = conditional_association_coordinates(
        first_values,
        second_values,
        first_levels=STATE_COUNT,
        second_levels=STATE_COUNT,
        pseudocount=PSEUDOCOUNT,
        permutations=NULL_PERMUTATIONS,
        seed=seed,
    )
    return (
        estimate.coordinates.ravel(),
        covariance,
        endpoint,
        estimate.destroyed_coordinates.ravel(),
        estimate.null_variance_coordinates.ravel(),
        len(indices),
    )


def _build_factorial_fields(
    first: np.ndarray,
    second: np.ndarray,
    donor: np.ndarray,
    hash_id: np.ndarray,
    target: np.ndarray,
    scoring: np.ndarray,
    targets: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    outputs = {
        name: np.empty((len(DONORS), len(targets), 8), dtype=float)
        for name in ("field", "covariance", "endpoint", "destroyed", "variance")
    }
    target_support = np.empty(
        (len(DONORS), len(LINEAGES), len(targets), len(CONDITIONS)), dtype=int
    )
    control_support = np.empty(
        (len(DONORS), len(LINEAGES), len(CONDITIONS)), dtype=int
    )
    statistic_names = ("field", "covariance", "endpoint", "destroyed")
    for donor_index, donor_name in enumerate(DONORS):
        for lineage_index, lineage in enumerate(LINEAGES):
            control_statistics = []
            for condition_index, condition in enumerate(CONDITIONS):
                mask = (
                    scoring
                    & (donor == donor_name)
                    & (hash_id == f"{condition}-{lineage}")
                    & (target == CONTROL)
                )
                statistics = _arm_statistics(
                    first,
                    second,
                    mask,
                    SEED
                    + 100_000 * donor_index
                    + 10_000 * lineage_index
                    + 1_000 * condition_index,
                )
                control_statistics.append(statistics)
                control_support[donor_index, lineage_index, condition_index] = statistics[5]
            for target_index, target_name in enumerate(targets):
                target_statistics = []
                for condition_index, condition in enumerate(CONDITIONS):
                    mask = (
                        scoring
                        & (donor == donor_name)
                        & (hash_id == f"{condition}-{lineage}")
                        & (target == target_name)
                    )
                    statistics = _arm_statistics(
                        first,
                        second,
                        mask,
                        SEED
                        + 100_000 * donor_index
                        + 10_000 * lineage_index
                        + 100 * (target_index + 1)
                        + 10 * condition_index,
                    )
                    target_statistics.append(statistics)
                    target_support[
                        donor_index, lineage_index, target_index, condition_index
                    ] = statistics[5]
                for statistic_index, name in enumerate(statistic_names):
                    contrast = (
                        target_statistics[1][statistic_index]
                        - target_statistics[0][statistic_index]
                        - control_statistics[1][statistic_index]
                        + control_statistics[0][statistic_index]
                    )
                    start = 4 * lineage_index
                    outputs[name][donor_index, target_index, start : start + 4] = contrast
                start = 4 * lineage_index
                outputs["variance"][
                    donor_index, target_index, start : start + 4
                ] = (
                    target_statistics[1][4]
                    + target_statistics[0][4]
                    + control_statistics[1][4]
                    + control_statistics[0][4]
                )
    return outputs, {
        "target_support": target_support,
        "control_support": control_support,
        "minimum_target_arm_cells": int(target_support.min()),
        "minimum_control_arm_cells": int(control_support.min()),
    }


def _scgpt_laplacian(targets: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(SCGPT_EMBEDDING, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=float)
    lookup = {name.upper(): index for index, name in enumerate(names)}
    missing = [target for target in targets if target.upper() not in lookup]
    if missing:
        raise ValueError(f"scGPT embedding lacks confirmation targets: {missing}")
    values = np.asarray([embedding[lookup[target.upper()]] for target in targets])
    norm = np.linalg.norm(values, axis=1)
    values = values / norm[:, None]
    similarity = values @ values.T
    incidence = np.zeros((len(targets), len(targets)), dtype=float)
    for index in range(len(targets)):
        order = np.argsort(similarity[index])[::-1]
        incidence[order[: HYPERGRAPH_NEIGHBORS + 1], index] = 1.0
    return normalized_hypergraph_laplacian(incidence), {
        "source": str(SCGPT_EMBEDDING.relative_to(ROOT)),
        "source_sha256": _sha256(SCGPT_EMBEDDING),
        "covered_targets": len(targets),
        "hyperedges": len(targets),
        "neighbors_excluding_self": HYPERGRAPH_NEIGHBORS,
    }


def _field_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    estimate_flat = estimate.ravel()
    truth_flat = truth.ravel()
    correlation = (
        float(np.corrcoef(estimate_flat, truth_flat)[0, 1])
        if np.std(estimate_flat) > 0.0 and np.std(truth_flat) > 0.0
        else None
    )
    estimate_norm = np.linalg.norm(estimate, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    cosine = np.divide(
        np.sum(estimate * truth, axis=1),
        estimate_norm * truth_norm,
        out=np.zeros_like(estimate_norm),
        where=estimate_norm * truth_norm > 0.0,
    )
    standardized_rmse = float(
        np.sqrt(
            np.mean((estimate - truth) ** 2)
            / max(float(np.mean(truth**2)), 1e-12)
        )
    )
    return {
        "pooled_pearson": correlation,
        "macro_target_cosine": float(cosine.mean()),
        "standardized_rmse": standardized_rmse,
    }


def _bootstrap_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
) -> dict[str, list[float] | None]:
    samples = {name: [] for name in _field_metrics(estimate, truth)}
    for draw in indices:
        metrics = _field_metrics(estimate[draw], truth[draw])
        for name, value in metrics.items():
            if value is not None:
                samples[name].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(values, [0.025, 0.975])]
            if values
            else None
        )
        for name, values in samples.items()
    }


def _support_records(values: np.ndarray, targets: list[str]) -> list[dict[str, object]]:
    records = []
    for donor_index, donor in enumerate(DONORS):
        for lineage_index, lineage in enumerate(LINEAGES):
            for target_index, target in enumerate(targets):
                for condition_index, condition in enumerate(CONDITIONS):
                    records.append(
                        {
                            "donor": donor,
                            "lineage": lineage,
                            "condition": condition,
                            "target": target,
                            "cells": int(
                                values[
                                    donor_index,
                                    lineage_index,
                                    target_index,
                                    condition_index,
                                ]
                            ),
                        }
                    )
    return records


def run_confirmation(output: Path = OUTPUT) -> dict[str, object]:
    lock = _write_or_validate_lock()
    if not CACHE.exists() or not SOURCE_MANIFEST.exists():
        prepare_cache()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest["cache_sha256"] != _sha256(CACHE):
        raise ValueError("confirmation cache hash changed")

    with np.load(CACHE, allow_pickle=False) as cache:
        cells = cache["cell"].astype(str)
        donor = cache["donor"].astype(str)
        hash_id = cache["hash_id"].astype(str)
        target = cache["target"].astype(str)
        ncount_rna = np.asarray(cache["ncount_rna"], dtype=float)
        rna_counts = np.asarray(cache["rna_counts"], dtype=np.float32)
        adt_counts = np.asarray(cache["adt_counts"], dtype=np.float32)
        matched_genes = cache["matched_genes"].astype(str)
        matched_adts = cache["matched_adts"].astype(str)
        biological_adts = cache["biological_adts"].astype(str)
        protein_positions = np.asarray(cache["protein_positions"], dtype=int)
        matrix_sha = str(cache["matrix_sha256"])
    if matrix_sha != manifest["matrix"]["sha256"]:
        raise ValueError("cache and source manifest matrix hashes differ")
    if np.any(ncount_rna <= 0.0):
        raise ValueError("RNA normalization denominator is nonpositive")

    rna = np.log1p(rna_counts / ncount_rna[:, None] * 10_000.0)
    protein_clr = np.log1p(adt_counts)
    protein_clr -= protein_clr.mean(axis=1, keepdims=True)
    protein = protein_clr[:, protein_positions]
    if not (np.isfinite(rna).all() and np.isfinite(protein).all()):
        raise FloatingPointError("normalized assay values are non-finite")

    calibration = _stable_calibration(cells, donor, target)
    first_state, second_state, encoder = _fit_state_encoder(
        rna, protein, calibration
    )
    scoring = ~calibration
    targets = sorted(set(target) - {CONTROL})
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        arrays, support = _build_factorial_fields(
            first_state,
            second_state,
            donor,
            hash_id,
            target,
            scoring,
            targets,
        )
        laplacian, graph = _scgpt_laplacian(targets)
        donor_a = arrays["field"][0]
        truth = arrays["field"][1]
        variance = arrays["variance"][0]
        if not np.isfinite(variance).all() or np.any(variance <= 0.0):
            raise FloatingPointError("factorial null variance is nonpositive or non-finite")
        observation_weight = inverse_permutation_variance_weights(variance)
        singular_maximum = float(np.linalg.svd(donor_a, compute_uv=False)[0])
        structured_fit = fit_structured_coupling_fields(
            donor_a,
            observation_weight=observation_weight,
            graph_laplacian=laplacian,
            nuclear_penalty=STRUCTURED_NUCLEAR_FRACTION * singular_maximum,
            graph_penalty=STRUCTURED_GRAPH_PENALTY,
            tolerance=1e-9,
        )
        endpoint_model = Ridge(alpha=ENDPOINT_RIDGE_PENALTY).fit(
            arrays["endpoint"][0], donor_a
        )
        predictions = {
            "direct": donor_a,
            "fixed_structured": structured_fit.coefficient,
            "endpoint_ridge": endpoint_model.predict(arrays["endpoint"][1]),
            "linear_cross_covariance": arrays["covariance"][0],
            "destroyed_links": arrays["destroyed"][0],
        }
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} prediction is non-finite")

    rng = np.random.default_rng(SEED + 900_000)
    bootstrap_indices = rng.integers(0, len(targets), size=(BOOTSTRAPS, len(targets)))
    methods = {}
    for name, estimate in predictions.items():
        methods[name] = {
            "metrics": _field_metrics(estimate, truth),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                estimate, truth, bootstrap_indices
            ),
            "lineage_metrics": {
                "Treg": _field_metrics(estimate[:, :4], truth[:, :4]),
                "Teff": _field_metrics(estimate[:, 4:], truth[:, 4:]),
            },
        }

    paired = {}
    structured_error = np.mean((predictions["fixed_structured"] - truth) ** 2, axis=1)
    for name, estimate in predictions.items():
        if name == "fixed_structured":
            continue
        comparator_error = np.mean((estimate - truth) ** 2, axis=1)
        difference = structured_error - comparator_error
        draws = np.mean(difference[bootstrap_indices], axis=1)
        paired[name] = {
            "mean_structured_minus_comparator_target_mse": float(difference.mean()),
            "target_bootstrap_95_ci": [
                float(value) for value in np.quantile(draws, [0.025, 0.975])
            ],
        }

    structured_metrics = methods["fixed_structured"]["metrics"]
    rmse_wins = {
        name: bool(
            structured_metrics["standardized_rmse"]
            < methods[name]["metrics"]["standardized_rmse"]
        )
        for name in predictions
        if name != "fixed_structured"
    }
    interval_wins = {
        name: bool(values["target_bootstrap_95_ci"][1] < 0.0)
        for name, values in paired.items()
    }
    positive_correlation = bool(
        structured_metrics["pooled_pearson"] is not None
        and structured_metrics["pooled_pearson"] > 0.0
    )
    passed = bool(
        positive_correlation and all(rmse_wins.values()) and all(interval_wins.values())
    )

    result = {
        "schema": "v2p2r.arce-gse278572-conditional-field-confirmation/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preanalysis_contract": str(PROTOCOL.relative_to(ROOT)),
        "preanalysis_contract_sha256": _sha256(PROTOCOL),
        "preanalysis_lock": str(PREANALYSIS_LOCK.relative_to(ROOT)),
        "preanalysis_lock_sha256": _sha256(PREANALYSIS_LOCK),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "source_matrix_url": MATRIX_URL,
        "source_matrix_sha256": matrix_sha,
        "source_matrix_compressed_bytes": MATRIX_COMPRESSED_BYTES,
        "replication": {
            "biological_donors": 2,
            "development_donor": "A",
            "analytically_held_out_confirmation_donor": "B",
            "literal_byte_level_unopened_before_corrected_lock": False,
            "donor_b_statistic_state_field_or_score_before_corrected_lock": False,
            "target_bootstrap_units": len(targets),
            "cells_are_not_biological_replicates": True,
        },
        "estimand": (
            "target-minus-nontargeting change in stimulated-minus-resting "
            "conditionally centered RNA-protein cycle-space coordinates, "
            "concatenated across Treg and Teff"
        ),
        "targets": targets,
        "field_coordinates": 8,
        "matched_features": {
            "count": len(matched_genes),
            "rna": matched_genes.tolist(),
            "protein": matched_adts.tolist(),
            "protein_clr_reference_features": len(biological_adts),
        },
        "encoder": encoder,
        "support": {
            "minimum_target_arm_cells": support["minimum_target_arm_cells"],
            "minimum_control_arm_cells": support["minimum_control_arm_cells"],
            "target_arms": _support_records(support["target_support"], targets),
            "control_arms": {
                f"{donor}|{lineage}|{condition}": int(
                    support["control_support"][donor_index, lineage_index, condition_index]
                )
                for donor_index, donor in enumerate(DONORS)
                for lineage_index, lineage in enumerate(LINEAGES)
                for condition_index, condition in enumerate(CONDITIONS)
            },
        },
        "fixed_structured_fit": {
            "nuclear_fraction_of_largest_donor_a_singular_value": STRUCTURED_NUCLEAR_FRACTION,
            "nuclear_penalty": structured_fit.nuclear_penalty,
            "graph_penalty": structured_fit.graph_penalty,
            "effective_rank": structured_fit.effective_rank,
            "iterations": structured_fit.iterations,
            "converged": structured_fit.converged,
            "factorial_variance_definition": (
                "sum of the four arm-level conditional-null coordinate variances"
            ),
            "observation_weight_definition": (
                "clip((1 / variance) / median(1 / variance), 0.05, 20)"
            ),
            "observation_weight_minimum": float(observation_weight.min()),
            "observation_weight_median": float(np.median(observation_weight)),
            "observation_weight_maximum": float(observation_weight.max()),
            "graph": graph,
        },
        "endpoint_ridge_penalty": ENDPOINT_RIDGE_PENALTY,
        "methods": methods,
        "paired_target_error_comparisons": paired,
        "confirmation_gate": {
            "positive_structured_pooled_correlation": positive_correlation,
            "structured_rmse_lower_than_each_comparator": rmse_wins,
            "paired_target_bootstrap_upper_below_zero": interval_wins,
            "passed": passed,
        },
        "coordinates": {
            "truth_donor_b": truth.tolist(),
            "predictions": {name: values.tolist() for name, values in predictions.items()},
            "destroyed_donor_b": arrays["destroyed"][1].tolist(),
            "factorial_null_variance_donor_a": variance.tolist(),
            "observation_weight_donor_a": observation_weight.tolist(),
        },
        "scope": {
            "identified": (
                "guide-by-restimulation factorial coupling across same-cell RNA and "
                "protein with donor-held confirmation"
            ),
            "not_identified": [
                "dual-pulse chronology",
                "molecular age",
                "lineage tracing",
                "population-level donor inference",
            ],
            "aborted_stream_disclosure": (
                "donor-B raw counts entered an interrupted prior process buffer, "
                "but no donor-B summary, state, field, score, cache, result, or "
                "human-visible value informed any estimator choice"
            ),
        },
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "estimator_sha256": _sha256(ROOT / "mapreg/coupling_fields.py"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "seed": SEED,
            "state_count": STATE_COUNT,
            "pseudocount": PSEUDOCOUNT,
            "conditional_null_permutations": NULL_PERMUTATIONS,
            "target_bootstrap_draws": BOOTSTRAPS,
        },
        "lock": lock,
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--prepare-donor-a-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()

    if arguments.metadata_only:
        sources = _ensure_small_sources()
        audit, _ = _metadata_audit(sources)
        print(
            json.dumps(
                {
                    "metadata_audit": str(METADATA_AUDIT),
                    "eligible_targets": len(audit["eligible_targets"]),
                    "minimum_target_arm_cells": audit["minimum_target_arm_cells"],
                },
                indent=2,
            )
        )
        return

    if arguments.prepare_donor_a_only:
        manifest = prepare_donor_a_cache()
        print(
            json.dumps(
                {
                    "cache": str(DONOR_A_CACHE),
                    "cache_sha256": manifest["cache_sha256"],
                    "matrix_sha256": manifest["matrix"]["sha256"],
                    "donor_b_matrix_values_accessed": False,
                },
                indent=2,
            )
        )
        return

    _write_or_validate_lock()
    manifest = prepare_cache()
    if arguments.prepare_only:
        print(
            json.dumps(
                {
                    "cache": str(CACHE),
                    "cache_sha256": manifest["cache_sha256"],
                    "matrix_sha256": manifest["matrix"]["sha256"],
                },
                indent=2,
            )
        )
        return

    result = run_confirmation(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "matrix_sha256": result["source_matrix_sha256"],
                "fixed_structured": result["methods"]["fixed_structured"]["metrics"],
                "confirmation_gate": result["confirmation_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
