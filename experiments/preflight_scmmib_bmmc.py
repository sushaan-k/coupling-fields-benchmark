"""Metadata-only preflight for a held-site BMMC CITE-seq confirmation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


METADATA_SHA256 = "b267d4a820b062d0a05227c9cab61d389dcf924c3a6e062fb2389ce1be2f6e4f"
FEATURE_FILE_SHA256 = "322c30a7a4905f7f113472442d4aa2c81a1ad736c86651f6c0b81e5b2ff94ac8"
BRIDGE_DONOR = "15078"
HELD_SITE = "site3"
DONOR_SPLIT_SALT = "GSE194122:SCMMIB-v2:"
LOCKED_MARKERS = [
    "CD4",
    "CD7",
    "CD14",
    "CD19",
    "CD33",
    "CD38",
    "CD44",
    "CD47",
    "CD52",
    "CD93",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "cells": len(rows),
        "sites": _sorted_counts([row["Site"] for row in rows]),
        "donors": _sorted_counts([row["DonorID"] for row in rows]),
        "batches": _sorted_counts([row["batch"] for row in rows]),
        "lineages": _sorted_counts([row["cell_type.l1"] for row in rows]),
        "original_partition_flags": _sorted_counts([row["is_train"] for row in rows]),
    }


def build_preflight(metadata_path: Path, feature_path: Path) -> dict[str, Any]:
    """Build the deterministic preflight without reading expression counts."""

    metadata_hash = _sha256(metadata_path)
    feature_hash = _sha256(feature_path)
    if metadata_hash != METADATA_SHA256:
        raise ValueError("metadata SHA-256 does not match the pinned source")
    if feature_hash != FEATURE_FILE_SHA256:
        raise ValueError("feature-file SHA-256 does not match the pinned source")

    with gzip.open(metadata_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {
        "Site",
        "DonorID",
        "batch",
        "barcode",
        "cell_type.l1",
        "is_train",
    }
    if not required.issubset(fields):
        raise ValueError("metadata is missing required split columns")
    if len({row["barcode"] for row in rows}) != len(rows):
        raise ValueError("metadata barcodes are not unique")

    fit_rows = [
        row
        for row in rows
        if row["Site"] in {"site1", "site2"} and row["DonorID"] != BRIDGE_DONOR
    ]
    development_rows = [
        row for row in rows if row["Site"] == "site4" and row["DonorID"] != BRIDGE_DONOR
    ]
    held_rows = [row for row in rows if row["Site"] == HELD_SITE]
    excluded_rows = [
        row
        for row in rows
        if row["DonorID"] == BRIDGE_DONOR and row["Site"] != HELD_SITE
    ]
    split_rows = {
        "fit": fit_rows,
        "development": development_rows,
        "held": held_rows,
        "excluded_bridge_elsewhere": excluded_rows,
    }
    split_donors = {
        name: {row["DonorID"] for row in selected}
        for name, selected in split_rows.items()
    }
    split_sites = {
        name: {row["Site"] for row in selected} for name, selected in split_rows.items()
    }
    if split_donors["fit"] & split_donors["development"]:
        raise ValueError("fit and development donors overlap")
    if split_donors["fit"] & split_donors["held"]:
        raise ValueError("fit and held donors overlap")
    if split_donors["development"] & split_donors["held"]:
        raise ValueError("development and held donors overlap")
    if split_sites["fit"] & split_sites["development"]:
        raise ValueError("fit and development sites overlap")
    if split_sites["fit"] & split_sites["held"]:
        raise ValueError("fit and held sites overlap")
    if split_sites["development"] & split_sites["held"]:
        raise ValueError("development and held sites overlap")

    donor_sites = {
        donor: {row["Site"] for row in rows if row["DonorID"] == donor}
        for donor in {row["DonorID"] for row in rows}
    }
    bridge_donors = sorted(
        donor for donor, sites in donor_sites.items() if len(sites) > 1
    )
    if bridge_donors != [BRIDGE_DONOR]:
        raise ValueError("expected one four-site bridge donor")
    site_specific_donors = sorted(
        (donor for donor in donor_sites if donor != BRIDGE_DONOR),
        key=lambda donor: hashlib.sha256(
            f"{DONOR_SPLIT_SALT}{donor}".encode()
        ).hexdigest(),
    )
    if len(site_specific_donors) != 8:
        raise ValueError("expected eight site-specific physical donors")
    donor_held = site_specific_donors[:6]
    donor_fit = site_specific_donors[6:]
    donor_development = [BRIDGE_DONOR]
    donor_split_rows = {
        "fit": [row for row in rows if row["DonorID"] in donor_fit],
        "development": [row for row in rows if row["DonorID"] in donor_development],
        "held": [row for row in rows if row["DonorID"] in donor_held],
    }
    donor_split_donors = {
        name: {row["DonorID"] for row in selected}
        for name, selected in donor_split_rows.items()
    }
    donor_split_sites = {
        name: {row["Site"] for row in selected}
        for name, selected in donor_split_rows.items()
    }
    if set().union(*donor_split_donors.values()) != set(donor_sites):
        raise ValueError("donor-powered split does not assign every donor once")
    if sum(map(len, donor_split_donors.values())) != len(donor_sites):
        raise ValueError("donor-powered split leaks a donor between roles")
    all_lineages = {row["cell_type.l1"] for row in rows}
    if any(
        {row["cell_type.l1"] for row in selected} != all_lineages
        for selected in donor_split_rows.values()
    ):
        raise ValueError("a donor-powered role is missing a broad lineage")

    with h5py.File(feature_path, "r") as handle:
        feature_group = handle["matrix/features"]
        names = [value.decode() for value in feature_group["name"][...]]
        feature_types = [value.decode() for value in feature_group["feature_type"][...]]
    genes = {
        name
        for name, feature_type in zip(names, feature_types)
        if feature_type == "Gene Expression"
    }
    antibodies = [
        name
        for name, feature_type in zip(names, feature_types)
        if feature_type == "Antibody Capture"
    ]
    isotype_controls = sorted(name for name in antibodies if "IgG" in name)
    biological_antibodies = sorted(
        name for name in antibodies if name not in isotype_controls
    )
    exact_matches = sorted(set(biological_antibodies) & genes)
    if not set(LOCKED_MARKERS).issubset(exact_matches):
        raise ValueError("the locked marker panel is not present in both modalities")

    held_donor_count = len(split_donors["held"])
    return {
        "schema_version": "scmmib_bmmc_metadata_preflight_v1",
        "status": "PREFLIGHT_ELIGIBLE_NOT_FROZEN",
        "dataset": {
            "name": "NeurIPS 2021 BMMC CITE-seq",
            "modalities": ["RNA", "ADT"],
            "processed_cells_reported": 90261,
            "physical_donor_structure": (
                "eight site-specific donors plus donor 15078 processed at all four sites"
            ),
        },
        "sources": {
            "scmmib_archive": {
                "figshare_article": "https://figshare.com/articles/dataset/27161451/2",
                "file_url": "https://ndownloader.figshare.com/files/49589406",
                "bytes": 1771132942,
                "md5": "e25a6ac213ddea23dbab1f5ee8caefa9",
                "downloaded": False,
            },
            "metadata": {
                "url": (
                    "https://raw.githubusercontent.com/bm2-lab/SCMMI_Benchmark/"
                    "5341740b541c9d8050fb74009c1605aa1bd1b27a/"
                    "manuscript_figure_script_and_data/stage2_res/metadata/"
                    "BMMC_RNA%2BADT_p10_metadata.csv.gz"
                ),
                "repository_commit": "5341740b541c9d8050fb74009c1605aa1bd1b27a",
                "bytes": metadata_path.stat().st_size,
                "sha256": metadata_hash,
                "decoded_rows": len(rows),
            },
            "feature_panel": {
                "figshare_article": "https://figshare.com/articles/dataset/22716739",
                "file_url": "https://ndownloader.figshare.com/files/40347877",
                "sample": "s1d1",
                "bytes": feature_path.stat().st_size,
                "md5": "a99285913ea3f3d22600d3d2f8a88e34",
                "sha256": feature_hash,
            },
            "license": "CC BY 4.0",
            "geo_accession": "GSE194122",
        },
        "metadata_snapshot": {
            "all": _summarize(rows),
            "split": {
                name: _summarize(selected) for name, selected in split_rows.items()
            },
        },
        "prospective_split": {
            "fit": {
                "sites": sorted(split_sites["fit"]),
                "donors": sorted(split_donors["fit"]),
                "role": "fit candidate parameters on sites 1 and 2",
            },
            "development": {
                "sites": sorted(split_sites["development"]),
                "donors": sorted(split_donors["development"]),
                "role": "select penalties and gates on site 4 only",
            },
            "held": {
                "sites": sorted(split_sites["held"]),
                "donors": sorted(split_donors["held"]),
                "role": "score site 3 exactly once after public freeze",
            },
            "excluded_bridge_elsewhere": {
                "sites": sorted(split_sites["excluded_bridge_elsewhere"]),
                "donors": sorted(split_donors["excluded_bridge_elsewhere"]),
                "reason": "reserve physical donor 15078 to the held site",
            },
            "donor_disjoint": True,
            "site_disjoint": True,
            "original_partition_flags_used": False,
        },
        "donor_powered_alternative": {
            "status": "ELIGIBLE_WITH_SINGLE_DONOR_DEVELOPMENT",
            "allocation_rule": (
                "reserve the sole four-site bridge donor for development; sort "
                "the eight site-specific donors by SHA-256 of "
                f"'{DONOR_SPLIT_SALT}' plus DonorID; hold the first six and fit "
                "the remaining two"
            ),
            "hash_salt": DONOR_SPLIT_SALT,
            "fit": {
                "sites": sorted(donor_split_sites["fit"]),
                "donors": sorted(donor_split_donors["fit"]),
                "summary": _summarize(donor_split_rows["fit"]),
            },
            "development": {
                "sites": sorted(donor_split_sites["development"]),
                "donors": sorted(donor_split_donors["development"]),
                "summary": _summarize(donor_split_rows["development"]),
            },
            "held": {
                "sites": sorted(donor_split_sites["held"]),
                "donors": sorted(donor_split_donors["held"]),
                "summary": _summarize(donor_split_rows["held"]),
            },
            "physical_donor_disjoint": True,
            "site_disjoint": False,
            "site_overlap_declared": True,
            "original_partition_flags_used": False,
            "all_roles_have_all_broad_lineages": True,
            "held_physical_donors": 6,
            "minimum_two_sided_exact_sign_p": 2.0 / (2**6),
            "minimum_one_sided_exact_sign_p": 1.0 / (2**6),
            "selection": (
                "fit on the two fit donors; choose every penalty and gate on "
                "bridge donor 15078; after locking, refit the three non-held "
                "donors and score the six held donors once"
            ),
            "held_access": (
                "feature-level outcome arrays for all six held donors remain "
                "unopened until a public freeze"
            ),
            "allocation_limit": (
                "nine physical donors cannot supply six held donors and at least "
                "two physical donors to both fit and development"
            ),
        },
        "marker_preflight": {
            "raw_antibody_features": len(antibodies),
            "isotype_controls": isotype_controls,
            "biological_antibody_features": len(biological_antibodies),
            "exact_rna_adt_name_matches": len(exact_matches),
            "exact_matches": exact_matches,
            "locked_biology_only_panel": LOCKED_MARKERS,
            "locked_ordered_pairs": len(LOCKED_MARKERS) ** 2,
        },
        "planned_analysis": {
            "primary_parameter": "exact conditional full log odds",
            "primary_estimator": "structured noncentral-hypergeometric likelihood",
            "target_prediction": "exact expected 2x2 table at held donor margins",
            "comparators": [
                "independence",
                "null-centered Haldane Paule-Mandel ablation",
                "one-df Pearson coordinate",
                "one-df signed-deviance coordinate",
            ],
            "selection": (
                "lock the marker panel from feature metadata; choose penalties and "
                "all performance gates on site 4; refit sites 1, 2, and 4 only"
            ),
            "held_access": "site 3 outcome arrays remain unopened until public freeze",
        },
        "inference_limit": {
            "held_physical_donors": held_donor_count,
            "minimum_two_sided_exact_sign_p": 2.0 / (2**held_donor_count),
            "minimum_one_sided_exact_sign_p": 1.0 / (2**held_donor_count),
            "conclusion": (
                "eligible as a held-site performance confirmation, but incapable "
                "of donor-level exact p <= 0.025"
            ),
        },
        "access_audit": {
            "decoded": [
                (
                    "SCMMIB p10 metadata CSV, including deposited per-cell QC "
                    "summaries but no feature-level values"
                ),
                "s1d1 matrix/features/name",
                "s1d1 matrix/features/feature_type",
            ],
            "opaque_byte_reads": [
                "complete metadata gzip for SHA-256",
                "complete s1d1 HDF5 file for SHA-256",
            ],
            "not_decoded": [
                "s1d1 matrix/data",
                "s1d1 matrix/indices",
                "s1d1 matrix/indptr",
                "all site 3 feature-level RNA outcome counts",
                "all site 3 feature-level ADT outcome counts",
                "all donor-powered held feature-level RNA outcome counts",
                "all donor-powered held feature-level ADT outcome counts",
                "any held 2x2 table or held coupling statistic",
            ],
            "covid_sanger_touched": False,
            "large_scmmib_archive_downloaded": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preflight(args.metadata, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
