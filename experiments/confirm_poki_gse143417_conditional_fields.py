"""Prospective held-donor confirmation on the GSE143417 PoKI-seq screen."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import mmread
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapreg.classical_residuals import poisson_independence_residuals  # noqa: E402
from mapreg.coupling_fields import (  # noqa: E402
    ConditionalAssociationEstimate,
    association_coordinates,
    association_field,
    fit_structured_coupling_fields,
    helmert_contrast,
    inverse_permutation_variance_weights,
    normalized_hypergraph_laplacian,
)
from mapreg.table_prediction import (  # noqa: E402
    field_coordinates_to_table,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


DATA_DIR = ROOT / "data/confirmation/gse143417_pokiseq"
RAW_TAR = DATA_DIR / "GSE143417_RAW.tar"
CACHE = DATA_DIR / "poki_marker_cells_v1.npz"
LOCK = DATA_DIR / "preanalysis_lock_v1.json"
OUTPUT = ROOT / "results/gse143417_pokiseq_held_donor_confirmation.json"
PREDICTIONS = ROOT / "results/gse143417_pokiseq_pretruth_predictions.json"
PRETRUTH_DESIGNATION = DATA_DIR / "pretruth_prediction_designation_v1.json"
PROTOCOL_PATH = (
    ROOT / "docs/GSE143417_POKISEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-27.md"
)

SEED = 20260827
STATE_COUNT = 3
PSEUDOCOUNT = 0.5
NULL_PERMUTATIONS = 64
BOOTSTRAPS = 2_000
MINIMUM_ARM_CELLS = 30
MINIMUM_STATE_FRACTION = 0.05
EXPECTED_LIBRARY_SIZE = 36
FINAL_NUCLEAR_FRACTION = 0.1
FINAL_GRAPH_PENALTY = 5.0
DEVELOPMENT_DONOR = "Donor1"
HELD_DONOR = "Donor2"
STIM = "Stim"
TGFB = "TGFB"
CONTROL_CONSTRUCTS = ("GFP", "MCHERRY")
EXPECTED_METADATA_EXCLUSION = "CTLA4DN"
PROLIFERATION_GENES = ("MKI67", "TOP2A", "CDK1")
EFFECTOR_GENES = ("GZMA", "FASLG", "IFNG")
MARKER_GENES = PROLIFERATION_GENES + EFFECTOR_GENES
POSITIVE_CONTROL_CLASS = ("TGFBR2DN", "TGFBR241BB", "TGFBR2MYD88")
FALSIFICATION_TARGET = "TNGFR"
RAW_BYTES = 492_216_320
RAW_SHA256 = "6bc8bf810fbca8f0585c337ed143d39d8bfbc3f85d623894ebadf4c6f357b632"

ARCHITECTURE_COMPONENTS = {
    "receptor:BTLA": ("BTLA",),
    "receptor:FAS": ("FAS",),
    "receptor:PD1": ("PD1", "PDCD1"),
    "receptor:TGFBR2": ("TGFBR2",),
    "receptor:TIGIT": ("TIGIT",),
    "receptor:TIM3": ("TIM3", "HAVCR2"),
    "receptor:CTLA4": ("CTLA4",),
    "receptor:IL7R": ("IL7R", "IL7RA"),
    "domain:DN": ("DN",),
    "domain:CD28": ("CD28",),
    "domain:41BB": ("41BB", "CD137", "TNFRSF9"),
    "domain:ICOS": ("ICOS",),
    "domain:MYD88": ("MYD88",),
    "domain:IL7RA": ("IL7RA",),
    "domain:OX40": ("OX40", "TNFRSF4"),
    "domain:CD3Z": ("CD3Z", "CD247"),
}

# The accessions and donor/context labels come from GEO metadata only. GEX and
# barcode calls are paired by replicate; the runner never pools across donors.
SAMPLES = (
    ("Donor1", "Stim", 1, "GSM4259039", "GSM4259040"),
    ("Donor1", "Stim", 2, "GSM4259041", "GSM4259042"),
    ("Donor1", "Stim", 3, "GSM4259043", "GSM4259044"),
    ("Donor1", "TGFB", 1, "GSM4259045", "GSM4259046"),
    ("Donor1", "TGFB", 2, "GSM4259047", "GSM4259048"),
    ("Donor1", "TGFB", 3, "GSM4259049", "GSM4259050"),
    ("Donor2", "Stim", 1, "GSM4259051", "GSM4259052"),
    ("Donor2", "Stim", 2, "GSM4259053", "GSM4259054"),
    ("Donor2", "Stim", 3, "GSM4259055", "GSM4259056"),
    ("Donor2", "TGFB", 1, "GSM4259057", "GSM4259058"),
    ("Donor2", "TGFB", 2, "GSM4259059", "GSM4259060"),
    ("Donor2", "TGFB", 3, "GSM4259061", "GSM4259062"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "mapreg/coupling_fields.py",
        ROOT / "mapreg/classical_residuals.py",
        ROOT / "mapreg/table_prediction.py",
        PROTOCOL_PATH,
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _canonical_construct(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().replace("Β", "B"))


def _terminal_barcode(value: str) -> str | None:
    matches = re.findall(r"(?<![ACGT])[ACGT]{16}(?![ACGT])", str(value).upper())
    return matches[-1] if matches else None


def _read_member_bytes(archive: tarfile.TarFile, accession: str, suffix: str) -> bytes:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and Path(member.name).name.startswith(accession + "_")
        and Path(member.name).name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {accession} {suffix} member, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise ValueError(f"could not read {matches[0].name}")
    return handle.read()


def _uncompressed_member(archive: tarfile.TarFile, patterns: tuple[str, ...]) -> bytes:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and any(Path(member.name).name.endswith(pattern) for pattern in patterns)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one nested member matching {patterns}, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise ValueError(f"could not read {matches[0].name}")
    payload = handle.read()
    return gzip.decompress(payload) if matches[0].name.endswith(".gz") else payload


def _read_gex(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        matrix_bytes = _uncompressed_member(archive, ("matrix.mtx", "matrix.mtx.gz"))
        genes_bytes = _uncompressed_member(
            archive,
            ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"),
        )
        barcodes_bytes = _uncompressed_member(
            archive, ("barcodes.tsv", "barcodes.tsv.gz")
        )
    features = list(csv.reader(io.StringIO(genes_bytes.decode()), delimiter="\t"))
    genes = np.asarray([row[1] if len(row) > 1 else row[0] for row in features])
    barcodes = np.asarray(
        [row[0] for row in csv.reader(io.StringIO(barcodes_bytes.decode()), delimiter="\t")]
    )
    matrix = mmread(io.BytesIO(matrix_bytes)).tocsr()
    if matrix.shape == (len(barcodes), len(genes)):
        matrix = matrix.T.tocsr()
    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError("GEX matrix dimensions do not match genes and barcodes")
    if len(genes) != 20_953:
        raise ValueError(f"expected the frozen 20,953-row feature space, found {len(genes)}")
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, gene in enumerate(genes):
        lookup[str(gene).upper()].append(index)
    missing = [gene for gene in MARKER_GENES if gene not in lookup]
    if missing:
        raise ValueError(f"frozen marker genes are absent: {missing}")
    counts = np.column_stack(
        [np.asarray(matrix[lookup[gene]].sum(axis=0)).ravel() for gene in MARKER_GENES]
    )
    library_size = np.asarray(matrix.sum(axis=0)).ravel()
    return barcodes.astype(str), counts.astype(float), library_size.astype(float)


def _read_construct_calls(payload: bytes) -> tuple[dict[str, str], set[str]]:
    text = gzip.decompress(payload).decode()
    grouped: dict[str, list[str]] = defaultdict(list)
    roster: set[str] = set()
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(row) < 2:
            continue
        barcode_column = next((_terminal_barcode(value) for value in row if _terminal_barcode(value)), None)
        if barcode_column is None:
            continue
        candidates = [value for value in row if _terminal_barcode(value) is None]
        if not candidates:
            continue
        for call in re.split(r"[,;|]", candidates[-1]):
            construct = call.strip()
            if not construct or _canonical_construct(construct) in {"NA", "NONE", "NAN"}:
                continue
            grouped[barcode_column].append(construct)
            roster.add(construct)
    singlets: dict[str, str] = {}
    for barcode, calls in grouped.items():
        unique = {_canonical_construct(call): call for call in calls}
        if len(unique) == 1:
            singlets[barcode] = next(iter(unique.values()))
    return singlets, roster


def prepare_cache(raw_tar: Path, cache_path: Path) -> dict[str, object]:
    """Reduce the sealed GEO archive to six marker counts in mono-construct cells."""

    if raw_tar.stat().st_size != RAW_BYTES or _sha256(raw_tar) != RAW_SHA256:
        raise ValueError("GSE143417_RAW.tar does not match the frozen size and SHA-256")
    columns: dict[str, list[np.ndarray]] = {
        "cell_id": [],
        "donor": [],
        "context": [],
        "replicate": [],
        "construct": [],
        "marker_counts": [],
        "library_size": [],
    }
    roster: set[str] = set()
    with tarfile.open(raw_tar, mode="r:") as outer:
        for donor, context, replicate, gex_accession, tcr_accession in SAMPLES:
            gex = _read_member_bytes(outer, gex_accession, "GEX.tar.gz")
            calls = _read_member_bytes(outer, tcr_accession, "TCR.tsv.gz")
            barcodes, marker_counts, library_size = _read_gex(gex)
            singlets, sample_roster = _read_construct_calls(calls)
            roster.update(sample_roster)
            keys = np.asarray([_terminal_barcode(barcode) for barcode in barcodes])
            keep = np.asarray([key in singlets for key in keys], dtype=bool)
            kept_keys = keys[keep]
            columns["cell_id"].append(
                np.asarray([f"{gex_accession}:{key}" for key in kept_keys])
            )
            columns["donor"].append(np.full(keep.sum(), donor))
            columns["context"].append(np.full(keep.sum(), context))
            columns["replicate"].append(np.full(keep.sum(), replicate, dtype=int))
            columns["construct"].append(
                np.asarray([singlets[str(key)] for key in kept_keys])
            )
            columns["marker_counts"].append(marker_counts[keep])
            columns["library_size"].append(library_size[keep])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{name: np.concatenate(values) for name, values in columns.items()},
        marker_names=np.asarray(MARKER_GENES),
        construct_roster=np.asarray(sorted(roster, key=_canonical_construct)),
        source_sha256=np.asarray(RAW_SHA256),
    )
    return {
        "cache": str(cache_path.relative_to(ROOT)),
        "cache_sha256": _sha256(cache_path),
        "mono_construct_cells": int(sum(len(values) for values in columns["donor"])),
        "constructs": len(roster),
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "cell_id",
            "donor",
            "context",
            "replicate",
            "construct",
            "marker_counts",
            "library_size",
            "marker_names",
            "construct_roster",
            "source_sha256",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"cache is missing frozen arrays: {sorted(missing)}")
        return {name: archive[name].copy() for name in required}


def _eligibility(data: dict[str, np.ndarray]) -> tuple[list[str], dict[str, dict[str, int]]]:
    donors = data["donor"].astype(str)
    contexts = data["context"].astype(str)
    constructs = data["construct"].astype(str)
    roster = data["construct_roster"].astype(str)
    canonical = {_canonical_construct(name): name for name in roster}
    if len(canonical) != EXPECTED_LIBRARY_SIZE:
        raise ValueError(
            f"expected {EXPECTED_LIBRARY_SIZE} distinct constructs, found {len(canonical)}"
        )
    support: dict[str, dict[str, int]] = {}
    eligible = []
    for key, name in sorted(canonical.items()):
        support[key] = {}
        for donor in (DEVELOPMENT_DONOR, HELD_DONOR):
            for context in (STIM, TGFB):
                count = int(
                    np.sum(
                        (donors == donor)
                        & (contexts == context)
                        & (np.asarray([_canonical_construct(value) for value in constructs]) == key)
                    )
                )
                support[key][f"{donor}:{context}"] = count
        if min(support[key].values()) >= MINIMUM_ARM_CELLS:
            eligible.append(key)
    excluded = set(canonical).difference(eligible)
    if excluded != {EXPECTED_METADATA_EXCLUSION}:
        raise ValueError(
            "metadata eligibility differs from the frozen audit: "
            f"expected only {EXPECTED_METADATA_EXCLUSION}, found {sorted(excluded)}"
        )
    if not set(CONTROL_CONSTRUCTS).issubset(eligible):
        raise ValueError("both pooled control constructs must pass metadata eligibility")
    return eligible, support


def _module_scores(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(data["marker_names"].astype(str))
    if names != MARKER_GENES:
        raise ValueError(f"marker order must be exactly {MARKER_GENES}")
    counts = np.asarray(data["marker_counts"], dtype=float)
    library = np.asarray(data["library_size"], dtype=float)
    if counts.shape != (len(library), len(MARKER_GENES)):
        raise ValueError("marker_counts shape is inconsistent with the cache contract")
    if (
        not np.isfinite(counts).all()
        or not np.isfinite(library).all()
        or np.any(counts < 0.0)
        or np.any(library <= 0.0)
        or np.any(counts.sum(axis=1) > library + 1e-8)
    ):
        raise ValueError("counts and library sizes must be finite nonnegative raw counts")
    normalized = np.log1p(10_000.0 * counts / library[:, None])
    return normalized[:, :3].mean(axis=1), normalized[:, 3:].mean(axis=1)


def _fit_states(
    data: dict[str, np.ndarray], first_score: np.ndarray, second_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    donors = data["donor"].astype(str)
    contexts = data["context"].astype(str)
    constructs = np.asarray(
        [_canonical_construct(value) for value in data["construct"].astype(str)]
    )
    calibration = (
        (donors == DEVELOPMENT_DONOR)
        & (contexts == STIM)
        & np.isin(constructs, CONTROL_CONSTRUCTS)
    )
    if calibration.sum() < 2 * MINIMUM_ARM_CELLS:
        raise ValueError("development Stim GFP/mCherry calibration subset is too small")

    def encode(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
        cuts = np.quantile(values[calibration], [1.0 / 3.0, 2.0 / 3.0])
        if not np.isfinite(cuts).all() or cuts[0] >= cuts[1]:
            raise ValueError("development-control tertile thresholds are degenerate")
        states = np.digitize(values, cuts, right=False).astype(int)
        return states, [float(cuts[0]), float(cuts[1])]

    first, first_cuts = encode(first_score)
    second, second_cuts = encode(second_score)
    return first, second, {
        "calibration": "Donor1 Stim pooled GFP+mCherry mono-construct cells",
        "calibration_cells": int(calibration.sum()),
        "proliferation": first_cuts,
        "effector": second_cuts,
    }


def _state_occupancy_preflight(
    data: dict[str, np.ndarray],
    first: np.ndarray,
    second: np.ndarray,
    eligible: list[str],
) -> dict[str, object]:
    donors = data["donor"].astype(str)
    contexts = data["context"].astype(str)
    constructs = np.asarray(
        [_canonical_construct(value) for value in data["construct"].astype(str)]
    )
    minimum_fraction = 1.0
    arm_count = 0
    for donor in (DEVELOPMENT_DONOR, HELD_DONOR):
        for context in (STIM, TGFB):
            for construct in eligible:
                mask = (
                    (donors == donor)
                    & (contexts == context)
                    & (constructs == construct)
                )
                total = int(mask.sum())
                for states in (first, second):
                    counts = np.bincount(states[mask], minlength=STATE_COUNT)
                    fraction = float(counts.min() / total)
                    minimum_fraction = min(minimum_fraction, fraction)
                    if fraction < MINIMUM_STATE_FRACTION:
                        raise ValueError(
                            "state-occupancy preflight failed for "
                            f"{donor}:{context}:{construct}"
                        )
                arm_count += 1
    return {
        "arms": arm_count,
        "minimum_state_fraction": minimum_fraction,
        "required_minimum_state_fraction": MINIMUM_STATE_FRACTION,
        "passed": True,
    }


def _table_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    seed: int,
    permutations: int,
) -> dict[str, object]:
    first_values = first[mask]
    second_values = second[mask]
    if len(first_values) < MINIMUM_ARM_CELLS:
        raise ValueError("a frozen arm fell below the 30-cell support threshold")
    table = np.bincount(
        first_values * STATE_COUNT + second_values,
        minlength=STATE_COUNT**2,
    ).reshape(STATE_COUNT, STATE_COUNT)
    field = _conditional_table_estimate(table, "field", seed, permutations)
    pearson = _conditional_table_estimate(table, "pearson", seed, permutations)
    deviance = _conditional_table_estimate(table, "deviance", seed, permutations)
    probability = table / table.sum()
    basis = helmert_contrast(STATE_COUNT)
    endpoint = np.concatenate(
        (basis.T @ probability.sum(axis=1), basis.T @ probability.sum(axis=0))
    )
    return {
        "table": table,
        "field": field,
        "pearson": pearson,
        "deviance": deviance,
        "endpoint": endpoint,
        "cells": len(first_values),
    }


def _conditional_table_estimate(
    table: np.ndarray,
    family: str,
    seed: int,
    permutations: int,
) -> ConditionalAssociationEstimate:
    """Center a table against canonical fixed-margin permutations.

    Canonical row and column state vectors make the null mean a function only
    of the two margins, table size, and frozen seed. This permits prediction
    code to compute the held-target null correction without opening pairing.
    """

    values = np.asarray(table, dtype=int)
    row_margin = values.sum(axis=1)
    column_margin = values.sum(axis=0)
    if np.any(row_margin <= 0) or np.any(column_margin <= 0):
        raise ValueError("every state must have positive support")
    first = np.repeat(np.arange(STATE_COUNT), row_margin)
    second = np.repeat(np.arange(STATE_COUNT), column_margin)

    def transform(candidate: np.ndarray) -> np.ndarray:
        if family == "field":
            return association_coordinates(
                association_field(candidate, pseudocount=PSEUDOCOUNT)
            )
        if family in {"pearson", "deviance"}:
            return poisson_independence_residuals(candidate, residual=family)
        raise ValueError(f"unknown table family: {family}")

    generator = np.random.default_rng(seed)
    null = []
    for _ in range(permutations + 1):
        permuted = second[generator.permutation(len(second))]
        candidate = np.bincount(
            first * STATE_COUNT + permuted, minlength=STATE_COUNT**2
        ).reshape(STATE_COUNT, STATE_COUNT)
        null.append(transform(candidate))
    null_values = np.asarray(null)
    reference = null_values[1:].mean(axis=0)
    observed = transform(values)
    return ConditionalAssociationEstimate(
        coordinates=observed - reference,
        observed_coordinates=observed,
        null_mean_coordinates=reference,
        null_variance_coordinates=np.var(null_values[1:], axis=0, ddof=1)
        * (1.0 + 1.0 / permutations),
        destroyed_coordinates=null_values[0] - reference,
        permutations=permutations,
    )


def _margin_only_statistics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    seed: int,
    permutations: int,
) -> dict[str, object]:
    """Build target margins and null corrections without forming joint pairs."""

    row_margin = np.bincount(first[mask], minlength=STATE_COUNT)
    column_margin = np.bincount(second[mask], minlength=STATE_COUNT)
    if (
        int(row_margin.sum()) < MINIMUM_ARM_CELLS
        or np.any(row_margin <= 0)
        or np.any(column_margin <= 0)
    ):
        raise ValueError("a held margin failed the frozen support contract")
    # Only null means are retained. Canonical state vectors make them functions
    # of the margins, table size, and seed rather than the sealed pairing.
    output: dict[str, object] = {
        "row_margin": row_margin,
        "column_margin": column_margin,
        "cells": int(row_margin.sum()),
    }
    for family in ("field", "pearson", "deviance"):
        row_states = np.repeat(np.arange(STATE_COUNT), row_margin)
        col_states = np.repeat(np.arange(STATE_COUNT), column_margin)
        generator = np.random.default_rng(seed)
        null = []
        for _ in range(permutations + 1):
            permuted = col_states[generator.permutation(len(col_states))]
            table = np.bincount(
                row_states * STATE_COUNT + permuted,
                minlength=STATE_COUNT**2,
            ).reshape(STATE_COUNT, STATE_COUNT)
            if family == "field":
                transformed = association_coordinates(
                    association_field(table, pseudocount=PSEUDOCOUNT)
                )
            else:
                transformed = poisson_independence_residuals(table, residual=family)
            null.append(transformed)
        null_values = np.asarray(null)
        reference = null_values[1:].mean(axis=0)
        output[family] = ConditionalAssociationEstimate(
            coordinates=np.zeros_like(reference),
            observed_coordinates=reference,
            null_mean_coordinates=reference,
            null_variance_coordinates=np.var(null_values[1:], axis=0, ddof=1)
            * (1.0 + 1.0 / permutations),
            destroyed_coordinates=null_values[0] - reference,
            permutations=permutations,
        )
    basis = helmert_contrast(STATE_COUNT)
    output["endpoint"] = np.concatenate(
        (
            basis.T @ (row_margin / row_margin.sum()),
            basis.T @ (column_margin / column_margin.sum()),
        )
    )
    return output


def _factorial_contrast(
    target_tgfb: ConditionalAssociationEstimate,
    target_stim: ConditionalAssociationEstimate,
    control_tgfb: ConditionalAssociationEstimate,
    control_stim: ConditionalAssociationEstimate,
    *,
    scale: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scale is None:
        scale = (1.0, 1.0, 1.0, 1.0)
    estimates = (target_tgfb, target_stim, control_tgfb, control_stim)
    centered = [estimate.coordinates / factor for estimate, factor in zip(estimates, scale)]
    destroyed = [
        estimate.destroyed_coordinates / factor
        for estimate, factor in zip(estimates, scale)
    ]
    variances = [
        estimate.null_variance_coordinates / factor**2
        for estimate, factor in zip(estimates, scale)
    ]
    return (
        centered[0] - centered[1] - centered[2] + centered[3],
        destroyed[0] - destroyed[1] - destroyed[2] + destroyed[3],
        sum(variances),
    )


def _construct_laplacian(
    constructs: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    normalized = [_canonical_construct(name) for name in constructs]
    memberships = []
    retained = []
    for component, aliases in ARCHITECTURE_COMPONENTS.items():
        members = [
            index
            for index, label in enumerate(normalized)
            if (
                label.endswith("DN")
                if component == "domain:DN"
                else any(alias in label for alias in aliases)
            )
        ]
        if len(members) >= 2:
            memberships.append(members)
            retained.append(component)
    if not memberships:
        raise ValueError("no shared frozen construct-architecture component was found")
    incidence = np.zeros((len(constructs), len(memberships)), dtype=float)
    for column, members in enumerate(memberships):
        incidence[members, column] = 1.0
    covered = np.flatnonzero(incidence.sum(axis=1) > 0.0)
    return normalized_hypergraph_laplacian(incidence), {
        "components": retained,
        "members": {
            component: [constructs[index] for index in memberships[column]]
            for column, component in enumerate(retained)
        },
        "covered_constructs": int(len(covered)),
        "isolated_constructs": [
            constructs[index]
            for index in range(len(constructs))
            if index not in set(covered)
        ],
    }


def _fixed_prediction(
    values: np.ndarray,
    variance: np.ndarray,
    laplacian: np.ndarray | None,
    *,
    nuclear_fraction: float = FINAL_NUCLEAR_FRACTION,
    graph_penalty: float = FINAL_GRAPH_PENALTY,
) -> tuple[np.ndarray, dict[str, object]]:
    weights = inverse_permutation_variance_weights(variance)
    leading = float(np.linalg.svd(values, compute_uv=False)[0])
    fit = fit_structured_coupling_fields(
        values,
        observation_weight=weights,
        graph_laplacian=laplacian,
        nuclear_penalty=nuclear_fraction * leading,
        graph_penalty=graph_penalty,
        tolerance=1e-9,
    )
    return fit.coefficient, {
        "nuclear_fraction": nuclear_fraction,
        "graph_penalty": graph_penalty,
        "effective_rank": fit.effective_rank,
        "iterations": fit.iterations,
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
    }


def _variance_scalar_prediction(
    values: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    signal = float(np.sum(values**2))
    noise = float(np.sum(variance))
    scale = float(np.clip(1.0 - noise / max(signal, 1e-12), 0.0, 1.0))
    return scale * values, {"scale": scale, "signal_sum_squares": signal, "noise_sum": noise}


def _field_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    pred = prediction.ravel()
    held = truth.ravel()
    correlation = (
        float(np.corrcoef(pred, held)[0, 1])
        if np.std(pred) > 0.0 and np.std(held) > 0.0
        else None
    )
    rmse = float(
        np.sqrt(np.mean((prediction - truth) ** 2) / max(np.mean(truth**2), 1e-12))
    )
    return {"pooled_pearson": correlation, "standardized_rmse": rmse}


def _bootstrap_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, list[float] | None]:
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"pooled_pearson": [], "standardized_rmse": []}
    for _ in range(draws):
        indices = generator.integers(0, len(truth), size=len(truth))
        metrics = _field_metrics(prediction[indices], truth[indices])
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    return {
        name: (
            [float(value) for value in np.quantile(samples, [0.025, 0.975])]
            if samples
            else None
        )
        for name, samples in values.items()
    }


def _bootstrap_loss_difference(
    first: np.ndarray,
    alternatives: list[np.ndarray],
    draws: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        indices = generator.integers(0, len(first), size=len(first))
        first_loss = float(np.mean(first[indices]))
        alternative_loss = min(float(np.mean(values[indices])) for values in alternatives)
        differences.append(first_loss - alternative_loss)
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def _reconstruct_field(
    target_stim: dict[str, object],
    target_tgfb: dict[str, object],
    control_stim: dict[str, object],
    control_tgfb: dict[str, object],
    contrast: np.ndarray,
) -> np.ndarray:
    target_stim_estimate = target_stim["field"]
    target_tgfb_estimate = target_tgfb["field"]
    control_stim_estimate = control_stim["field"]
    control_tgfb_estimate = control_tgfb["field"]
    assert isinstance(target_stim_estimate, ConditionalAssociationEstimate)
    assert isinstance(target_tgfb_estimate, ConditionalAssociationEstimate)
    assert isinstance(control_stim_estimate, ConditionalAssociationEstimate)
    assert isinstance(control_tgfb_estimate, ConditionalAssociationEstimate)
    raw = (
        target_stim_estimate.coordinates
        + contrast
        + control_tgfb_estimate.coordinates
        - control_stim_estimate.coordinates
        + target_tgfb_estimate.null_mean_coordinates
    )
    if "table" in target_tgfb:
        table = np.asarray(target_tgfb["table"], dtype=float)
        rows, columns = table.sum(axis=1), table.sum(axis=0)
    else:
        rows = np.asarray(target_tgfb["row_margin"], dtype=float)
        columns = np.asarray(target_tgfb["column_margin"], dtype=float)
    return field_coordinates_to_table(raw, rows, columns)


def _reconstruct_residual(
    family: str,
    target_stim: dict[str, object],
    target_tgfb: dict[str, object],
    control_stim: dict[str, object],
    control_tgfb: dict[str, object],
    contrast: np.ndarray,
) -> np.ndarray:
    estimates = [
        target_stim[family],
        target_tgfb[family],
        control_stim[family],
        control_tgfb[family],
    ]
    if not all(isinstance(value, ConditionalAssociationEstimate) for value in estimates):
        raise TypeError("residual family did not contain conditional estimates")
    target_stim_estimate, target_tgfb_estimate, control_stim_estimate, control_tgfb_estimate = estimates
    def total(values: dict[str, object]) -> float:
        if "table" in values:
            return float(np.asarray(values["table"]).sum())
        return float(np.asarray(values["row_margin"]).sum())

    scale = [
        np.sqrt(total(values))
        for values in (target_stim, target_tgfb, control_stim, control_tgfb)
    ]
    centered_target_scale = (
        target_stim_estimate.coordinates / scale[0]
        + contrast
        + control_tgfb_estimate.coordinates / scale[3]
        - control_stim_estimate.coordinates / scale[2]
    )
    raw = centered_target_scale * scale[1] + target_tgfb_estimate.null_mean_coordinates
    if "table" in target_tgfb:
        table = np.asarray(target_tgfb["table"], dtype=float)
        rows, columns = table.sum(axis=1), table.sum(axis=0)
    else:
        rows = np.asarray(target_tgfb["row_margin"], dtype=float)
        columns = np.asarray(target_tgfb["column_margin"], dtype=float)
    return residual_coordinates_to_table(
        raw,
        rows,
        columns,
        residual=family,
    )


def _factorial_endpoint(
    target_tgfb: dict[str, object],
    target_stim: dict[str, object],
    control_tgfb: dict[str, object],
    control_stim: dict[str, object],
) -> np.ndarray:
    return (
        np.asarray(target_tgfb["endpoint"])
        - np.asarray(target_stim["endpoint"])
        - np.asarray(control_tgfb["endpoint"])
        + np.asarray(control_stim["endpoint"])
    )


def fit_pretruth_predictions(
    data: dict[str, np.ndarray],
    *,
    permutations: int = NULL_PERMUTATIONS,
) -> dict[str, object]:
    """Fit and write all predictions without forming held target-TGFB pairs."""

    eligible, support = _eligibility(data)
    query = [name for name in eligible if name not in CONTROL_CONSTRUCTS]
    first_score, second_score = _module_scores(data)
    first, second, thresholds = _fit_states(data, first_score, second_score)
    occupancy = _state_occupancy_preflight(data, first, second, eligible)
    donors = data["donor"].astype(str)
    contexts = data["context"].astype(str)
    constructs = np.asarray(
        [_canonical_construct(value) for value in data["construct"].astype(str)]
    )

    controls: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for donor_index, donor in enumerate((DEVELOPMENT_DONOR, HELD_DONOR)):
        for context_index, context in enumerate((STIM, TGFB)):
            mask = (
                (donors == donor)
                & (contexts == context)
                & np.isin(constructs, CONTROL_CONSTRUCTS)
            )
            controls[donor][context] = _table_statistics(
                first,
                second,
                mask,
                SEED + 100_000 * donor_index + 10_000 * context_index,
                permutations,
            )

    development: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    held_stim: dict[str, dict[str, object]] = {}
    held_tgfb_margins: dict[str, dict[str, object]] = {}
    for target_index, target in enumerate(query):
        for context_index, context in enumerate((STIM, TGFB)):
            mask = (
                (donors == DEVELOPMENT_DONOR)
                & (contexts == context)
                & (constructs == target)
            )
            development[context][target] = _table_statistics(
                first,
                second,
                mask,
                SEED + 10_000 * context_index + target_index + 1,
                permutations,
            )
        stim_mask = (
            (donors == HELD_DONOR)
            & (contexts == STIM)
            & (constructs == target)
        )
        held_stim[target] = _table_statistics(
            first,
            second,
            stim_mask,
            SEED + 100_000 + target_index + 1,
            permutations,
        )
        tgfb_mask = (
            (donors == HELD_DONOR)
            & (contexts == TGFB)
            & (constructs == target)
        )
        held_tgfb_margins[target] = _margin_only_statistics(
            first,
            second,
            tgfb_mask,
            SEED + 110_000 + target_index + 1,
            permutations,
        )

    contrasts: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    destroyed: dict[str, np.ndarray] = {}
    for family in ("field", "pearson", "deviance"):
        family_values, family_variance, family_destroyed = [], [], []
        for target in query:
            estimates = (
                development[TGFB][target][family],
                development[STIM][target][family],
                controls[DEVELOPMENT_DONOR][TGFB][family],
                controls[DEVELOPMENT_DONOR][STIM][family],
            )
            if not all(isinstance(value, ConditionalAssociationEstimate) for value in estimates):
                raise TypeError("conditional estimate contract failed")
            scale = None
            if family != "field":
                scale = tuple(
                    np.sqrt(float(np.asarray(item["table"]).sum()))
                    for item in (
                        development[TGFB][target],
                        development[STIM][target],
                        controls[DEVELOPMENT_DONOR][TGFB],
                        controls[DEVELOPMENT_DONOR][STIM],
                    )
                )
            value, broken, variance = _factorial_contrast(*estimates, scale=scale)
            family_values.append(value.ravel())
            family_destroyed.append(broken.ravel())
            family_variance.append(variance.ravel())
        contrasts[family] = np.asarray(family_values)
        destroyed[family] = np.asarray(family_destroyed)
        variances[family] = np.asarray(family_variance)

    laplacian, architecture = _construct_laplacian(query)
    permutation = np.random.default_rng(SEED + 500).permutation(len(query))
    shuffled_laplacian = laplacian[np.ix_(permutation, permutation)]
    predictions: dict[str, np.ndarray] = {
        "field_zero": np.zeros_like(contrasts["field"]),
        "field_direct": contrasts["field"],
        "field_destroyed": destroyed["field"],
        "pearson_direct": contrasts["pearson"],
        "deviance_direct": contrasts["deviance"],
    }
    selection: dict[str, object] = {}
    predictions["field_scalar"], selection["field_scalar"] = _variance_scalar_prediction(
        contrasts["field"], variances["field"]
    )
    predictions["field_nuclear"], selection["field_nuclear"] = _fixed_prediction(
        contrasts["field"], variances["field"], None, graph_penalty=0.0
    )
    predictions["field_hypergraph"], selection["field_hypergraph"] = _fixed_prediction(
        contrasts["field"],
        variances["field"],
        laplacian,
        nuclear_fraction=0.0,
    )
    predictions["field_fixed"], selection["field_fixed"] = _fixed_prediction(
        contrasts["field"], variances["field"], laplacian
    )
    predictions["field_label_permuted"], selection["field_label_permuted"] = (
        _fixed_prediction(contrasts["field"], variances["field"], shuffled_laplacian)
    )
    for family in ("pearson", "deviance"):
        predictions[family + "_fixed"], selection[family + "_fixed"] = _fixed_prediction(
            contrasts[family], variances[family], laplacian
        )

    development_endpoint = np.asarray(
        [
            _factorial_endpoint(
                development[TGFB][target],
                development[STIM][target],
                controls[DEVELOPMENT_DONOR][TGFB],
                controls[DEVELOPMENT_DONOR][STIM],
            )
            for target in query
        ]
    )
    held_endpoint = np.asarray(
        [
            _factorial_endpoint(
                held_tgfb_margins[target],
                held_stim[target],
                controls[HELD_DONOR][TGFB],
                controls[HELD_DONOR][STIM],
            )
            for target in query
        ]
    )
    endpoint_model = Ridge(alpha=0.1).fit(development_endpoint, contrasts["field"])
    predictions["field_endpoint"] = endpoint_model.predict(held_endpoint)
    selection["field_endpoint"] = {"ridge_penalty": 0.1}

    predicted_tables: dict[str, list[list[list[float]]]] = {
        name: [] for name in predictions
    }
    for target_index, target in enumerate(query):
        for name, prediction in predictions.items():
            family = name.split("_", maxsplit=1)[0]
            if family == "field":
                table = _reconstruct_field(
                    held_stim[target],
                    held_tgfb_margins[target],
                    controls[HELD_DONOR][STIM],
                    controls[HELD_DONOR][TGFB],
                    prediction[target_index].reshape(2, 2),
                )
            else:
                table = _reconstruct_residual(
                    family,
                    held_stim[target],
                    held_tgfb_margins[target],
                    controls[HELD_DONOR][STIM],
                    controls[HELD_DONOR][TGFB],
                    prediction[target_index].reshape(3, 3),
                )
            predicted_tables[name].append(table.tolist())

    return {
        "protocol": "gse143417-pokiseq-held-donor/1.0",
        "stage": "PRETRUTH_PREDICTIONS_WRITTEN",
        "query_constructs": query,
        "state_thresholds": thresholds,
        "state_occupancy_preflight": occupancy,
        "support": {
            "raw_constructs": EXPECTED_LIBRARY_SIZE,
            "eligible_constructs_including_controls": len(eligible),
            "query_constructs": len(query),
            "excluded": [EXPECTED_METADATA_EXCLUSION],
            "counts": support,
        },
        "architecture_hypergraph": architecture,
        "selection": selection,
        "coordinate_predictions": {
            name: values.tolist() for name, values in predictions.items()
        },
        "predicted_held_tgfb_tables": predicted_tables,
        "design": {
            "development_donor": DEVELOPMENT_DONOR,
            "held_donor": HELD_DONOR,
            "factorial_target": "(construct TGFB - construct Stim) - (pooled GFP+mCherry TGFB - pooled GFP+mCherry Stim)",
            "sealed_at_this_stage": "held Donor2 target-TGFB pairing and joint tables",
            "available_at_this_stage": "held target-TGFB margins, target-Stim pairs, and pooled-control pairs",
            "null_permutations": permutations,
        },
    }


def score_pretruth_predictions(
    data: dict[str, np.ndarray],
    prediction_record: dict[str, object],
    *,
    permutations: int = NULL_PERMUTATIONS,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, object]:
    """Open held target-TGFB pairing only after predictions are immutable."""

    if prediction_record.get("stage") != "PRETRUTH_PREDICTIONS_WRITTEN":
        raise ValueError("prediction record does not prove the pre-truth stage")
    eligible, support = _eligibility(data)
    query = [name for name in eligible if name not in CONTROL_CONSTRUCTS]
    if prediction_record.get("query_constructs") != query:
        raise ValueError("prediction target order differs from the frozen query order")
    first_score, second_score = _module_scores(data)
    first, second, thresholds = _fit_states(data, first_score, second_score)
    occupancy = _state_occupancy_preflight(data, first, second, eligible)
    if prediction_record.get("state_thresholds") != thresholds:
        raise ValueError("state thresholds differ from the pre-truth prediction record")
    if prediction_record.get("state_occupancy_preflight") != occupancy:
        raise ValueError("state occupancy differs from the pre-truth prediction record")
    donors = data["donor"].astype(str)
    contexts = data["context"].astype(str)
    constructs = np.asarray(
        [_canonical_construct(value) for value in data["construct"].astype(str)]
    )
    control_mask = (
        (donors == HELD_DONOR)
        & np.isin(constructs, CONTROL_CONSTRUCTS)
    )
    controls = {}
    for context_index, context in enumerate((STIM, TGFB)):
        controls[context] = _table_statistics(
            first,
            second,
            control_mask & (contexts == context),
            SEED + 100_000 + 10_000 * context_index,
            permutations,
        )
    held_stim = {}
    held_tgfb = {}
    for target_index, target in enumerate(query):
        for context_index, destination in enumerate((held_stim, held_tgfb)):
            context = (STIM, TGFB)[context_index]
            mask = (
                (donors == HELD_DONOR)
                & (contexts == context)
                & (constructs == target)
            )
            destination[target] = _table_statistics(
                first,
                second,
                mask,
                SEED + 100_000 + 10_000 * context_index + target_index + 1,
                permutations,
            )

    truths: dict[str, np.ndarray] = {}
    for family in ("field", "pearson", "deviance"):
        values = []
        for target in query:
            estimates = (
                held_tgfb[target][family],
                held_stim[target][family],
                controls[TGFB][family],
                controls[STIM][family],
            )
            scale = None
            if family != "field":
                scale = tuple(
                    np.sqrt(float(np.asarray(item["table"]).sum()))
                    for item in (
                        held_tgfb[target],
                        held_stim[target],
                        controls[TGFB],
                        controls[STIM],
                    )
                )
            value, _, _ = _factorial_contrast(*estimates, scale=scale)
            values.append(value.ravel())
        truths[family] = np.asarray(values)

    coordinate_predictions = {
        name: np.asarray(values, dtype=float)
        for name, values in prediction_record["coordinate_predictions"].items()
    }
    table_predictions = {
        name: np.asarray(values, dtype=float)
        for name, values in prediction_record["predicted_held_tgfb_tables"].items()
    }
    common_loss_arrays = {}
    for name, tables in table_predictions.items():
        common_loss_arrays[name] = np.asarray(
            [
                multinomial_deviance_per_observation(
                    np.asarray(held_tgfb[target]["table"], dtype=float), tables[index]
                )
                for index, target in enumerate(query)
            ]
        )

    primary = coordinate_predictions["field_fixed"]
    truth = truths["field"]
    correlation_interval = _bootstrap_metrics(
        primary, truth, bootstraps, SEED + 1_000
    )["pooled_pearson"]
    primary_minus_destroyed = _bootstrap_loss_difference(
        common_loss_arrays["field_fixed"],
        [common_loss_arrays["field_destroyed"]],
        bootstraps,
        SEED + 2_000,
    )
    primary_minus_classical = _bootstrap_loss_difference(
        common_loss_arrays["field_fixed"],
        [
            common_loss_arrays["pearson_direct"],
            common_loss_arrays["pearson_fixed"],
            common_loss_arrays["deviance_direct"],
            common_loss_arrays["deviance_fixed"],
        ],
        bootstraps,
        SEED + 3_000,
    )
    matched_names = (
        "field_zero",
        "field_direct",
        "field_scalar",
        "field_nuclear",
        "field_hypergraph",
        "field_endpoint",
        "field_label_permuted",
    )
    primary_minus_matched = _bootstrap_loss_difference(
        common_loss_arrays["field_fixed"],
        [common_loss_arrays[name] for name in matched_names],
        bootstraps,
        SEED + 4_000,
    )
    passed = bool(
        correlation_interval is not None
        and correlation_interval[0] > 0.0
        and primary_minus_destroyed[1] < 0.0
        and primary_minus_classical[1] < 0.0
        and primary_minus_matched[1] < 0.0
    )

    representation = {}
    for name, prediction in coordinate_predictions.items():
        family = name.split("_", maxsplit=1)[0]
        representation[name] = {
            "metrics": _field_metrics(prediction, truths[family]),
            "target_bootstrap_95_ci": _bootstrap_metrics(
                prediction,
                truths[family],
                bootstraps,
                SEED + 20_000 + len(representation) * 1_000,
            ),
        }

    def subset(names: tuple[str, ...]) -> dict[str, object]:
        indices = [query.index(name) for name in names if name in query]
        return {
            "targets": [query[index] for index in indices],
            "truth_norm_mean": float(np.linalg.norm(truth[indices], axis=1).mean())
            if indices
            else None,
            "prediction_norm_mean": float(np.linalg.norm(primary[indices], axis=1).mean())
            if indices
            else None,
            "common_table_deviance_mean": float(
                common_loss_arrays["field_fixed"][indices].mean()
            )
            if indices
            else None,
        }

    return {
        "protocol": "gse143417-pokiseq-held-donor/1.0",
        "stage": "HELD_PAIRING_SCORED_AFTER_PREDICTION",
        "design": prediction_record["design"],
        "support": {**prediction_record["support"], "counts": support},
        "query_constructs": query,
        "architecture_hypergraph": prediction_record["architecture_hypergraph"],
        "selection": prediction_record["selection"],
        "common_held_table": {
            name: {
                "mean_multinomial_deviance_per_cell": float(values.mean()),
                "per_target": values.tolist(),
            }
            for name, values in common_loss_arrays.items()
        },
        "representation_space_secondary": representation,
        "primary_gate": {
            "field_fixed_pooled_pearson_95_ci": correlation_interval,
            "field_fixed_minus_destroyed_deviance_95_ci": primary_minus_destroyed,
            "field_fixed_minus_best_classical_deviance_95_ci": primary_minus_classical,
            "field_fixed_minus_best_matched_deviance_95_ci": primary_minus_matched,
            "best_classical_candidates": [
                "pearson_direct",
                "pearson_fixed",
                "deviance_direct",
                "deviance_fixed",
            ],
            "best_matched_candidates": list(matched_names),
            "passed": passed,
        },
        "biological_controls": {
            "positive_tgfbr2_class": subset(POSITIVE_CONTROL_CLASS),
            "tngfr_falsification": subset((FALSIFICATION_TARGET,)),
        },
        "held_truth": truth.tolist(),
        "primary_prediction": primary.tolist(),
    }


def analyze_cache(
    data: dict[str, np.ndarray],
    *,
    permutations: int = NULL_PERMUTATIONS,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, object]:
    """Synthetic-test convenience wrapper; production CLI enforces two stages."""

    prediction = fit_pretruth_predictions(data, permutations=permutations)
    return score_pretruth_predictions(
        data, prediction, permutations=permutations, bootstraps=bootstraps
    )


def _require_authorized_lock(lock_path: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "SEALED" or lock.get("outcome_access_authorized") is not True:
        raise PermissionError("the prospective POKI outcome lock is not authorized")
    public_commit = lock.get("public_freeze_commit")
    public_url = lock.get("public_freeze_url")
    if not isinstance(public_commit, str) or re.fullmatch(r"[0-9a-f]{40}", public_commit) is None:
        raise PermissionError("the prospective POKI lock has no public freeze commit")
    if (
        not isinstance(public_url, str)
        or not public_url.startswith("https://github.com/")
        or public_commit not in public_url
    ):
        raise PermissionError("the prospective POKI lock has no commit-bound public URL")
    bound = lock.get("implementation_sha256", {})
    if not isinstance(bound, dict) or not bound:
        raise PermissionError("authorized lock has no implementation manifest")
    for relative, expected in bound.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise PermissionError(f"bound bytes differ from the lock: {relative}")
    return lock


def _require_pretruth_designation(
    designation_path: Path,
    predictions_path: Path,
    cache_path: Path,
) -> dict[str, object]:
    if not designation_path.is_file():
        raise PermissionError("a sealed pre-truth prediction designation is required")
    designation = json.loads(designation_path.read_text())
    if designation.get("status") != "SEALED_FOR_SCORING":
        raise PermissionError("pre-truth prediction record is not sealed for scoring")
    if designation.get("pretruth_predictions_sha256") != _sha256(predictions_path):
        raise PermissionError("pre-truth prediction bytes differ from the designation")
    if designation.get("cache_sha256") != _sha256(cache_path):
        raise PermissionError("cache bytes differ from the pre-truth designation")
    if designation.get("implementation_sha256") != _implementation_sha256():
        raise PermissionError("implementation bytes differ from the pre-truth designation")
    public_commit = designation.get("pretruth_predictions_public_commit")
    public_url = designation.get("pretruth_predictions_public_url")
    if not isinstance(public_commit, str) or re.fullmatch(r"[0-9a-f]{40}", public_commit) is None:
        raise PermissionError("pre-truth predictions have no immutable public commit")
    if (
        not isinstance(public_url, str)
        or not public_url.startswith("https://github.com/")
        or public_commit not in public_url
    ):
        raise PermissionError("pre-truth predictions have no commit-bound public URL")
    return designation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw-tar", type=Path, default=RAW_TAR)
    prepare.add_argument("--cache", type=Path, default=CACHE)
    prepare.add_argument("--lock", type=Path, default=LOCK)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--cache", type=Path, default=CACHE)
    predict.add_argument("--output", type=Path, default=PREDICTIONS)
    predict.add_argument("--lock", type=Path, default=LOCK)
    score = subparsers.add_parser("score")
    score.add_argument("--cache", type=Path, default=CACHE)
    score.add_argument("--predictions", type=Path, default=PREDICTIONS)
    score.add_argument(
        "--pretruth-designation", type=Path, default=PRETRUTH_DESIGNATION
    )
    score.add_argument("--output", type=Path, default=OUTPUT)
    score.add_argument("--lock", type=Path, default=LOCK)
    arguments = parser.parse_args()
    _require_authorized_lock(arguments.lock)
    if arguments.command == "prepare":
        print(json.dumps(prepare_cache(arguments.raw_tar, arguments.cache), indent=2))
        return
    if arguments.command == "predict":
        data = _load_cache(arguments.cache)
        prediction = fit_pretruth_predictions(data)
        prediction["provenance"] = {
            "cache": str(arguments.cache.relative_to(ROOT)),
            "cache_sha256": _sha256(arguments.cache),
            "source_sha256": str(np.asarray(data["source_sha256"]).item()),
            "lock_sha256": _sha256(arguments.lock),
            "implementation_sha256": _implementation_sha256(),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(prediction, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"stage": prediction["stage"], "output": str(arguments.output)}, indent=2))
        return
    if not arguments.predictions.exists():
        raise FileNotFoundError("pre-truth prediction record must exist before scoring")
    designation = _require_pretruth_designation(
        arguments.pretruth_designation, arguments.predictions, arguments.cache
    )
    prediction_sha256 = _sha256(arguments.predictions)
    prediction = json.loads(arguments.predictions.read_text())
    if prediction.get("provenance", {}).get(
        "implementation_sha256"
    ) != _implementation_sha256():
        raise PermissionError("prediction record was made by different implementation bytes")
    data = _load_cache(arguments.cache)
    result = score_pretruth_predictions(data, prediction)
    result["provenance"] = {
        "cache": str(arguments.cache.relative_to(ROOT)),
        "cache_sha256": _sha256(arguments.cache),
        "source_sha256": str(np.asarray(data["source_sha256"]).item()),
        "pretruth_predictions": str(arguments.predictions.relative_to(ROOT)),
        "pretruth_predictions_sha256": prediction_sha256,
        "pretruth_designation": str(
            arguments.pretruth_designation.relative_to(ROOT)
        ),
        "pretruth_designation_sha256": _sha256(arguments.pretruth_designation),
        "pretruth_designation_created_utc": designation.get("created_utc"),
        "pretruth_predictions_public_url": designation[
            "pretruth_predictions_public_url"
        ],
        "pretruth_predictions_public_commit": designation[
            "pretruth_predictions_public_commit"
        ],
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha256(Path(__file__).resolve()),
            "mapreg/coupling_fields.py": _sha256(ROOT / "mapreg/coupling_fields.py"),
            "mapreg/classical_residuals.py": _sha256(ROOT / "mapreg/classical_residuals.py"),
            "mapreg/table_prediction.py": _sha256(ROOT / "mapreg/table_prediction.py"),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary_gate"], indent=2))


if __name__ == "__main__":
    main()
