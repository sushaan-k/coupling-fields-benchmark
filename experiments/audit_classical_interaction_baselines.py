"""Post-hoc classical interaction baselines on the two held CITE-seq panels.

This audit does not alter either confirmation.  It reuses their development
and held splits, the same transport-multiplier grid, the frozen primary held
predictions, and the same donor-level loss.  Two classical source fields are
fit independently for every RNA--ADT pair:

* a common-effect exact conditional maximum-likelihood estimate (CMLE) from
  the product of donor-specific fixed-margin likelihoods; and
* the interaction coefficient from a saturated Poisson log-linear model fit
  to the donor-pooled table.

For a pooled 2x2 table the second coefficient is exactly its sample log odds
ratio.  It is therefore not reported as a separate "Poisson" and "pooled log
odds" baseline.  It is generally distinct from the stratified exact CMLE,
whose score retains every donor's margins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from experiments import confirm_gse239452_citeseq as gse239452
from experiments import confirm_stephenson_citeseq as stephenson
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    expected_binary_table_from_log_odds,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = (
    ROOT / "results/development/classical_interaction_baselines_posthoc.json"
)
DEFAULT_TSV = (
    ROOT / "results/development/classical_interaction_baselines_posthoc.tsv"
)
DEFAULT_STEPHENSON_H5AD = (
    ROOT
    / "data/confirmation/stephenson_citeseq/covid_portal_210320_with_raw.h5ad"
)
DEFAULT_GSE239452_SOURCE_ROOT = (
    ROOT / "data/confirmation/gse239452_citeseq/source_cache"
)

ALPHA_GRID = (0.5, 0.75, 1.0, 1.25)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260828
ROOT_TOLERANCE = 1e-11
SCORE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class InteractionField:
    """One log-odds estimate and boundary code per entity pair."""

    log_odds: np.ndarray
    boundary: np.ndarray
    certificate: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)


def _validate_tables(tables: np.ndarray) -> np.ndarray:
    values = np.asarray(tables)
    if (
        values.ndim != 5
        or values.shape[-2:] != (2, 2)
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values.sum(axis=(-2, -1)) <= 0)
    ):
        raise ValueError(
            "tables must be nonnegative integer donor x entity x entity x 2 x 2"
        )
    return values.astype(np.int64, copy=False)


def _conditional_record(table: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    counts = np.asarray(table, dtype=np.int64)
    rows = counts.sum(axis=1)
    columns = counts.sum(axis=0)
    total = int(counts.sum())
    lower = max(0, int(rows[0] + columns[0] - total))
    upper = min(int(rows[0]), int(columns[0]))
    support = np.arange(lower, upper + 1, dtype=float)
    log_weight = (
        gammaln(float(columns[0]) + 1.0)
        - gammaln(support + 1.0)
        - gammaln(float(columns[0]) - support + 1.0)
        + gammaln(float(columns[1]) + 1.0)
        - gammaln(float(rows[0]) - support + 1.0)
        - gammaln(float(columns[1] - rows[0]) + support + 1.0)
    )
    return float(counts[0, 0]), support, log_weight


def _conditional_score(
    log_odds: float, records: list[tuple[float, np.ndarray, np.ndarray]]
) -> float:
    score = 0.0
    for observed, support, log_weight in records:
        log_mass = log_weight + float(log_odds) * support
        probability = np.exp(log_mass - logsumexp(log_mass))
        score += float(np.sum(probability * support)) - observed
    return score


def _common_effect_exact_cmle(tables: np.ndarray) -> InteractionField:
    """Fit one extended exact conditional common odds ratio per entity."""

    values = _validate_tables(tables)
    shape = values.shape[1:3]
    log_odds = np.empty(shape, dtype=float)
    boundary = np.zeros(shape, dtype=np.int8)
    informative_counts = np.empty(shape, dtype=np.int64)
    score_errors: list[float] = []
    for entity in np.ndindex(shape):
        records = [
            _conditional_record(values[(donor, *entity)])
            for donor in range(values.shape[0])
        ]
        informative = [record for record in records if len(record[1]) > 1]
        if not informative:
            raise CouplingEstimationRefusal(
                f"common-effect CMLE has no informative stratum at {entity}"
            )
        informative_counts[entity] = len(informative)
        observed = sum(record[0] for record in informative)
        lower = sum(float(record[1][0]) for record in informative)
        upper = sum(float(record[1][-1]) for record in informative)
        if observed <= lower:
            log_odds[entity] = -np.inf
            boundary[entity] = -1
            continue
        if observed >= upper:
            log_odds[entity] = np.inf
            boundary[entity] = 1
            continue
        left, right = -1.0, 1.0
        while _conditional_score(left, informative) >= 0.0:
            left *= 2.0
            if left < -1024.0:
                raise CouplingEstimationRefusal("CMLE lower bracket did not close")
        while _conditional_score(right, informative) <= 0.0:
            right *= 2.0
            if right > 1024.0:
                raise CouplingEstimationRefusal("CMLE upper bracket did not close")
        estimate = brentq(
            lambda theta: _conditional_score(theta, informative),
            left,
            right,
            xtol=ROOT_TOLERANCE,
            rtol=4.0 * np.finfo(float).eps,
        )
        log_odds[entity] = estimate
        score_errors.append(abs(_conditional_score(estimate, informative)))
    maximum_score = max(score_errors, default=0.0)
    if maximum_score > SCORE_TOLERANCE:
        raise CouplingEstimationRefusal("common-effect CMLE missed its score certificate")
    return InteractionField(
        log_odds=log_odds,
        boundary=boundary,
        certificate={
            "estimator": "extended exact conditional CMLE",
            "score_equation": (
                "sum_d E_theta[n_d00 | donor d row and column margins] "
                "= sum_d n_d00"
            ),
            "root_tolerance": ROOT_TOLERANCE,
            "maximum_absolute_finite_score": maximum_score,
            "boundary_entities": int(np.count_nonzero(boundary)),
            "informative_donors_range": [
                int(informative_counts.min()),
                int(informative_counts.max()),
            ],
            "ridge_penalty": 0.0,
            "graph_penalty": 0.0,
            "donor_deviation_parameters": 0,
        },
    )


def _pooled_poisson_interaction(tables: np.ndarray) -> InteractionField:
    """Fit the saturated Poisson interaction to each donor-pooled table."""

    values = _validate_tables(tables)
    pooled = values.sum(axis=0).astype(float)
    numerator = pooled[..., 0, 0] * pooled[..., 1, 1]
    denominator = pooled[..., 0, 1] * pooled[..., 1, 0]
    ambiguous = (numerator == 0.0) & (denominator == 0.0)
    if np.any(ambiguous):
        raise CouplingEstimationRefusal(
            "pooled Poisson interaction is not identifiable for an entity"
        )
    boundary = np.zeros(values.shape[1:3], dtype=np.int8)
    boundary[(numerator == 0.0) & (denominator > 0.0)] = -1
    boundary[(numerator > 0.0) & (denominator == 0.0)] = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = np.log(numerator) - np.log(denominator)
    finite = boundary == 0
    if not np.isfinite(log_odds[finite]).all():
        raise FloatingPointError("finite pooled Poisson interactions are invalid")
    return InteractionField(
        log_odds=log_odds,
        boundary=boundary,
        certificate={
            "estimator": "saturated Poisson log-linear pooled interaction",
            "model": (
                "log E[N_ij] = intercept + row_i + column_j "
                "+ beta*I(i=1,j=1), fitted separately per entity"
            ),
            "closed_form": "beta = log(N00*N11/(N01*N10))",
            "donor_pooling": "N_ij = sum_d n_dij before fitting",
            "boundary_entities": int(np.count_nonzero(boundary)),
            "minimum_pooled_cell_count": int(pooled.min()),
            "duplicate_baseline_not_reported": (
                "The saturated Poisson interaction is exactly the donor-pooled "
                "sample log odds ratio."
            ),
        },
    )


def _boundary_table(
    boundary: int, row_margin: np.ndarray, column_margin: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margin, dtype=np.int64)
    columns = np.asarray(column_margin, dtype=np.int64)
    total = int(rows.sum())
    upper_left = (
        max(0, int(rows[0] + columns[0] - total))
        if boundary < 0
        else min(int(rows[0]), int(columns[0]))
    )
    return np.asarray(
        [
            [upper_left, int(rows[0] - upper_left)],
            [int(columns[0] - upper_left), int(rows[1] - columns[0] + upper_left)],
        ],
        dtype=float,
    )


def _predict_field(
    field: InteractionField,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    alpha: float,
) -> np.ndarray:
    rows = np.asarray(row_margins, dtype=np.int64)
    columns = np.asarray(column_margins, dtype=np.int64)
    if (
        rows.shape != (*field.log_odds.shape[:1], 2)
        or columns.shape != (*field.log_odds.shape[1:], 2)
        or alpha <= 0.0
    ):
        raise ValueError("field prediction margins or alpha are invalid")
    prediction = np.empty((*field.log_odds.shape, 2, 2), dtype=float)
    for first, second in np.ndindex(field.log_odds.shape):
        code = int(field.boundary[first, second])
        if code:
            prediction[first, second] = _boundary_table(
                code, rows[first], columns[second]
            )
        else:
            prediction[first, second] = expected_binary_table_from_log_odds(
                float(alpha) * float(field.log_odds[first, second]),
                rows[first],
                columns[second],
            )
    return prediction


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    if values.ndim != 4 or values.shape[-2:] != (2, 2):
        raise ValueError("one entity map must end in 2x2 tables")
    rows = values[:, 0].sum(axis=-1)
    columns = values[0].sum(axis=-2)
    if not np.array_equal(
        values.sum(axis=-1), np.broadcast_to(rows[:, None, :], values.shape[:2] + (2,))
    ) or not np.array_equal(
        values.sum(axis=-2),
        np.broadcast_to(columns[None, :, :], values.shape[:2] + (2,)),
    ):
        raise ValueError("entity tables do not share endpoint margins")
    return rows.astype(np.int64), columns.astype(np.int64)


def _select_alpha_on_panel(
    field: InteractionField,
    target_tables: np.ndarray,
    loss: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    targets = np.asarray(target_tables)
    rows = []
    for alpha in ALPHA_GRID:
        losses = []
        for truth in targets:
            row_margin, column_margin = _margins(truth)
            predicted = _predict_field(field, row_margin, column_margin, alpha)
            losses.append(loss(truth, predicted))
        rows.append(
            {
                "alpha": alpha,
                "mean_loss": float(np.mean(losses)),
                "unit_losses": [float(value) for value in losses],
            }
        )
    selected = min(rows, key=lambda row: (row["mean_loss"], row["alpha"]))
    return {"selected_alpha": selected["alpha"], "candidates": rows}


def _select_alpha_leave_one_out(
    tables: np.ndarray,
    estimator: Callable[[np.ndarray], InteractionField],
    loss: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    values = np.asarray(tables)
    losses = {alpha: np.empty(len(values), dtype=float) for alpha in ALPHA_GRID}
    fold_certificates = []
    for fold, truth in enumerate(values):
        training = np.delete(values, fold, axis=0)
        field = estimator(training)
        fold_certificates.append(field.certificate)
        row_margin, column_margin = _margins(truth)
        for alpha in ALPHA_GRID:
            predicted = _predict_field(field, row_margin, column_margin, alpha)
            losses[alpha][fold] = loss(truth, predicted)
    candidates = [
        {
            "alpha": alpha,
            "mean_loss": float(losses[alpha].mean()),
            "unit_losses": losses[alpha].tolist(),
        }
        for alpha in ALPHA_GRID
    ]
    selected = min(candidates, key=lambda row: (row["mean_loss"], row["alpha"]))
    return {
        "selected_alpha": selected["alpha"],
        "candidates": candidates,
        "fold_certificates": fold_certificates,
    }


def _comparison(
    unit_ids: Iterable[str],
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    units = list(unit_ids)
    primary_values = np.asarray(primary, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if (
        primary_values.shape != (len(units),)
        or comparator_values.shape != primary_values.shape
        or not np.isfinite(primary_values).all()
        or not np.isfinite(comparator_values).all()
        or np.any(comparator_values <= 0.0)
    ):
        raise ValueError("comparison requires positive paired finite losses")
    seed = _seed(label)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(units), size=(BOOTSTRAPS, len(units)), endpoint=False
    )
    difference = primary_values - comparator_values
    boot_difference = difference[indices].mean(axis=1)
    boot_relative = 1.0 - (
        primary_values[indices].mean(axis=1)
        / comparator_values[indices].mean(axis=1)
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    one_sided_sign_p = sum(
        math.comb(len(units), count) for count in range(favorable, len(units) + 1)
    ) / float(2 ** len(units))
    return {
        "units": len(units),
        "unit": "physical donor",
        "primary_mean_loss": float(primary_values.mean()),
        "comparator_mean_loss": float(comparator_values.mean()),
        "relative_loss_reduction": 1.0
        - float(primary_values.mean() / comparator_values.mean()),
        "relative_loss_reduction_paired_bootstrap_95_ci": np.quantile(
            boot_relative, [0.025, 0.975], method="linear"
        ).tolist(),
        "paired_difference_primary_minus_comparator": {
            unit: float(value) for unit, value in zip(units, difference)
        },
        "paired_difference_95_ci": np.quantile(
            boot_difference, [0.025, 0.975], method="linear"
        ).tolist(),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": seed,
        "favorable_donors": favorable,
        "exact_one_sided_sign_p": one_sided_sign_p,
        "post_hoc_inference": True,
    }


def _serialize_field(field: InteractionField) -> dict[str, Any]:
    coordinate: list[list[float | None]] = []
    for first in range(field.log_odds.shape[0]):
        row: list[float | None] = []
        for second in range(field.log_odds.shape[1]):
            value = float(field.log_odds[first, second])
            row.append(value if np.isfinite(value) else None)
        coordinate.append(row)
    return {
        "log_odds": coordinate,
        "boundary": field.boundary.astype(int).tolist(),
        "certificate": field.certificate,
    }


def _stephenson_loss(truth: np.ndarray, prediction: np.ndarray) -> float:
    support = stephenson.numerics._informative(truth).reshape(-1)
    return float(stephenson.numerics._donor_loss(truth, prediction, support))


def _gse239452_loss(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(gse239452._donor_loss(truth, prediction))


def _read_csr_columns_fast(
    matrix: h5py.Group | h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Read sparse row slices without HDF5 multi-point hyperslab selection."""

    selected_rows = np.asarray(rows, dtype=np.int64)
    selected_columns = np.asarray(columns, dtype=np.int64)
    if isinstance(matrix, h5py.Dataset):
        return np.asarray(matrix[selected_rows][:, selected_columns], dtype=float)
    encoding = matrix.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
    if encoding != "csr_matrix" or len(shape) != 2:
        raise ValueError("Stephenson raw layer is not a two-dimensional CSR matrix")
    if (
        np.any(selected_rows < 0)
        or np.any(selected_rows >= shape[0])
        or np.any(selected_columns < 0)
        or np.any(selected_columns >= shape[1])
    ):
        raise IndexError("selected sparse matrix axis is out of range")
    # Reading the 5 MB pointer vector contiguously avoids an extremely slow
    # 50,000-point HDF5 hyperslab construction in the protocol utility.
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    indices_dataset = matrix["indices"]
    data_dataset = matrix["data"]
    column_to_output = {
        int(column): index for index, column in enumerate(selected_columns)
    }
    output = np.zeros((len(selected_rows), len(selected_columns)), dtype=float)
    for output_row, row in enumerate(selected_rows):
        left, right = int(indptr[row]), int(indptr[row + 1])
        indices = np.asarray(indices_dataset[left:right], dtype=np.int64)
        destinations = [
            (offset, column_to_output[int(column)])
            for offset, column in enumerate(indices)
            if int(column) in column_to_output
        ]
        if destinations:
            data = np.asarray(data_dataset[left:right])
            for offset, output_column in destinations:
                output[output_row, output_column] = float(data[offset])
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise ValueError("selected Stephenson counts are not finite and nonnegative")
    return output


def _stephenson_panels(h5ad: Path) -> dict[str, Any]:
    old = os.environ.get("STEPHENSON_CITESEQ_H5AD")
    os.environ["STEPHENSON_CITESEQ_H5AD"] = str(h5ad.resolve())
    try:
        source = stephenson._validated_source(
            stephenson.DEFAULT_SOURCE, verify_hash=False
        )
        calibration_units = stephenson._development_samples(source, "calibration")
        pilot_units = stephenson._development_samples(source, "pilot")
        held_records = stephenson._held_records(source)
        held_units = tuple(record["sample"] for record in held_records)
        requested_units = calibration_units + pilot_units + held_units
        requested_records = [
            source["by_sample"][sample] for sample in requested_units
        ]
        selections = stephenson._selected_rows(source["h5ad"], requested_records)
        all_rows = np.concatenate(
            [selections[sample]["rows"] for sample in requested_units]
        )
        row_order = np.argsort(all_rows, kind="mergesort")
        sorted_rows = all_rows[row_order]
        with h5py.File(source["h5ad"], "r") as handle:
            feature_columns = stephenson._feature_columns(handle)
            rna_columns = np.asarray(feature_columns["rna"], dtype=np.int64)
            adt_columns = np.asarray(feature_columns["adt"], dtype=np.int64)
            combined_columns = np.concatenate((rna_columns, adt_columns))
            matrix = handle["layers/raw"]
            sorted_values = _read_csr_columns_fast(
                matrix, sorted_rows, combined_columns
            )
        values = np.empty_like(sorted_values)
        values[row_order] = sorted_values
        rna_counts: dict[str, np.ndarray] = {}
        adt_counts: dict[str, np.ndarray] = {}
        offset = 0
        for sample in requested_units:
            block = values[offset : offset + stephenson.CELL_BUDGET]
            rna_counts[sample] = block[:, : len(rna_columns)].T
            adt_counts[sample] = block[:, len(rna_columns) :].T
            offset += stephenson.CELL_BUDGET
        rna_states = {
            sample: (
                stephenson.numerics._integer_counts(rna_counts[sample], "RNA") > 0
            ).astype(np.uint8)
            for sample in requested_units
        }
        by_sample = {record["sample"]: record for record in requested_records}
        adt_states = {
            sample: stephenson._adt_states(
                adt_counts[sample],
                selections[sample]["barcodes"],
                by_sample[sample]["donor"],
                sample,
            )
            for sample in requested_units
        }
        all_tables = stephenson.numerics._form_tables(
            rna_states, adt_states, list(requested_units)
        )
        calibration_end = len(calibration_units)
        pilot_end = calibration_end + len(pilot_units)
        calibration = all_tables[:calibration_end]
        pilot = all_tables[calibration_end:pilot_end]
        held = all_tables[pilot_end:]
        frozen = json.loads(stephenson.DEFAULT_PREDICTION.read_text())
        prediction_by_sample = {row["sample"]: row for row in frozen["samples"]}
        primary = np.asarray(
            [
                np.asarray(
                    prediction_by_sample[sample]["predictions"]["primary"],
                    dtype=float,
                ).reshape(len(stephenson.MARKERS), len(stephenson.MARKERS), 2, 2)
                for sample in held_units
            ]
        )
        donor_ids = [by_sample[sample]["donor"] for sample in held_units]
        return {
            "calibration": calibration,
            "pilot": pilot,
            "held": held,
            "held_primary_predictions": primary,
            "calibration_units": list(calibration_units),
            "pilot_units": list(pilot_units),
            "held_units": donor_ids,
        }
    finally:
        if old is None:
            os.environ.pop("STEPHENSON_CITESEQ_H5AD", None)
        else:
            os.environ["STEPHENSON_CITESEQ_H5AD"] = old


def _gse239452_panels(source_root: Path) -> dict[str, Any]:
    manifest = gse239452._sample_manifest(gse239452.DEFAULT_SOURCE)
    preflight = gse239452._preflight_records(gse239452.DEFAULT_PREFLIGHT)
    reduced = json.loads(gse239452.DEFAULT_REDUCED.read_text())
    records = gse239452._validated_reduced_records(
        reduced["samples"], manifest, preflight
    )
    calibration, _, _, _ = gse239452._records_arrays(records, gse239452.CALIBRATION)
    pilot, _, _, _ = gse239452._records_arrays(records, gse239452.PILOT)

    frozen = json.loads(gse239452.DEFAULT_PREDICTION.read_text())
    prediction_by_donor = {row["donor"]: row for row in frozen["samples"]}
    held = []
    primary = []
    for donor in gse239452.HELD:
        truth_record = gse239452._reduce_one(
            donor, source_root, manifest, preflight, read_adt_numeric=True
        )
        truth = np.asarray(truth_record["tables"], dtype=np.int64)
        row_margin, column_margin = gse239452._margins(truth)
        frozen_row = prediction_by_donor[donor]
        if (
            row_margin.tolist() != frozen_row["row_margins"]
            or column_margin.tolist() != frozen_row["column_margins"]
        ):
            raise PermissionError("GSE239452 held margins differ from the freeze")
        held.append(truth)
        primary.append(
            np.asarray(frozen_row["predicted_tables"]["primary"], dtype=float)
        )
    return {
        "calibration": calibration,
        "pilot": pilot,
        "held": np.asarray(held),
        "held_primary_predictions": np.asarray(primary),
        "calibration_units": list(gse239452.CALIBRATION),
        "pilot_units": list(gse239452.PILOT),
        "held_units": list(gse239452.HELD),
    }


def _evaluate_held(
    held: np.ndarray,
    primary_predictions: np.ndarray,
    models: dict[str, tuple[InteractionField, float]],
    loss: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    losses = {"primary": np.empty(len(held), dtype=float)}
    hashes: dict[str, list[str]] = {"primary": []}
    for name in models:
        losses[name] = np.empty(len(held), dtype=float)
        hashes[name] = []
    for index, truth in enumerate(held):
        primary = np.asarray(primary_predictions[index], dtype=float)
        losses["primary"][index] = loss(truth, primary)
        hashes["primary"].append(
            hashlib.sha256(np.ascontiguousarray(primary).tobytes()).hexdigest()
        )
        row_margin, column_margin = _margins(truth)
        for name, (field, alpha) in models.items():
            prediction = _predict_field(field, row_margin, column_margin, alpha)
            losses[name][index] = loss(truth, prediction)
            hashes[name].append(
                hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
            )
    return losses, hashes


def _study_analysis(
    name: str,
    panels: dict[str, Any],
    loss: Callable[[np.ndarray, np.ndarray], float],
    *,
    selection_design: str,
) -> dict[str, Any]:
    calibration = np.asarray(panels["calibration"])
    pilot = np.asarray(panels["pilot"])
    development = np.concatenate((calibration, pilot), axis=0)
    selections: dict[str, dict[str, Any]] = {}
    estimators = {
        "common_effect_exact_cmle": _common_effect_exact_cmle,
        "pooled_poisson_loglinear_interaction": _pooled_poisson_interaction,
    }
    for method, estimator in estimators.items():
        if selection_design == "fit_calibration_select_pilot":
            calibration_field = estimator(calibration)
            selection = _select_alpha_on_panel(calibration_field, pilot, loss)
            selection["calibration_fit"] = _serialize_field(calibration_field)
        elif selection_design == "calibration_leave_one_out":
            selection = _select_alpha_leave_one_out(calibration, estimator, loss)
        else:
            raise ValueError("unknown selection design")
        selections[method] = selection

    fitted = {method: estimator(development) for method, estimator in estimators.items()}
    models = {
        method: (fitted[method], float(selections[method]["selected_alpha"]))
        for method in fitted
    }
    held_losses, prediction_hashes = _evaluate_held(
        np.asarray(panels["held"]),
        np.asarray(panels["held_primary_predictions"]),
        models,
        loss,
    )
    comparisons = {
        f"primary_vs_{method}": _comparison(
            panels["held_units"],
            held_losses["primary"],
            held_losses[method],
            label=f"{name}|primary_vs_{method}",
        )
        for method in estimators
    }
    comparisons[
        "common_effect_exact_cmle_vs_pooled_poisson_loglinear_interaction"
    ] = _comparison(
        panels["held_units"],
        held_losses["common_effect_exact_cmle"],
        held_losses["pooled_poisson_loglinear_interaction"],
        label=(
            f"{name}|common_effect_exact_cmle_vs_"
            "pooled_poisson_loglinear_interaction"
        ),
    )
    pilot_losses = {}
    for method, (field, alpha) in models.items():
        values = []
        for truth in pilot:
            row_margin, column_margin = _margins(truth)
            values.append(loss(truth, _predict_field(field, row_margin, column_margin, alpha)))
        pilot_losses[method] = values
    return {
        "status": "POST_HOC_BASELINE_AUDIT",
        "confirmatory": False,
        "selection_design": selection_design,
        "splits": {
            "calibration": panels["calibration_units"],
            "selection_or_pilot": panels["pilot_units"],
            "held": panels["held_units"],
        },
        "selection": selections,
        "refit_on_calibration_plus_pilot": True,
        "fitted_fields": {
            method: _serialize_field(field) for method, field in fitted.items()
        },
        "held_losses": {
            method: {
                unit: float(value)
                for unit, value in zip(panels["held_units"], values)
            }
            for method, values in held_losses.items()
        },
        "held_prediction_sha256": prediction_hashes,
        "comparisons": comparisons,
        "post_refit_pilot_losses_diagnostic_only": {
            method: {
                unit: float(value)
                for unit, value in zip(panels["pilot_units"], values)
            }
            for method, values in pilot_losses.items()
        },
    }


def _equivalence_boundary() -> dict[str, Any]:
    counterexample = np.asarray(
        [
            [[[1, 9], [11, 3]]],
            [[[8, 2], [1, 9]]],
        ],
        dtype=np.int64,
    ).reshape(2, 1, 1, 2, 2)
    conditional = _common_effect_exact_cmle(counterexample)
    poisson = _pooled_poisson_interaction(counterexample)
    conditional_value = float(conditional.log_odds[0, 0])
    poisson_value = float(poisson.log_odds[0, 0])
    return {
        "poisson_equals_pooled_sample_log_odds": {
            "statement": (
                "For a positive pooled 2x2 table, the row-by-column "
                "coefficient of the saturated Poisson log-linear model is "
                "log(N00*N11/(N01*N10))."
            ),
            "derivation": (
                "Saturation makes the four fitted means equal the four pooled "
                "counts; the interaction contrast log(mu00)+log(mu11)-"
                "log(mu01)-log(mu10) therefore has the stated closed form."
            ),
            "consequence": (
                "A second baseline named pooled log odds would be identical "
                "and is not reported."
            ),
        },
        "poisson_is_not_the_stratified_exact_cmle": {
            "statement": (
                "Pooling removes donor-specific margins. The exact CMLE instead "
                "solves a sum of donor-conditional expectation equations, so "
                "the estimates are not generally equal."
            ),
            "exact_cmle_score": (
                "sum_d E_theta[n_d00 | margins_d] - sum_d n_d00 = 0"
            ),
            "pooled_poisson_score": (
                "beta = log((sum_d n_d00)(sum_d n_d11)/"
                "((sum_d n_d01)(sum_d n_d10)))"
            ),
            "finite_counterexample_tables": counterexample.reshape(2, 2, 2).tolist(),
            "exact_cmle_log_odds": conditional_value,
            "pooled_poisson_log_odds": poisson_value,
            "absolute_difference": abs(conditional_value - poisson_value),
        },
    }


def _tsv(payload: dict[str, Any]) -> str:
    output = io.StringIO()
    fields = (
        "study",
        "comparison",
        "held_donors",
        "primary_mean_loss",
        "comparator_mean_loss",
        "relative_loss_reduction",
        "relative_loss_reduction_ci_low",
        "relative_loss_reduction_ci_high",
        "paired_difference_ci_low",
        "paired_difference_ci_high",
        "favorable_donors",
        "exact_one_sided_sign_p",
        "analysis_status",
    )
    writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for study, result in payload["studies"].items():
        for comparison, row in result["comparisons"].items():
            relative_ci = row["relative_loss_reduction_paired_bootstrap_95_ci"]
            difference_ci = row["paired_difference_95_ci"]
            writer.writerow(
                {
                    "study": study,
                    "comparison": comparison,
                    "held_donors": row["units"],
                    "primary_mean_loss": f"{row['primary_mean_loss']:.12g}",
                    "comparator_mean_loss": f"{row['comparator_mean_loss']:.12g}",
                    "relative_loss_reduction": f"{row['relative_loss_reduction']:.12g}",
                    "relative_loss_reduction_ci_low": f"{relative_ci[0]:.12g}",
                    "relative_loss_reduction_ci_high": f"{relative_ci[1]:.12g}",
                    "paired_difference_ci_low": f"{difference_ci[0]:.12g}",
                    "paired_difference_ci_high": f"{difference_ci[1]:.12g}",
                    "favorable_donors": row["favorable_donors"],
                    "exact_one_sided_sign_p": f"{row['exact_one_sided_sign_p']:.12g}",
                    "analysis_status": "post_hoc_nonconfirmatory",
                }
            )
    return output.getvalue()


def run(
    stephenson_h5ad: Path,
    gse239452_source_root: Path,
    json_path: Path,
    tsv_path: Path,
) -> dict[str, Any]:
    if json_path.exists() or tsv_path.exists():
        raise FileExistsError("post-hoc baseline outputs already exist")
    if not stephenson_h5ad.is_file() or not gse239452_source_root.is_dir():
        raise FileNotFoundError("one or more bound source payloads are absent")
    studies = {
        "stephenson_newcastle_held_site": _study_analysis(
            "stephenson_newcastle_held_site",
            _stephenson_panels(stephenson_h5ad),
            _stephenson_loss,
            selection_design="fit_calibration_select_pilot",
        ),
        "gse239452_held_cohort_post_access_correction": _study_analysis(
            "gse239452_held_cohort_post_access_correction",
            _gse239452_panels(gse239452_source_root),
            _gse239452_loss,
            selection_design="calibration_leave_one_out",
        ),
    }
    payload = {
        "schema": "classical-interaction-baseline-audit/1.0",
        "status": "POST_HOC_NONCONFIRMATORY_BASELINE_AUDIT",
        "confirmatory": False,
        "reason_post_hoc": (
            "Both held outcome panels had already been accessed before these "
            "baseline definitions and comparisons were executed."
        ),
        "methods": {
            "primary": (
                "Existing frozen held predictions from the donor-heterogeneity-aware "
                "penalized composite exact conditional estimator."
            ),
            "common_effect_exact_cmle": (
                "One unpenalized common log odds per RNA--ADT pair, maximizing "
                "the product of donor-specific exact fixed-margin likelihoods."
            ),
            "pooled_poisson_loglinear_interaction": (
                "One saturated Poisson row-by-column interaction per RNA--ADT "
                "pair after summing source tables across donors."
            ),
            "reconstruction": (
                "Every finite field is multiplied by a development-selected alpha "
                "and converted to the noncentral-hypergeometric expected table at "
                "the frozen held donor margins."
            ),
            "loss": (
                "The original study-specific mean Poisson deviance per cell over "
                "informative RNA--ADT pairs."
            ),
        },
        "transport_multiplier_grid": list(ALPHA_GRID),
        "equivalence_boundary": _equivalence_boundary(),
        "studies": studies,
        "bindings": {
            "runner": {
                "path": "experiments/audit_classical_interaction_baselines.py",
                "sha256": _sha256(Path(__file__)),
            },
            "stephenson_frozen_predictions_sha256": _sha256(
                stephenson.DEFAULT_PREDICTION
            ),
            "stephenson_confirmation_sha256": _sha256(stephenson.DEFAULT_SCORE),
            "gse239452_frozen_predictions_sha256": _sha256(
                gse239452.DEFAULT_PREDICTION
            ),
            "gse239452_post_access_correction_sha256": _sha256(
                ROOT / "results/gse239452_citeseq_post_access_correction.json"
            ),
        },
        "inference": {
            "paired_unit": "physical donor",
            "bootstrap_draws": BOOTSTRAPS,
            "base_seed": BOOTSTRAP_SEED,
            "interval": "percentile paired bootstrap, 2.5% and 97.5% quantiles",
            "sign_test": "exact one-sided binomial; zero differences are nonfavorable",
            "interpretation": "descriptive post-hoc uncertainty, not confirmation",
        },
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tsv_text = _tsv(payload)
    _write_exclusive(json_path, json_text)
    try:
        _write_exclusive(tsv_path, tsv_text)
    except Exception:
        json_path.unlink()
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stephenson-h5ad", type=Path, default=DEFAULT_STEPHENSON_H5AD
    )
    parser.add_argument(
        "--gse239452-source-root",
        type=Path,
        default=DEFAULT_GSE239452_SOURCE_ROOT,
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args()
    payload = run(
        args.stephenson_h5ad,
        args.gse239452_source_root,
        args.json,
        args.tsv,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
