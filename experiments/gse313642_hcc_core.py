"""Pure numerical core for the sealed GSE313642 HCC confirmation.

The module has no filesystem or network access.  It converts already reduced
patient tables into source-selected, cohort-conditioned transfer models and
evaluates the prespecified patient-level gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from experiments.gse217494_heart_core import (
    entity_deviance,
    joint_binary_tables,
    predict_conditional_tables,
)
from mapreg.common_effect_conditional import (
    fit_common_effect_conditional_log_odds,
)
from mapreg.heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    signed_deviance_coordinate,
)
from mapreg.poisson_loglinear import (
    PoissonLoglinearFit,
    PoissonLoglinearRefusal,
    fit_poisson_loglinear_interaction,
    reconstruct_poisson_tables,
)
from mapreg.structured_context_conditional import (
    StructuredContextConditionalFit,
    fit_structured_context_conditional_log_odds,
)


MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
COHORTS = ("A", "B")
CELL_BUDGET = 512
ADT_HIGH_COUNT = 256
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 31_364_201
CELL_SALT = "GSE313642-HCC-CELL-v1"
ADT_TIE_SALT = "GSE313642-HCC-FB-TIE-v1"
DESTROY_SALT = "GSE313642-HCC-DESTROY-v1"

SOURCE_GATE_COMPARATORS = (
    "cohort_poisson",
    "cohort_signed_deviance",
    "destroyed_links",
)
HELD_GATE_COMPARATORS = SOURCE_GATE_COMPARATORS
ALL_METHODS = (
    ("primary",)
    + SOURCE_GATE_COMPARATORS
    + ("cohort_exact_cmle", "independence", "pooled_poisson")
)


@dataclass(frozen=True, order=True)
class PrimaryConfig:
    """One prespecified primary configuration in tie-break order."""

    donor_deviation_penalty: float
    transport_multiplier: float
    coefficient_ridge_penalty: float = 0.01
    graph_penalty: float = 0.0


CONFIGURATIONS = tuple(
    PrimaryConfig(eta, alpha) for eta in (0.1, 1.0) for alpha in (0.75, 1.0)
)


def _digest(salt: str, *parts: str) -> bytes:
    payload = salt.encode("utf-8")
    for part in parts:
        payload += b"\0" + part.encode("utf-8")
    return hashlib.sha256(payload).digest()


def _unique_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    axis = tuple(values)
    if not axis or any(not isinstance(value, str) or not value for value in axis):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(axis)) != len(axis):
        raise ValueError(f"{label} must be unique")
    return axis


def select_barcodes(
    barcodes: Sequence[str], sample_id: str, *, count: int = CELL_BUDGET
) -> tuple[str, ...]:
    """Select cells by a salted identifier rank, independent of axis order."""

    axis = _unique_strings(barcodes, "barcodes")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be nonempty")
    requested = int(count)
    if requested < 1 or requested > len(axis):
        raise ValueError("requested barcode count is outside the shared axis")
    return tuple(
        sorted(
            axis,
            key=lambda barcode: (
                _digest(CELL_SALT, sample_id, barcode),
                barcode,
            ),
        )[:requested]
    )


def rna_detection_states(counts: np.ndarray) -> np.ndarray:
    """Binarize selected GEX counts at one detected UMI."""

    values = np.asarray(counts)
    if values.shape != (CELL_BUDGET, len(MARKERS)):
        raise ValueError("GEX counts must be 512 cells by nine markers")
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("GEX counts must be nonnegative integers")
    return (values > 0).astype(np.uint8)


def adt_midrank_states(
    counts: np.ndarray, barcodes: Sequence[str], sample_id: str
) -> np.ndarray:
    """Assign exactly half the selected cells to the high state per protein."""

    values = np.asarray(counts)
    axis = _unique_strings(barcodes, "selected barcodes")
    if values.shape != (CELL_BUDGET, len(MARKERS)) or len(axis) != CELL_BUDGET:
        raise ValueError("FB counts and barcodes must cover the selected 512 cells")
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("FB counts must be nonnegative integers")
    output = np.zeros(values.shape, dtype=np.uint8)
    for marker_index, marker in enumerate(MARKERS):
        order = sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                -int(values[cell, marker_index]),
                _digest(ADT_TIE_SALT, sample_id, marker, axis[cell]),
                axis[cell],
            ),
        )
        output[np.asarray(order[:ADT_HIGH_COUNT]), marker_index] = 1
    if not np.all(output.sum(axis=0) == ADT_HIGH_COUNT):
        raise AssertionError("midrank states changed their frozen margin")
    return output


def destroy_adt_states(
    states: np.ndarray, barcodes: Sequence[str], sample_id: str
) -> np.ndarray:
    """Destroy pairing by a fixed half-cycle shift of complete ADT vectors."""

    values = np.asarray(states)
    axis = _unique_strings(barcodes, "selected barcodes")
    if values.shape != (CELL_BUDGET, len(MARKERS)) or len(axis) != CELL_BUDGET:
        raise ValueError("ADT states and barcodes must cover 512 selected cells")
    order = np.asarray(
        sorted(
            range(CELL_BUDGET),
            key=lambda cell: (
                _digest(DESTROY_SALT, sample_id, axis[cell]),
                axis[cell],
            ),
        ),
        dtype=np.int64,
    )
    output = np.empty_like(values)
    output[order] = values[np.roll(order, -ADT_HIGH_COUNT)]
    if not np.array_equal(output.sum(axis=0), values.sum(axis=0)):
        raise AssertionError("destroyed link changed ADT margins")
    return output


def patient_tables(
    gex_counts: np.ndarray,
    fb_counts: np.ndarray,
    barcodes: Sequence[str],
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the 81 real and destroyed ordered-pair tables."""

    rna = rna_detection_states(gex_counts)
    adt = adt_midrank_states(fb_counts, barcodes, sample_id)
    destroyed = destroy_adt_states(adt, barcodes, sample_id)
    return joint_binary_tables(rna, adt), joint_binary_tables(rna, destroyed)


def cohort_design(cohorts: Sequence[str]) -> np.ndarray:
    """Encode A and B as symmetric one-hot contexts."""

    labels = tuple(cohorts)
    if not labels or any(label not in COHORTS for label in labels):
        raise ValueError("cohort labels must be A or B")
    design = np.zeros((len(labels), len(COHORTS)), dtype=float)
    design[np.arange(len(labels)), [COHORTS.index(label) for label in labels]] = 1.0
    return design


def _margins(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    return values.sum(axis=-1), values.sum(axis=-2)


def _loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(entity_deviance(observed, predicted)))


def _fit_primary(
    tables: np.ndarray, cohorts: Sequence[str], config: PrimaryConfig
) -> StructuredContextConditionalFit:
    return fit_structured_context_conditional_log_odds(
        np.asarray(tables),
        cohort_design(cohorts),
        donor_deviation_penalty=config.donor_deviation_penalty,
        coefficient_ridge_penalty=config.coefficient_ridge_penalty,
        graph_penalty=config.graph_penalty,
        minimum_informative_donors=2,
        maximum_condition_number=1e12,
        maximum_iterations=100,
        tolerance=1e-8,
    )


def _primary_field(
    fit: StructuredContextConditionalFit, cohort: str, alpha: float
) -> np.ndarray:
    if cohort not in COHORTS:
        raise ValueError("unknown cohort")
    return float(alpha) * np.asarray(fit.coefficient[COHORTS.index(cohort)])


def _predict_log_odds(field: np.ndarray, truth: np.ndarray) -> np.ndarray:
    rows, columns = _margins(truth)
    return predict_conditional_tables(np.asarray(field), rows, columns)


def _predict_log_odds_at_margins(
    field: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    return predict_conditional_tables(
        np.asarray(field), np.asarray(rows), np.asarray(columns)
    )


def _cohort_index(labels: Sequence[str], cohort: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(tuple(labels), dtype=object) == cohort)


def _fit_cohort_cmle(
    tables: np.ndarray, cohorts: Sequence[str]
) -> dict[str, np.ndarray]:
    output = {}
    for cohort in COHORTS:
        indices = _cohort_index(cohorts, cohort)
        if len(indices) < 2:
            raise CouplingEstimationRefusal("cohort CMLE needs two patients")
        output[cohort] = fit_common_effect_conditional_log_odds(
            np.asarray(tables)[indices], minimum_informative_donors=2
        ).log_odds
    return output


def _fit_cohort_poisson(
    tables: np.ndarray, cohorts: Sequence[str]
) -> PoissonLoglinearFit:
    return fit_poisson_loglinear_interaction(
        np.asarray(tables), np.asarray(tuple(cohorts), dtype=object)
    )


def _poisson_field(fit: PoissonLoglinearFit, cohort: str) -> np.ndarray:
    return np.asarray(fit.log_odds[fit.group_labels.index(cohort)])


def _fit_cohort_residual(
    tables: np.ndarray, cohorts: Sequence[str]
) -> dict[str, np.ndarray]:
    values = np.asarray(tables)
    output = {}
    for cohort in COHORTS:
        selected = values[_cohort_index(cohorts, cohort)]
        coordinates = np.empty(selected.shape[:3], dtype=float)
        for index in np.ndindex(coordinates.shape):
            coordinates[index] = signed_deviance_coordinate(selected[index])
        output[cohort] = coordinates.mean(axis=0)
    return output


def _fractional_signed_deviance(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    positive = values > 0.0
    deviance = 2.0 * float(
        np.sum(values[positive] * np.log(values[positive] / expected[positive]))
    )
    determinant = float(values[0, 0] * values[1, 1] - values[0, 1] * values[1, 0])
    return float(np.sign(determinant) * math.sqrt(max(0.0, deviance)))


def _residual_table(
    coordinate: float, row_margin: np.ndarray, column_margin: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    total = float(rows.sum())
    lower = max(0.0, float(rows[0] + columns[0] - total))
    upper = min(float(rows[0]), float(columns[0]))

    def table_at(value: float) -> np.ndarray:
        return np.asarray(
            [
                [value, rows[0] - value],
                [columns[0] - value, rows[1] - columns[0] + value],
            ]
        )

    if upper <= lower:
        return table_at(lower)
    epsilon = min(1e-8, 0.25 * (upper - lower))
    left, right = lower + epsilon, upper - epsilon
    target = min(
        max(float(coordinate), _fractional_signed_deviance(table_at(left))),
        _fractional_signed_deviance(table_at(right)),
    )
    for _ in range(96):
        midpoint = 0.5 * (left + right)
        if _fractional_signed_deviance(table_at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    return table_at(0.5 * (left + right))


def _predict_residual(field: np.ndarray, truth: np.ndarray) -> np.ndarray:
    rows, columns = _margins(truth)
    return _predict_residual_at_margins(field, rows, columns)


def _predict_residual_at_margins(
    field: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    output = np.empty((*np.asarray(field).shape, 2, 2), dtype=float)
    for index in np.ndindex(np.asarray(field).shape):
        output[index] = _residual_table(
            float(field[index]), rows[index], columns[index]
        )
    return output


def select_primary_configuration(
    calibration_tables: np.ndarray, calibration_cohorts: Sequence[str]
) -> tuple[PrimaryConfig, dict[PrimaryConfig, np.ndarray]]:
    """Select eta and transport by calibration-only leave-patient-out loss."""

    tables = np.asarray(calibration_tables)
    labels = tuple(calibration_cohorts)
    if tables.shape != (11, len(MARKERS), len(MARKERS), 2, 2) or len(labels) != 11:
        raise ValueError("calibration must contain the frozen eleven patients")
    losses = {config: np.empty(11, dtype=float) for config in CONFIGURATIONS}
    axis = np.arange(11)
    for validation in range(11):
        training = axis[axis != validation]
        for eta in (0.1, 1.0):
            representative = PrimaryConfig(eta, 0.75)
            fit = _fit_primary(
                tables[training], [labels[index] for index in training], representative
            )
            for alpha in (0.75, 1.0):
                config = PrimaryConfig(eta, alpha)
                prediction = _predict_log_odds(
                    _primary_field(fit, labels[validation], alpha), tables[validation]
                )
                losses[config][validation] = _loss(tables[validation], prediction)
    selected = min(
        CONFIGURATIONS, key=lambda config: (float(losses[config].mean()), config)
    )
    return selected, losses


def select_comparator_alphas(
    calibration_tables: np.ndarray, calibration_cohorts: Sequence[str]
) -> tuple[dict[str, float], dict[str, dict[float, np.ndarray]]]:
    """Select matched comparator transport using calibration LOPO only."""

    tables = np.asarray(calibration_tables)
    labels = tuple(calibration_cohorts)
    if tables.shape != (11, len(MARKERS), len(MARKERS), 2, 2) or len(labels) != 11:
        raise ValueError("calibration must contain the frozen eleven patients")
    methods = ("cohort_poisson", "cohort_signed_deviance")
    losses = {
        method: {alpha: np.empty(11, dtype=float) for alpha in (0.75, 1.0)}
        for method in methods
    }
    axis = np.arange(11)
    for validation in range(11):
        training = axis[axis != validation]
        training_tables = tables[training]
        training_labels = [labels[index] for index in training]
        poisson = _fit_cohort_poisson(training_tables, training_labels)
        residual = _fit_cohort_residual(training_tables, training_labels)
        truth = tables[validation]
        rows, columns = _margins(truth)
        cohort = labels[validation]
        for alpha in (0.75, 1.0):
            poisson_prediction = reconstruct_poisson_tables(
                _poisson_field(poisson, cohort),
                rows,
                columns,
                transport_scale=alpha,
            ).table
            residual_prediction = _predict_residual(alpha * residual[cohort], truth)
            losses["cohort_poisson"][alpha][validation] = _loss(
                truth, poisson_prediction
            )
            losses["cohort_signed_deviance"][alpha][validation] = _loss(
                truth, residual_prediction
            )
    selected = {
        method: min(
            (0.75, 1.0),
            key=lambda alpha: (float(losses[method][alpha].mean()), alpha),
        )
        for method in methods
    }
    return selected, losses


def fit_models(
    tables: np.ndarray,
    destroyed_tables: np.ndarray,
    cohorts: Sequence[str],
    config: PrimaryConfig,
    comparator_alphas: Mapping[str, float],
) -> dict[str, object]:
    """Fit the primary field and all matched source-only comparators."""

    values = np.asarray(tables)
    destroyed = np.asarray(destroyed_tables)
    labels = tuple(cohorts)
    if values.shape != destroyed.shape or values.shape[0] != len(labels):
        raise ValueError("real, destroyed, and cohort patient axes must agree")
    expected_alphas = {"cohort_poisson", "cohort_signed_deviance"}
    if set(comparator_alphas) != expected_alphas or any(
        float(value) not in (0.75, 1.0) for value in comparator_alphas.values()
    ):
        raise ValueError("matched comparator alphas differ from the frozen grid")
    primary = _fit_primary(values, labels, config)
    destroyed_fit = _fit_primary(destroyed, labels, config)
    try:
        pooled_poisson: PoissonLoglinearFit | None = fit_poisson_loglinear_interaction(
            values
        )
        pooled_poisson_refusal = None
    except (ValueError, FloatingPointError, PoissonLoglinearRefusal):
        pooled_poisson = None
        pooled_poisson_refusal = "FINITE_POOLED_POISSON_UNAVAILABLE"
    try:
        cohort_cmle: dict[str, np.ndarray] | None = _fit_cohort_cmle(values, labels)
        cohort_cmle_refusal = None
    except (ValueError, FloatingPointError, CouplingEstimationRefusal):
        cohort_cmle = None
        cohort_cmle_refusal = "FINITE_COHORT_EXACT_CMLE_UNAVAILABLE"
    return {
        "primary": primary,
        "destroyed_links": destroyed_fit,
        "cohort_poisson": _fit_cohort_poisson(values, labels),
        "pooled_poisson": pooled_poisson,
        "pooled_poisson_refusal": pooled_poisson_refusal,
        "cohort_signed_deviance": _fit_cohort_residual(values, labels),
        "cohort_exact_cmle": cohort_cmle,
        "cohort_exact_cmle_refusal": cohort_cmle_refusal,
        "independence": None,
        "configuration": config,
        "comparator_alphas": {
            name: float(comparator_alphas[name]) for name in sorted(expected_alphas)
        },
    }


def predict_models(
    models: Mapping[str, object], truth: np.ndarray, cohort: str
) -> dict[str, np.ndarray]:
    """Predict one patient's 81 tables at that patient's observed margins."""

    config = models["configuration"]
    if not isinstance(config, PrimaryConfig):
        raise TypeError("models lack a PrimaryConfig")
    observed = np.asarray(truth)
    rows, columns = _margins(observed)
    primary = models["primary"]
    destroyed = models["destroyed_links"]
    if not isinstance(primary, StructuredContextConditionalFit) or not isinstance(
        destroyed, StructuredContextConditionalFit
    ):
        raise TypeError("structured fits are absent")
    cohort_poisson = models["cohort_poisson"]
    pooled_poisson = models["pooled_poisson"]
    if not isinstance(cohort_poisson, PoissonLoglinearFit):
        raise TypeError("mandatory cohort Poisson fit is absent")
    residual = models["cohort_signed_deviance"]
    cmle = models["cohort_exact_cmle"]
    if not isinstance(residual, dict):
        raise TypeError("cohort residual fields are absent")
    comparator_alphas = models.get("comparator_alphas")
    if not isinstance(comparator_alphas, dict):
        raise TypeError("matched comparator alphas are absent")
    output = {
        "primary": _predict_log_odds(
            _primary_field(primary, cohort, config.transport_multiplier), observed
        ),
        "destroyed_links": _predict_log_odds(
            _primary_field(destroyed, cohort, config.transport_multiplier), observed
        ),
        "cohort_poisson": reconstruct_poisson_tables(
            _poisson_field(cohort_poisson, cohort),
            rows,
            columns,
            transport_scale=float(comparator_alphas["cohort_poisson"]),
        ).table,
        "cohort_signed_deviance": _predict_residual(
            float(comparator_alphas["cohort_signed_deviance"]) * residual[cohort],
            observed,
        ),
        "independence": _predict_log_odds(np.zeros(observed.shape[:-2]), observed),
    }
    if isinstance(pooled_poisson, PoissonLoglinearFit):
        output["pooled_poisson"] = reconstruct_poisson_tables(
            np.asarray(pooled_poisson.log_odds[0]), rows, columns
        ).table
    if isinstance(cmle, dict):
        output["cohort_exact_cmle"] = _predict_log_odds(cmle[cohort], observed)
    return output


def panel_losses(
    models: Mapping[str, object], tables: np.ndarray, cohorts: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return patient-equal losses for every method."""

    values = np.asarray(tables)
    labels = tuple(cohorts)
    if values.shape[0] != len(labels):
        raise ValueError("tables and cohorts must share the patient axis")
    output: dict[str, np.ndarray] = {}
    for patient, cohort in enumerate(labels):
        predictions = predict_models(models, values[patient], cohort)
        if patient == 0:
            output = {
                method: np.empty(len(labels), dtype=float) for method in predictions
            }
        elif set(predictions) != set(output):
            raise AssertionError("prediction availability changed across patients")
        for method in predictions:
            output[method][patient] = _loss(values[patient], predictions[method])
    return output


def _sign_probability(differences: np.ndarray) -> float:
    nonzero = np.asarray(differences)[np.asarray(differences) != 0.0]
    favorable = int(np.count_nonzero(nonzero < 0.0))
    count = len(nonzero)
    return float(
        sum(math.comb(count, index) for index in range(favorable, count + 1))
        / (2**count)
    )


def _bootstrap(
    differences: np.ndarray,
    cohorts: Sequence[str],
    *,
    seed: int,
    draws: int = BOOTSTRAPS,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    labels = np.asarray(tuple(cohorts), dtype=object)
    if values.shape != labels.shape or set(labels.tolist()) != set(COHORTS):
        raise ValueError("bootstrap requires paired A/B patient losses")
    generator = np.random.default_rng(seed)
    distributions = []
    for cohort in COHORTS:
        selected = values[labels == cohort]
        indices = generator.integers(0, len(selected), size=(int(draws), len(selected)))
        distributions.append(selected[indices])
    means = np.concatenate(distributions, axis=1).mean(axis=1)
    interval = np.quantile(means, (0.025, 0.975), method="linear")
    return float(interval[0]), float(interval[1])


def source_gate(
    losses: Mapping[str, np.ndarray], cohorts: Sequence[str]
) -> dict[str, object]:
    """Apply the untouched 12-patient pilot promotion gate."""

    primary = np.asarray(losses["primary"], dtype=float)
    labels = np.asarray(tuple(cohorts), dtype=object)
    if (
        primary.shape != (12,)
        or labels.shape != primary.shape
        or set(labels.tolist()) != set(COHORTS)
        or not np.isfinite(primary).all()
    ):
        raise ValueError("source gate requires twelve finite pilot losses")
    comparisons = {}
    for method in SOURCE_GATE_COMPARATORS:
        comparator = np.asarray(losses[method], dtype=float)
        difference = primary - comparator
        checks = {
            "primary_mean_strictly_lower": float(difference.mean()) < 0.0,
            "at_least_eight_of_twelve_favorable": int(
                np.count_nonzero(difference < 0.0)
            )
            >= 8,
            "A_mean_improvement_strictly_positive": float(
                difference[labels == "A"].mean()
            )
            < 0.0,
            "B_mean_improvement_strictly_positive": float(
                difference[labels == "B"].mean()
            )
            < 0.0,
        }
        comparisons[method] = {
            "primary_mean_loss": float(primary.mean()),
            "comparator_mean_loss": float(comparator.mean()),
            "mean_difference": float(difference.mean()),
            "favorable_patients": int(np.count_nonzero(difference < 0.0)),
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "comparisons": comparisons,
        "passes": all(record["passes"] for record in comparisons.values()),
    }


def held_gate(
    losses: Mapping[str, np.ndarray], cohorts: Sequence[str]
) -> dict[str, object]:
    """Apply the frozen 12-patient confirmatory gate to every matched method."""

    primary = np.asarray(losses["primary"], dtype=float)
    labels = tuple(cohorts)
    if primary.shape != (12,) or len(labels) != 12 or not np.isfinite(primary).all():
        raise ValueError("held gate requires twelve finite patient losses")
    comparisons = {}
    for method in HELD_GATE_COMPARATORS:
        comparator = np.asarray(losses[method], dtype=float)
        difference = primary - comparator
        interval = _bootstrap(difference, labels, seed=BOOTSTRAP_SEED, draws=BOOTSTRAPS)
        reduction = 1.0 - float(primary.mean() / comparator.mean())
        favorable = int(np.count_nonzero(difference < 0.0))
        sign_probability = _sign_probability(difference)
        checks = {
            "relative_reduction_at_least_five_percent": reduction >= 0.05,
            "paired_patient_bootstrap_upper_95_below_zero": interval[1] < 0.0,
            "at_least_ten_of_twelve_favorable": favorable >= 10,
            "one_sided_sign_probability_at_most_0_025": sign_probability <= 0.025,
            "A_mean_improvement_strictly_positive": float(
                difference[np.asarray(labels) == "A"].mean()
            )
            < 0.0,
            "B_mean_improvement_strictly_positive": float(
                difference[np.asarray(labels) == "B"].mean()
            )
            < 0.0,
        }
        comparisons[method] = {
            "primary_mean_loss": float(primary.mean()),
            "comparator_mean_loss": float(comparator.mean()),
            "relative_reduction": reduction,
            "mean_difference": float(difference.mean()),
            "paired_patient_bootstrap_95_interval": interval,
            "favorable_patients": favorable,
            "one_sided_sign_probability": sign_probability,
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "comparisons": comparisons,
        "shared_stratified_bootstrap": {
            "draws": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "cohort_order": list(COHORTS),
            "same_draw_index_tensor_for_every_comparator": True,
        },
        "passes": all(record["passes"] for record in comparisons.values()),
    }


def serialize_models(models: Mapping[str, object]) -> dict[str, object]:
    """Serialize source-only fields and compact numerical certificates."""

    config = models["configuration"]
    primary = models["primary"]
    destroyed = models["destroyed_links"]
    cohort_poisson = models["cohort_poisson"]
    pooled_poisson = models["pooled_poisson"]
    if not isinstance(config, PrimaryConfig):
        raise TypeError("configuration is absent")
    if not isinstance(primary, StructuredContextConditionalFit) or not isinstance(
        destroyed, StructuredContextConditionalFit
    ):
        raise TypeError("structured fits are absent")
    if not isinstance(cohort_poisson, PoissonLoglinearFit):
        raise TypeError("mandatory cohort Poisson fit is absent")

    def structured(fit: StructuredContextConditionalFit) -> dict[str, object]:
        return {
            "cohort_log_odds": np.asarray(fit.coefficient).tolist(),
            "certificate": {
                "converged": fit.converged,
                "iterations": fit.iterations,
                "gradient_norm": fit.gradient_norm,
                "scaled_gradient_norm": fit.scaled_gradient_norm,
                "schur_condition_number": fit.schur_condition_number,
            },
        }

    cmle = models["cohort_exact_cmle"]
    residual = models["cohort_signed_deviance"]
    comparator_alphas = models.get("comparator_alphas")
    if not isinstance(residual, dict) or not isinstance(comparator_alphas, dict):
        raise TypeError("cohort fields are absent")
    output = {
        "configuration": asdict(config),
        "comparator_alphas": {
            name: float(comparator_alphas[name])
            for name in ("cohort_poisson", "cohort_signed_deviance")
        },
        "cohort_order": list(COHORTS),
        "primary": structured(primary),
        "destroyed_links": structured(destroyed),
        "cohort_poisson": {
            "group_labels": list(cohort_poisson.group_labels),
            "log_odds": cohort_poisson.log_odds.tolist(),
            "maximum_scaled_score": cohort_poisson.maximum_scaled_score,
        },
        "cohort_signed_deviance": {
            cohort: np.asarray(residual[cohort]).tolist() for cohort in COHORTS
        },
        "independence": {"log_odds": 0.0},
        "report_only_refusals": {},
    }
    if isinstance(pooled_poisson, PoissonLoglinearFit):
        output["pooled_poisson"] = {
            "status": "VALID",
            "log_odds": pooled_poisson.log_odds[0].tolist(),
            "maximum_scaled_score": pooled_poisson.maximum_scaled_score,
        }
    else:
        output["report_only_refusals"]["pooled_poisson"] = models.get(
            "pooled_poisson_refusal"
        )
    if isinstance(cmle, dict):
        output["cohort_exact_cmle"] = {
            "status": "VALID",
            "fields": {cohort: np.asarray(cmle[cohort]).tolist() for cohort in COHORTS},
        }
    else:
        output["report_only_refusals"]["cohort_exact_cmle"] = models.get(
            "cohort_exact_cmle_refusal"
        )
    return output


def deserialize_models(payload: Mapping[str, object]) -> dict[str, object]:
    """Restore prediction-only source fields without refitting held values."""

    configuration = payload.get("configuration")
    comparator_alphas = payload.get("comparator_alphas")
    if not isinstance(configuration, dict):
        raise ValueError("serialized configuration is absent")
    if not isinstance(comparator_alphas, dict):
        raise ValueError("serialized comparator alphas are absent")
    config = PrimaryConfig(**configuration)
    cohort_order = payload.get("cohort_order")
    if cohort_order != list(COHORTS):
        raise ValueError("serialized cohort order differs")

    class _CoefficientOnly:
        def __init__(self, coefficient: np.ndarray):
            self.coefficient = coefficient

    primary_payload = payload.get("primary")
    destroyed_payload = payload.get("destroyed_links")
    if not isinstance(primary_payload, dict) or not isinstance(destroyed_payload, dict):
        raise ValueError("serialized structured models are absent")
    primary = _CoefficientOnly(
        np.asarray(primary_payload["cohort_log_odds"], dtype=float)
    )
    destroyed = _CoefficientOnly(
        np.asarray(destroyed_payload["cohort_log_odds"], dtype=float)
    )
    poisson_payload = payload.get("cohort_poisson")
    pooled_payload = payload.get("pooled_poisson")
    residual = payload.get("cohort_signed_deviance")
    cmle = payload.get("cohort_exact_cmle")
    if not all(isinstance(item, dict) for item in (poisson_payload, residual)):
        raise ValueError("serialized mandatory comparator models are absent")
    output = {
        "configuration": config,
        "comparator_alphas": {
            name: float(comparator_alphas[name])
            for name in ("cohort_poisson", "cohort_signed_deviance")
        },
        "primary": primary,
        "destroyed_links": destroyed,
        "cohort_poisson_fields": {
            label: np.asarray(field, dtype=float)
            for label, field in zip(
                poisson_payload["group_labels"], poisson_payload["log_odds"]
            )
        },
        "cohort_signed_deviance": {
            cohort: np.asarray(residual[cohort], dtype=float) for cohort in COHORTS
        },
        "independence": None,
    }
    if isinstance(pooled_payload, dict) and pooled_payload.get("status") == "VALID":
        output["pooled_poisson_field"] = np.asarray(
            pooled_payload["log_odds"], dtype=float
        )
    if isinstance(cmle, dict) and cmle.get("status") == "VALID":
        output["cohort_exact_cmle"] = {
            cohort: np.asarray(cmle["fields"][cohort], dtype=float)
            for cohort in COHORTS
        }
    output["report_only_refusals"] = dict(payload.get("report_only_refusals", {}))
    return output


def predict_serialized_models(
    models: Mapping[str, object], truth: np.ndarray, cohort: str
) -> dict[str, np.ndarray]:
    """Predict from the frozen source serialization in the held stage."""

    config = models["configuration"]
    if not isinstance(config, PrimaryConfig) or cohort not in COHORTS:
        raise ValueError("prediction configuration or cohort differs")
    observed = np.asarray(truth)
    rows, columns = _margins(observed)
    primary = np.asarray(getattr(models["primary"], "coefficient"))
    destroyed = np.asarray(getattr(models["destroyed_links"], "coefficient"))
    poisson = models["cohort_poisson_fields"]
    residual = models["cohort_signed_deviance"]
    cmle = models.get("cohort_exact_cmle")
    comparator_alphas = models["comparator_alphas"]
    context = COHORTS.index(cohort)
    output = {
        "primary": _predict_log_odds(
            config.transport_multiplier * primary[context], observed
        ),
        "destroyed_links": _predict_log_odds(
            config.transport_multiplier * destroyed[context], observed
        ),
        "cohort_poisson": reconstruct_poisson_tables(
            poisson[cohort],
            rows,
            columns,
            transport_scale=float(comparator_alphas["cohort_poisson"]),
        ).table,
        "cohort_signed_deviance": _predict_residual(
            float(comparator_alphas["cohort_signed_deviance"]) * residual[cohort],
            observed,
        ),
        "independence": _predict_log_odds(np.zeros(observed.shape[:-2]), observed),
    }
    if "pooled_poisson_field" in models:
        output["pooled_poisson"] = reconstruct_poisson_tables(
            models["pooled_poisson_field"], rows, columns
        ).table
    if isinstance(cmle, dict):
        output["cohort_exact_cmle"] = _predict_log_odds(cmle[cohort], observed)
    return output


def predict_serialized_at_margins(
    models: Mapping[str, object],
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    cohort: str,
) -> dict[str, np.ndarray]:
    """Freeze held predictions from GEX margins and the fixed FB midrank margin."""

    rows = np.asarray(row_margins)
    columns = np.asarray(column_margins)
    expected = (len(MARKERS), len(MARKERS), 2)
    if (
        rows.shape != expected
        or columns.shape != expected
        or np.any(rows.sum(axis=-1) != CELL_BUDGET)
        or np.any(columns != np.asarray((ADT_HIGH_COUNT, ADT_HIGH_COUNT)))
    ):
        raise ValueError("held margins differ from the frozen 81-table design")
    config = models["configuration"]
    if not isinstance(config, PrimaryConfig) or cohort not in COHORTS:
        raise ValueError("prediction configuration or cohort differs")
    primary = np.asarray(getattr(models["primary"], "coefficient"))
    destroyed = np.asarray(getattr(models["destroyed_links"], "coefficient"))
    poisson = models["cohort_poisson_fields"]
    residual = models["cohort_signed_deviance"]
    cmle = models.get("cohort_exact_cmle")
    comparator_alphas = models["comparator_alphas"]
    context = COHORTS.index(cohort)
    output = {
        "primary": _predict_log_odds_at_margins(
            config.transport_multiplier * primary[context], rows, columns
        ),
        "destroyed_links": _predict_log_odds_at_margins(
            config.transport_multiplier * destroyed[context], rows, columns
        ),
        "cohort_poisson": reconstruct_poisson_tables(
            poisson[cohort],
            rows,
            columns,
            transport_scale=float(comparator_alphas["cohort_poisson"]),
        ).table,
        "cohort_signed_deviance": _predict_residual_at_margins(
            float(comparator_alphas["cohort_signed_deviance"]) * residual[cohort],
            rows,
            columns,
        ),
        "independence": _predict_log_odds_at_margins(
            np.zeros(rows.shape[:-1]), rows, columns
        ),
    }
    if "pooled_poisson_field" in models:
        output["pooled_poisson"] = reconstruct_poisson_tables(
            models["pooled_poisson_field"], rows, columns
        ).table
    if isinstance(cmle, dict):
        output["cohort_exact_cmle"] = _predict_log_odds_at_margins(
            cmle[cohort], rows, columns
        )
    return output


def serialized_panel_losses(
    models: Mapping[str, object], tables: np.ndarray, cohorts: Sequence[str]
) -> dict[str, np.ndarray]:
    """Score held patients from source-serialized fields without refitting."""

    values = np.asarray(tables)
    labels = tuple(cohorts)
    if values.shape[0] != len(labels):
        raise ValueError("tables and cohorts must share the patient axis")
    output: dict[str, np.ndarray] = {}
    for patient, cohort in enumerate(labels):
        predictions = predict_serialized_models(models, values[patient], cohort)
        if patient == 0:
            output = {
                method: np.empty(len(labels), dtype=float) for method in predictions
            }
        elif set(predictions) != set(output):
            raise AssertionError("prediction availability changed across patients")
        for method in predictions:
            output[method][patient] = _loss(values[patient], predictions[method])
    return output
