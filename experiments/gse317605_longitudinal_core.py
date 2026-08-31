"""Frozen estimator and decision rules for the GSE317605 campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from mapreg.coupling_fields import normalized_hypergraph_laplacian
from mapreg.heterogeneity_adaptive_coupling import (
    _fixed_margin_support,
    _log_choose,
    expected_binary_table_from_log_odds,
)
from mapreg.poisson_loglinear import (
    fit_ridge_profiled_poisson_interaction,
    reconstruct_poisson_tables,
)


TIMEPOINTS = ("T01", "T02", "T03", "T04")
MARKERS = (
    "CD3E",
    "CD4",
    "CD8A",
    "MS4A1",
    "CD14",
    "FCGR3A",
    "NCAM1",
    "IL7R",
    "KLRD1",
    "SELL",
    "CD27",
    "CD38",
    "ITGAM",
    "CD2",
    "CD44",
    "CD7",
)
ADT_MARKERS = (
    "CD3",
    "CD4",
    "CD8",
    "CD20",
    "CD14",
    "CD16",
    "CD56",
    "CD127",
    "CD94",
    "CD62L",
    "CD27",
    "CD38",
    "CD11b",
    "CD2",
    "CD44",
    "CD7",
)
HYPEREDGES = (
    ("T", ("CD3E", "CD4", "CD8A", "IL7R", "SELL", "CD27", "CD2", "CD7")),
    ("B", ("MS4A1", "CD27", "CD38")),
    ("NK", ("FCGR3A", "NCAM1", "KLRD1", "CD2", "CD7")),
    ("myeloid", ("CD14", "FCGR3A", "ITGAM", "CD44")),
    ("activation_memory", ("IL7R", "SELL", "CD27", "CD38", "CD44")),
)
DEVIATION_GRID = (0.1, 1.0, 10.0)
HYPERGRAPH_GRID = (0.0, 0.05, 0.2)
TEMPORAL_GRID = (0.0, 0.05, 0.2)
TRANSPORT_GRID = (0.0, 0.5, 0.75, 1.0, 1.25)
COEFFICIENT_RIDGE = 0.01
POISSON_RIDGE_GRID = (0.001, 0.01, 0.1, 1.0)
MANDATORY_METHODS = (
    "primary",
    "classical_time_conditioned_ridge_poisson",
    "fixed_margin_independence",
    "destroyed_link",
    "graph_zero_retuned_exact_coupling",
)
HELD_PAIRWISE_METHODS = (
    "classical_time_conditioned_ridge_poisson",
    "graph_zero_retuned_exact_coupling",
    "destroyed_link",
)
DESCRIPTIVE_METHODS = (
    "temporal_zero_retuned_exact_coupling",
    "structure_zero_exact_coupling",
    "poisson_interaction_conditional_reconstruction",
)


class EstimationRefusal(RuntimeError):
    """The frozen numerical certificate was not met."""


@dataclass(frozen=True, order=True)
class CouplingConfig:
    deviation_penalty: float
    hypergraph_penalty: float
    temporal_penalty: float
    transport_multiplier: float


@dataclass(frozen=True)
class CouplingFit:
    population_log_odds: np.ndarray
    donor_timepoint_deviation: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    configuration: CouplingConfig
    optimizer: str


@dataclass(frozen=True)
class _ConditionalRecord:
    observed: float
    support: np.ndarray
    null_log_probability: np.ndarray


def marker_incidence() -> np.ndarray:
    """Return the frozen 16-marker by five-edge unit incidence matrix."""

    index = {marker: position for position, marker in enumerate(MARKERS)}
    incidence = np.zeros((len(MARKERS), len(HYPEREDGES)), dtype=float)
    for edge, (_, members) in enumerate(HYPEREDGES):
        for marker in members:
            incidence[index[marker], edge] = 1.0
    if np.any(incidence.sum(axis=1) == 0.0):
        raise AssertionError("every frozen marker must belong to a hyperedge")
    return incidence


def marker_laplacian() -> np.ndarray:
    return normalized_hypergraph_laplacian(marker_incidence())


def pair_laplacian() -> np.ndarray:
    marker = marker_laplacian()
    identity = np.eye(len(MARKERS))
    return np.kron(marker, identity) + np.kron(identity, marker)


def temporal_laplacian() -> np.ndarray:
    adjacency = np.diag(np.ones(len(TIMEPOINTS) - 1), 1)
    adjacency += adjacency.T
    return np.diag(adjacency.sum(axis=1)) - adjacency


def _validate_tables(
    tables: np.ndarray, timepoints: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tables)
    expected_tail = (len(MARKERS), len(MARKERS), 2, 2)
    if values.ndim != 5 or values.shape[1:] != expected_tail:
        raise ValueError(f"tables must have shape visit x {expected_tail}")
    numeric = np.asarray(values, dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ValueError("tables must contain finite nonnegative integer counts")
    labels = tuple(timepoints)
    if len(labels) != len(values) or any(label not in TIMEPOINTS for label in labels):
        raise ValueError("timepoints must align to visits and use the frozen axis")
    indices = np.asarray([TIMEPOINTS.index(label) for label in labels], dtype=int)
    return numeric.astype(np.int64), indices


def _conditional_records(tables: np.ndarray) -> tuple[_ConditionalRecord | None, ...]:
    flat = tables.reshape(-1, 2, 2)
    records: list[_ConditionalRecord | None] = []
    for table in flat:
        counts, cells, _ = _fixed_margin_support(table)
        support = np.asarray(cells[:, 0], dtype=float)
        if len(support) < 2:
            records.append(None)
            continue
        row_zero = int(counts[0].sum())
        column_zero = int(counts[:, 0].sum())
        total = int(counts.sum())
        log_weights = _log_choose(column_zero, support) + _log_choose(
            total - column_zero, row_zero - support
        )
        null_log_probability = log_weights - logsumexp(log_weights)
        records.append(
            _ConditionalRecord(
                observed=float(counts[0, 0]),
                support=support,
                null_log_probability=np.asarray(null_log_probability, dtype=float),
            )
        )
    return tuple(records)


def _validate_config(configuration: CouplingConfig) -> CouplingConfig:
    if configuration.deviation_penalty not in DEVIATION_GRID:
        raise ValueError("deviation penalty is outside the frozen grid")
    if configuration.hypergraph_penalty not in HYPERGRAPH_GRID:
        raise ValueError("hypergraph penalty is outside the frozen grid")
    if configuration.temporal_penalty not in TEMPORAL_GRID:
        raise ValueError("temporal penalty is outside the frozen grid")
    if configuration.transport_multiplier not in TRANSPORT_GRID:
        raise ValueError("transport multiplier is outside the frozen grid")
    return configuration


def fit_coupling_field(
    tables: np.ndarray,
    timepoints: Sequence[str],
    configuration: CouplingConfig,
    *,
    maximum_iterations: int = 400,
    gradient_tolerance: float = 1e-6,
) -> CouplingFit:
    """Fit the frozen direct exact-conditional longitudinal objective."""

    values, time_index = _validate_tables(tables, timepoints)
    config = _validate_config(configuration)
    visits = len(values)
    entities = len(MARKERS) ** 2
    records = _conditional_records(values)
    visits_per_timepoint = np.bincount(time_index, minlength=len(TIMEPOINTS))
    visit_weight = 1.0 / visits_per_timepoint[time_index]
    marker_operator = pair_laplacian()
    time_operator = temporal_laplacian()
    beta_size = len(TIMEPOINTS) * entities
    total_size = beta_size + visits * entities

    def objective_and_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
        beta = vector[:beta_size].reshape(len(TIMEPOINTS), entities)
        deviation = vector[beta_size:].reshape(visits, entities)
        theta = beta[time_index] + deviation
        score = np.zeros_like(theta)
        objective = 0.0
        for flat_index, record in enumerate(records):
            if record is None:
                continue
            visit, entity = divmod(flat_index, entities)
            current = float(theta[visit, entity])
            centered = record.support - record.observed
            log_mass = record.null_log_probability + centered * current
            log_partition = float(logsumexp(log_mass))
            probability = np.exp(log_mass - log_partition)
            objective += visit_weight[visit] * log_partition
            score[visit, entity] = visit_weight[visit] * float(
                probability @ centered
            )

        objective += (
            0.5
            * config.deviation_penalty
            * float(np.sum(visit_weight[:, None] * np.square(deviation)))
        )
        objective += 0.5 * COEFFICIENT_RIDGE * float(np.sum(np.square(beta)))
        marker_action = np.einsum(
            "te,ef->tf", beta, marker_operator, optimize=False
        )
        temporal_action = np.einsum(
            "st,te->se", time_operator, beta, optimize=False
        )
        objective += (
            0.5
            * config.hypergraph_penalty
            * float(np.einsum("te,te->", beta, marker_action, optimize=False))
        )
        objective += (
            0.5
            * config.temporal_penalty
            * float(np.einsum("te,te->", beta, temporal_action, optimize=False))
        )

        beta_gradient = np.zeros_like(beta)
        np.add.at(beta_gradient, time_index, score)
        beta_gradient += COEFFICIENT_RIDGE * beta
        beta_gradient += config.hypergraph_penalty * marker_action
        beta_gradient += config.temporal_penalty * temporal_action
        deviation_gradient = score + (
            config.deviation_penalty * visit_weight[:, None] * deviation
        )
        gradient = np.concatenate((beta_gradient.ravel(), deviation_gradient.ravel()))
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise EstimationRefusal("exact conditional objective is not finite")
        return float(objective), gradient

    initial = np.zeros(total_size, dtype=float)
    result = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maximum_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1e-15,
            "maxls": 50,
        },
    )
    objective, gradient = objective_and_gradient(np.asarray(result.x, dtype=float))
    gradient_norm = float(np.max(np.abs(gradient)))
    if (
        not result.success
        or not np.isfinite(objective)
        or gradient_norm > 5.0 * float(gradient_tolerance)
    ):
        raise EstimationRefusal(
            "longitudinal coupling optimizer missed its convergence certificate: "
            f"success={result.success}, iterations={result.nit}, "
            f"gradient={gradient_norm:.6g}, message={result.message}"
        )
    beta = np.asarray(result.x[:beta_size], dtype=float).reshape(
        len(TIMEPOINTS), len(MARKERS), len(MARKERS)
    )
    deviation = np.asarray(result.x[beta_size:], dtype=float).reshape(
        visits, len(MARKERS), len(MARKERS)
    )
    return CouplingFit(
        population_log_odds=beta,
        donor_timepoint_deviation=deviation,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=int(result.nit),
        configuration=config,
        optimizer="direct_lbfgsb_exact_conditional_sparse_penalty",
    )


def fit_time_conditioned_ridge_poisson(
    tables: np.ndarray,
    timepoints: Sequence[str],
    *,
    ridge_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values, _ = _validate_tables(tables, timepoints)
    ridge = float(ridge_penalty)
    if ridge not in POISSON_RIDGE_GRID:
        raise ValueError("Poisson ridge penalty is outside the frozen grid")
    fit = fit_ridge_profiled_poisson_interaction(
        values,
        np.asarray(tuple(timepoints), dtype=object),
        ridge_penalty=ridge,
        score_tolerance=1e-10,
        certificate_tolerance=1e-8,
    )
    group_index = {str(label): index for index, label in enumerate(fit.group_labels)}
    if set(group_index) != set(TIMEPOINTS):
        raise EstimationRefusal("ridge Poisson fit lacks a frozen timepoint")
    field = np.asarray(fit.log_odds, dtype=float)[
        [group_index[timepoint] for timepoint in TIMEPOINTS]
    ]
    audit = {
        "ridge_penalty": float(fit.ridge_penalty),
        "maximum_scaled_penalized_score": float(fit.maximum_scaled_penalized_score),
        "maximum_absolute_row_margin_error": float(
            fit.maximum_absolute_row_margin_error
        ),
        "maximum_absolute_column_margin_error": float(
            fit.maximum_absolute_column_margin_error
        ),
        "maximum_absolute_log_odds_error": float(fit.maximum_absolute_log_odds_error),
        "fitted_group_order": [str(label) for label in fit.group_labels],
    }
    return field, audit


def joint_binary_tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    rna = np.asarray(rna_states)
    adt = np.asarray(adt_states)
    if (
        rna.ndim != 2
        or adt.ndim != 2
        or rna.shape != adt.shape
        or rna.shape[1] != len(MARKERS)
        or not np.isin(rna, (0, 1)).all()
        or not np.isin(adt, (0, 1)).all()
    ):
        raise ValueError("state matrices must be aligned binary cell by marker arrays")
    tables = np.empty((len(MARKERS), len(MARKERS), 2, 2), dtype=np.int64)
    for first in range(len(MARKERS)):
        for second in range(len(MARKERS)):
            tables[first, second] = np.bincount(
                2 * rna[:, first] + adt[:, second], minlength=4
            ).reshape(2, 2)
    return tables


def destroyed_link_tables(rna_states: np.ndarray, adt_states: np.ndarray) -> np.ndarray:
    adt = np.asarray(adt_states)
    if len(adt) < 2:
        raise ValueError("destroyed-link control requires at least two cells")
    return joint_binary_tables(rna_states, np.roll(adt, shift=1, axis=0))


def predict_tables_at_observed_margins(
    observed: np.ndarray, log_odds: np.ndarray
) -> np.ndarray:
    truth = np.asarray(observed)
    field = np.asarray(log_odds, dtype=float)
    if truth.shape != (len(MARKERS), len(MARKERS), 2, 2):
        raise ValueError("observed table panel differs from the frozen marker axes")
    if field.shape != truth.shape[:-2] or not np.isfinite(field).all():
        raise ValueError("log-odds field differs from the frozen marker axes")
    predicted = np.empty_like(truth, dtype=float)
    for entity in np.ndindex(field.shape):
        table = truth[entity]
        predicted[entity] = expected_binary_table_from_log_odds(
            float(field[entity]), table.sum(axis=1), table.sum(axis=0)
        )
    return predicted


def predict_poisson_tables_at_observed_margins(
    observed: np.ndarray,
    log_odds: np.ndarray,
    *,
    transport_scale: float = 1.0,
) -> np.ndarray:
    """Reconstruct the classical Poisson comparator at observed margins."""

    truth = np.asarray(observed)
    field = np.asarray(log_odds, dtype=float)
    if truth.shape != (len(MARKERS), len(MARKERS), 2, 2):
        raise ValueError("observed table panel differs from the frozen marker axes")
    if field.shape != truth.shape[:-2] or not np.isfinite(field).all():
        raise ValueError("log-odds field differs from the frozen marker axes")
    return reconstruct_poisson_tables(
        field,
        truth.sum(axis=-1),
        truth.sum(axis=-2),
        transport_scale=transport_scale,
    ).table


def deviance_per_observation(observed: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.shape[-2:] != (2, 2):
        raise ValueError("observed and predicted table panels must align")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(estimate).all()
        or np.any(truth < 0.0)
        or np.any(estimate < 0.0)
    ):
        raise ValueError("table panels must be finite and nonnegative")
    positive = truth > 0.0
    if np.any(estimate[positive] <= 0.0):
        raise ValueError("a positive observed cell has zero predicted mass")
    totals = truth.sum(axis=(-2, -1))
    if np.any(totals <= 0.0) or not np.allclose(
        totals, estimate.sum(axis=(-2, -1)), rtol=1e-8, atol=1e-8
    ):
        raise ValueError("predicted and observed margins or totals differ")
    terms = np.zeros_like(truth)
    terms[positive] = truth[positive] * np.log(truth[positive] / estimate[positive])
    deviance = 2.0 * terms.sum(axis=(-2, -1)) / totals
    return float(np.mean(deviance))


def losses_from_fields(
    observed: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
    fields: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float] | float]]:
    values, _ = _validate_tables(observed, timepoints)
    patients = tuple(str(value) for value in patient_ids)
    times = tuple(timepoints)
    if len(patients) != len(values):
        raise ValueError("patient ids must align to visits")
    result: dict[str, dict[str, dict[str, float] | float]] = {}
    for method, raw_field in fields.items():
        field = np.asarray(raw_field, dtype=float)
        if field.shape != (len(TIMEPOINTS), len(MARKERS), len(MARKERS)):
            raise ValueError(f"{method} field differs from the frozen axes")
        if method == "classical_time_conditioned_ridge_poisson":
            visit_losses = _visit_losses_for_poisson(values, times, field)
        else:
            visit_losses = _visit_losses_for_field(values, times, field)
        by_patient = {
            patient: float(np.mean(visit_losses[np.asarray(patients) == patient]))
            for patient in dict.fromkeys(patients)
        }
        by_timepoint = {
            timepoint: float(np.mean(visit_losses[np.asarray(times) == timepoint]))
            for timepoint in dict.fromkeys(times)
        }
        result[method] = {
            "mean": float(np.mean(tuple(by_patient.values()))),
            "by_patient": by_patient,
            "by_timepoint": by_timepoint,
        }
    return result


def _visit_losses_for_field(
    tables: np.ndarray, timepoints: Sequence[str], field: np.ndarray
) -> np.ndarray:
    values, _ = _validate_tables(tables, timepoints)
    log_odds = np.asarray(field, dtype=float)
    if log_odds.shape != (len(TIMEPOINTS), len(MARKERS), len(MARKERS)):
        raise ValueError("field differs from the frozen time and marker axes")
    return np.asarray(
        [
            deviance_per_observation(
                table,
                predict_tables_at_observed_margins(
                    table, log_odds[TIMEPOINTS.index(timepoint)]
                ),
            )
            for table, timepoint in zip(values, timepoints)
        ],
        dtype=float,
    )


def _visit_losses_for_poisson(
    tables: np.ndarray,
    timepoints: Sequence[str],
    field: np.ndarray,
    *,
    transport_scale: float = 1.0,
) -> np.ndarray:
    values, _ = _validate_tables(tables, timepoints)
    log_odds = np.asarray(field, dtype=float)
    if log_odds.shape != (len(TIMEPOINTS), len(MARKERS), len(MARKERS)):
        raise ValueError("Poisson field differs from the frozen time and marker axes")
    return np.asarray(
        [
            deviance_per_observation(
                table,
                predict_poisson_tables_at_observed_margins(
                    table,
                    log_odds[TIMEPOINTS.index(timepoint)],
                    transport_scale=transport_scale,
                ),
            )
            for table, timepoint in zip(values, timepoints)
        ],
        dtype=float,
    )


def _aggregate_visit_losses(
    visit_losses: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    values = np.asarray(visit_losses, dtype=float)
    patients = np.asarray(tuple(map(str, patient_ids)), dtype=object)
    times = np.asarray(tuple(timepoints), dtype=object)
    if values.shape != patients.shape or values.shape != times.shape:
        raise ValueError("visit losses and axes must align")
    by_patient = {
        patient: float(np.mean(values[patients == patient]))
        for patient in dict.fromkeys(map(str, patient_ids))
    }
    by_timepoint = {
        timepoint: float(np.mean(values[times == timepoint]))
        for timepoint in TIMEPOINTS
        if np.any(times == timepoint)
    }
    return by_patient, by_timepoint


def _complete_source_axes(
    patient_ids: Sequence[str], timepoints: Sequence[str], expected_patients: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    patients = np.asarray(tuple(map(str, patient_ids)), dtype=object)
    times = tuple(timepoints)
    order = tuple(dict.fromkeys(map(str, patient_ids)))
    if len(order) != expected_patients or len(patients) != expected_patients * 4:
        raise ValueError("source axis has the wrong number of complete patients")
    for patient in order:
        indices = np.flatnonzero(patients == patient)
        if tuple(times[index] for index in indices) != TIMEPOINTS:
            raise ValueError("each source patient must contribute T01-T04 in order")
    return patients, order


def _base_configurations(*, graph_zero: bool = False) -> tuple[CouplingConfig, ...]:
    hypergraph_values = (0.0,) if graph_zero else HYPERGRAPH_GRID
    return tuple(
        CouplingConfig(eta, graph, temporal, 1.0)
        for eta in DEVIATION_GRID
        for graph in hypergraph_values
        for temporal in TEMPORAL_GRID
    )


def _with_transport(base: CouplingConfig, transport: float) -> CouplingConfig:
    return CouplingConfig(
        base.deviation_penalty,
        base.hypergraph_penalty,
        base.temporal_penalty,
        transport,
    )


def _configuration_key(configuration: CouplingConfig) -> tuple[float, ...]:
    return (
        configuration.deviation_penalty,
        configuration.hypergraph_penalty,
        configuration.temporal_penalty,
        configuration.transport_multiplier,
    )


def _selection_record(
    by_configuration: Mapping[CouplingConfig, Mapping[str, Any]],
) -> tuple[CouplingConfig, Mapping[str, Any]]:
    def patient_mean(configuration: CouplingConfig) -> float:
        record = by_configuration[configuration]
        by_patient = record.get("by_patient")
        if isinstance(by_patient, Mapping):
            return float(np.mean(tuple(by_patient.values())))
        return float(record["mean"])

    selected = min(
        by_configuration,
        key=lambda config: (
            patient_mean(config),
            _configuration_key(config),
        ),
    )
    return selected, by_configuration[selected]


def _cv_exact_candidates(
    tables: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
) -> dict[CouplingConfig, dict[str, Any]]:
    values, _ = _validate_tables(tables, timepoints)
    patients, patient_order = _complete_source_axes(patient_ids, timepoints, 7)
    candidate_patient = {
        _with_transport(base, alpha): {}
        for base in _base_configurations()
        for alpha in TRANSPORT_GRID
    }
    candidate_time: dict[CouplingConfig, dict[str, list[float]]] = {
        config: {timepoint: [] for timepoint in TIMEPOINTS}
        for config in candidate_patient
    }
    for patient in patient_order:
        held = patients == patient
        train = ~held
        for base in _base_configurations():
            fitted = fit_coupling_field(
                values[train], np.asarray(timepoints, dtype=object)[train], base
            )
            for alpha in TRANSPORT_GRID:
                config = _with_transport(base, alpha)
                visit_losses = _visit_losses_for_field(
                    values[held],
                    np.asarray(timepoints, dtype=object)[held],
                    alpha * fitted.population_log_odds,
                )
                candidate_patient[config][patient] = float(np.mean(visit_losses))
                held_times = np.asarray(timepoints, dtype=object)[held]
                for timepoint, loss in zip(held_times, visit_losses):
                    candidate_time[config][str(timepoint)].append(float(loss))
    return {
        config: {
            "mean": float(np.mean(tuple(by_patient.values()))),
            "by_patient": dict(by_patient),
            "by_timepoint": {
                timepoint: float(np.mean(values))
                for timepoint, values in candidate_time[config].items()
            },
        }
        for config, by_patient in candidate_patient.items()
    }


def _cv_poisson_candidates(
    tables: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
) -> tuple[
    dict[tuple[float, float], dict[str, Any]],
    dict[tuple[float, float], dict[str, Any]],
]:
    values, _ = _validate_tables(tables, timepoints)
    patients, patient_order = _complete_source_axes(patient_ids, timepoints, 7)
    candidates = tuple(
        (ridge, alpha)
        for ridge in POISSON_RIDGE_GRID
        for alpha in TRANSPORT_GRID
    )
    classical_by_patient: dict[tuple[float, float], dict[str, float]] = {
        candidate: {} for candidate in candidates
    }
    conditional_by_patient: dict[tuple[float, float], dict[str, float]] = {
        candidate: {} for candidate in candidates
    }
    classical_by_time = {
        candidate: {timepoint: [] for timepoint in TIMEPOINTS}
        for candidate in candidates
    }
    conditional_by_time = {
        candidate: {timepoint: [] for timepoint in TIMEPOINTS}
        for candidate in candidates
    }
    labels = np.asarray(timepoints, dtype=object)
    for patient in patient_order:
        held = patients == patient
        train = ~held
        for ridge in POISSON_RIDGE_GRID:
            field, _ = fit_time_conditioned_ridge_poisson(
                values[train], labels[train], ridge_penalty=ridge
            )
            for alpha in TRANSPORT_GRID:
                candidate = (ridge, alpha)
                classical = _visit_losses_for_poisson(
                    values[held], labels[held], field, transport_scale=alpha
                )
                conditional = _visit_losses_for_field(
                    values[held], labels[held], alpha * field
                )
                classical_by_patient[candidate][patient] = float(
                    np.mean(classical)
                )
                conditional_by_patient[candidate][patient] = float(
                    np.mean(conditional)
                )
                for timepoint, classical_loss, conditional_loss in zip(
                    labels[held], classical, conditional
                ):
                    classical_by_time[candidate][str(timepoint)].append(
                        float(classical_loss)
                    )
                    conditional_by_time[candidate][str(timepoint)].append(
                        float(conditional_loss)
                    )

    def aggregate(
        by_patient: Mapping[tuple[float, float], Mapping[str, float]],
        by_time: Mapping[tuple[float, float], Mapping[str, Sequence[float]]],
    ) -> dict[tuple[float, float], dict[str, Any]]:
        return {
            candidate: {
                "mean": float(np.mean(tuple(patient_losses.values()))),
                "by_patient": dict(patient_losses),
                "by_timepoint": {
                    timepoint: float(np.mean(losses))
                    for timepoint, losses in by_time[candidate].items()
                },
            }
            for candidate, patient_losses in by_patient.items()
        }

    return (
        aggregate(classical_by_patient, classical_by_time),
        aggregate(conditional_by_patient, conditional_by_time),
    )


def _cv_destroyed(
    intact: np.ndarray,
    destroyed: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
    configuration: CouplingConfig,
) -> dict[str, Any]:
    truth, _ = _validate_tables(intact, timepoints)
    values, _ = _validate_tables(destroyed, timepoints)
    if truth.shape != values.shape:
        raise ValueError("intact and destroyed panels must align")
    patients, patient_order = _complete_source_axes(patient_ids, timepoints, 7)
    labels = np.asarray(timepoints, dtype=object)
    by_patient: dict[str, float] = {}
    by_time = {timepoint: [] for timepoint in TIMEPOINTS}
    base = _with_transport(configuration, 1.0)
    for patient in patient_order:
        held = patients == patient
        train = ~held
        fitted = fit_coupling_field(values[train], labels[train], base)
        visit_losses = _visit_losses_for_field(
            truth[held],
            labels[held],
            configuration.transport_multiplier * fitted.population_log_odds,
        )
        by_patient[patient] = float(np.mean(visit_losses))
        for timepoint, loss in zip(labels[held], visit_losses):
            by_time[str(timepoint)].append(float(loss))
    return {
        "mean": float(np.mean(tuple(by_patient.values()))),
        "by_patient": by_patient,
        "by_timepoint": {
            timepoint: float(np.mean(losses)) for timepoint, losses in by_time.items()
        },
    }


def select_calibration_models(
    tables: np.ndarray,
    destroyed: np.ndarray,
    patient_ids: Sequence[str],
    timepoints: Sequence[str],
) -> dict[str, Any]:
    """Run the frozen seven-patient LOPO selection without pilot access."""

    values, _ = _validate_tables(tables, timepoints)
    destroyed_values, _ = _validate_tables(destroyed, timepoints)
    if not np.array_equal(
        values.sum(axis=(-2, -1)), destroyed_values.sum(axis=(-2, -1))
    ):
        raise ValueError("destroyed controls must preserve every table total")
    _, patient_order = _complete_source_axes(patient_ids, timepoints, 7)
    exact = _cv_exact_candidates(values, patient_ids, timepoints)
    selected_primary, primary_losses = _selection_record(exact)
    zero_candidates = {
        config: record
        for config, record in exact.items()
        if config.hypergraph_penalty == 0.0
    }
    selected_graph_zero, graph_zero_losses = _selection_record(zero_candidates)
    temporal_zero_candidates = {
        config: record
        for config, record in exact.items()
        if config.temporal_penalty == 0.0
    }
    selected_temporal_zero, temporal_zero_losses = _selection_record(
        temporal_zero_candidates
    )
    structure_zero_candidates = {
        config: record
        for config, record in exact.items()
        if config.hypergraph_penalty == 0.0 and config.temporal_penalty == 0.0
    }
    selected_structure_zero, structure_zero_losses = _selection_record(
        structure_zero_candidates
    )
    poisson_candidates, poisson_conditional_candidates = _cv_poisson_candidates(
        values, patient_ids, timepoints
    )
    selected_poisson = min(
        poisson_candidates,
        key=lambda candidate: (
            poisson_candidates[candidate]["mean"],
            candidate,
        ),
    )
    selected_poisson_ridge, selected_poisson_alpha = selected_poisson
    destroyed_losses = _cv_destroyed(
        values, destroyed_values, patient_ids, timepoints, selected_primary
    )
    independence_visit = _visit_losses_for_field(
        values,
        timepoints,
        np.zeros((len(TIMEPOINTS), len(MARKERS), len(MARKERS))),
    )
    independence_patient, independence_time = _aggregate_visit_losses(
        independence_visit, patient_ids, timepoints
    )
    losses = {
        "primary": primary_losses,
        "classical_time_conditioned_ridge_poisson": poisson_candidates[
            selected_poisson
        ],
        "fixed_margin_independence": {
            "mean": float(np.mean(tuple(independence_patient.values()))),
            "by_patient": independence_patient,
            "by_timepoint": independence_time,
        },
        "destroyed_link": destroyed_losses,
        "graph_zero_retuned_exact_coupling": graph_zero_losses,
    }
    descriptive_losses = {
        "temporal_zero_retuned_exact_coupling": temporal_zero_losses,
        "structure_zero_exact_coupling": structure_zero_losses,
        "poisson_interaction_conditional_reconstruction": (
            poisson_conditional_candidates[selected_poisson]
        ),
    }
    gate = calibration_gate(losses, patient_order)
    return {
        "patient_order": list(patient_order),
        "selected_primary": asdict(selected_primary),
        "selected_graph_zero": asdict(selected_graph_zero),
        "selected_temporal_zero": asdict(selected_temporal_zero),
        "selected_structure_zero": asdict(selected_structure_zero),
        "selected_poisson_ridge": float(selected_poisson_ridge),
        "selected_poisson_transport": float(selected_poisson_alpha),
        "losses": losses,
        "descriptive_losses": descriptive_losses,
        "gate": gate,
        "configuration_count": len(exact),
        "graph_zero_configuration_count": len(zero_candidates),
        "temporal_zero_configuration_count": len(temporal_zero_candidates),
        "structure_zero_configuration_count": len(structure_zero_candidates),
        "exact_cv_ledger": [
            {"configuration": asdict(config), "losses": exact[config]}
            for config in sorted(exact)
        ],
        "poisson_cv_ledger": [
            {
                "ridge_penalty": ridge,
                "transport_multiplier": alpha,
                "losses": poisson_candidates[(ridge, alpha)],
                "conditional_reconstruction_losses": (
                    poisson_conditional_candidates[(ridge, alpha)]
                ),
            }
            for ridge, alpha in sorted(poisson_candidates)
        ],
    }


def replay_calibration_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    exact_ledger = selection.get("exact_cv_ledger")
    poisson_ledger = selection.get("poisson_cv_ledger")
    if not isinstance(exact_ledger, list) or len(exact_ledger) != 135:
        raise ValueError("exact CV ledger must contain all 135 configurations")
    expected_poisson_count = len(POISSON_RIDGE_GRID) * len(TRANSPORT_GRID)
    if (
        not isinstance(poisson_ledger, list)
        or len(poisson_ledger) != expected_poisson_count
    ):
        raise ValueError("Poisson CV ledger must contain the full frozen grid")
    exact: dict[CouplingConfig, Mapping[str, Any]] = {}
    for row in exact_ledger:
        config = _config_from_mapping(row["configuration"])
        if config in exact:
            raise ValueError("exact CV ledger duplicates a configuration")
        exact[config] = row["losses"]
    selected_primary, _ = _selection_record(exact)
    selected_graph_zero, _ = _selection_record(
        {
            config: losses
            for config, losses in exact.items()
            if config.hypergraph_penalty == 0.0
        }
    )
    selected_temporal_zero, _ = _selection_record(
        {
            config: losses
            for config, losses in exact.items()
            if config.temporal_penalty == 0.0
        }
    )
    selected_structure_zero, _ = _selection_record(
        {
            config: losses
            for config, losses in exact.items()
            if config.hypergraph_penalty == 0.0
            and config.temporal_penalty == 0.0
        }
    )
    poisson: dict[tuple[float, float], Mapping[str, Any]] = {}
    for row in poisson_ledger:
        candidate = (
            float(row["ridge_penalty"]),
            float(row["transport_multiplier"]),
        )
        if candidate in poisson:
            raise ValueError("Poisson CV ledger duplicates a configuration")
        poisson[candidate] = row["losses"]
    expected_poisson = {
        (ridge, alpha)
        for ridge in POISSON_RIDGE_GRID
        for alpha in TRANSPORT_GRID
    }
    if set(poisson) != expected_poisson:
        raise ValueError("Poisson CV ledger differs from the frozen grid")
    selected_poisson = min(
        poisson,
        key=lambda candidate: (float(poisson[candidate]["mean"]), candidate),
    )
    replay = {
        "selected_primary": asdict(selected_primary),
        "selected_graph_zero": asdict(selected_graph_zero),
        "selected_temporal_zero": asdict(selected_temporal_zero),
        "selected_structure_zero": asdict(selected_structure_zero),
        "selected_poisson_ridge": float(selected_poisson[0]),
        "selected_poisson_transport": float(selected_poisson[1]),
    }
    if any(selection.get(key) != value for key, value in replay.items()):
        raise ValueError("serialized calibration selection differs from the CV ledgers")
    return replay


def _config_from_mapping(value: Mapping[str, Any]) -> CouplingConfig:
    return _validate_config(
        CouplingConfig(
            float(value["deviation_penalty"]),
            float(value["hypergraph_penalty"]),
            float(value["temporal_penalty"]),
            float(value["transport_multiplier"]),
        )
    )


def fit_frozen_models(
    tables: np.ndarray,
    destroyed: np.ndarray,
    timepoints: Sequence[str],
    selected_primary: Mapping[str, Any],
    selected_graph_zero: Mapping[str, Any],
    selected_temporal_zero: Mapping[str, Any],
    selected_structure_zero: Mapping[str, Any],
    poisson_ridge: float,
    poisson_transport: float,
) -> dict[str, Any]:
    primary_config = _config_from_mapping(selected_primary)
    graph_zero_config = _config_from_mapping(selected_graph_zero)
    temporal_zero_config = _config_from_mapping(selected_temporal_zero)
    structure_zero_config = _config_from_mapping(selected_structure_zero)
    if graph_zero_config.hypergraph_penalty != 0.0:
        raise ValueError("graph-zero configuration has a nonzero marker penalty")
    if temporal_zero_config.temporal_penalty != 0.0:
        raise ValueError("temporal-zero configuration has a nonzero temporal penalty")
    if (
        structure_zero_config.hypergraph_penalty != 0.0
        or structure_zero_config.temporal_penalty != 0.0
    ):
        raise ValueError("structure-zero configuration has a nonzero structure penalty")
    if float(poisson_ridge) not in POISSON_RIDGE_GRID:
        raise ValueError("Poisson ridge penalty is outside the frozen grid")
    if float(poisson_transport) not in TRANSPORT_GRID:
        raise ValueError("Poisson transport is outside the frozen grid")
    primary = fit_coupling_field(
        tables, timepoints, _with_transport(primary_config, 1.0)
    )
    graph_zero = fit_coupling_field(
        tables, timepoints, _with_transport(graph_zero_config, 1.0)
    )
    temporal_zero = fit_coupling_field(
        tables, timepoints, _with_transport(temporal_zero_config, 1.0)
    )
    structure_zero = fit_coupling_field(
        tables, timepoints, _with_transport(structure_zero_config, 1.0)
    )
    destroyed_fit = fit_coupling_field(
        destroyed, timepoints, _with_transport(primary_config, 1.0)
    )
    poisson, poisson_audit = fit_time_conditioned_ridge_poisson(
        tables, timepoints, ridge_penalty=float(poisson_ridge)
    )
    transported_poisson = float(poisson_transport) * poisson
    fields = {
        "primary": primary_config.transport_multiplier * primary.population_log_odds,
        "classical_time_conditioned_ridge_poisson": transported_poisson,
        "fixed_margin_independence": np.zeros_like(primary.population_log_odds),
        "destroyed_link": primary_config.transport_multiplier
        * destroyed_fit.population_log_odds,
        "graph_zero_retuned_exact_coupling": graph_zero_config.transport_multiplier
        * graph_zero.population_log_odds,
    }
    descriptive_fields = {
        "temporal_zero_retuned_exact_coupling": (
            temporal_zero_config.transport_multiplier
            * temporal_zero.population_log_odds
        ),
        "structure_zero_exact_coupling": (
            structure_zero_config.transport_multiplier
            * structure_zero.population_log_odds
        ),
        "poisson_interaction_conditional_reconstruction": transported_poisson,
    }
    return {
        "fields": {method: field.tolist() for method, field in fields.items()},
        "field_sha256": {
            method: _array_sha256(field) for method, field in fields.items()
        },
        "descriptive_fields": {
            method: field.tolist() for method, field in descriptive_fields.items()
        },
        "descriptive_field_sha256": {
            method: _array_sha256(field)
            for method, field in descriptive_fields.items()
        },
        "primary_fit": serialize_fit(primary),
        "graph_zero_fit": serialize_fit(graph_zero),
        "temporal_zero_fit": serialize_fit(temporal_zero),
        "structure_zero_fit": serialize_fit(structure_zero),
        "destroyed_fit": serialize_fit(destroyed_fit),
        "poisson_audit": poisson_audit,
        "selected_primary": asdict(primary_config),
        "selected_graph_zero": asdict(graph_zero_config),
        "selected_temporal_zero": asdict(temporal_zero_config),
        "selected_structure_zero": asdict(structure_zero_config),
        "selected_poisson_ridge": float(poisson_ridge),
        "selected_poisson_transport": float(poisson_transport),
    }


def _loss_vectors(
    losses: Mapping[str, Mapping[str, Any]], patient_order: Sequence[str]
) -> dict[str, np.ndarray]:
    patients = tuple(str(value) for value in patient_order)
    if set(losses) != set(MANDATORY_METHODS):
        raise ValueError("loss payload must contain exactly the mandatory methods")
    vectors = {}
    for method in MANDATORY_METHODS:
        by_patient = losses[method].get("by_patient")
        if not isinstance(by_patient, Mapping) or set(by_patient) != set(patients):
            raise ValueError(f"{method} patient losses differ from the frozen panel")
        vector = np.asarray([by_patient[patient] for patient in patients], dtype=float)
        if not np.isfinite(vector).all() or np.any(vector < 0.0):
            raise ValueError("loss values must be finite and nonnegative")
        vectors[method] = vector
    return vectors


def _timepoint_favorable_count(
    losses: Mapping[str, Mapping[str, Any]], comparator: str
) -> int:
    primary = losses["primary"].get("by_timepoint")
    baseline = losses[comparator].get("by_timepoint")
    if not isinstance(primary, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("timepoint losses are required")
    if set(primary) != set(TIMEPOINTS) or set(baseline) != set(TIMEPOINTS):
        raise ValueError("timepoint losses must cover T01-T04")
    return sum(float(primary[t]) < float(baseline[t]) for t in TIMEPOINTS)


def calibration_gate(
    losses: Mapping[str, Mapping[str, Any]], patient_order: Sequence[str]
) -> dict[str, Any]:
    if len(tuple(patient_order)) != 7:
        raise ValueError("calibration gate requires seven physical patients")
    vectors = _loss_vectors(losses, patient_order)
    primary = vectors["primary"]
    comparisons = {}
    for method in MANDATORY_METHODS[1:]:
        difference = primary - vectors[method]
        reduction = 1.0 - float(primary.mean() / vectors[method].mean())
        checks = {
            "primary_mean_below": float(difference.mean()) < 0.0,
            "at_least_five_of_seven_favorable": int(np.count_nonzero(difference < 0.0))
            >= 5,
        }
        if method in {"classical_time_conditioned_ridge_poisson", "destroyed_link"}:
            checks["relative_reduction_at_least_five_percent"] = reduction >= 0.05
        comparisons[method] = {
            "mean_difference": float(difference.mean()),
            "relative_reduction": reduction,
            "favorable_patients": int(np.count_nonzero(difference < 0.0)),
            "checks": checks,
            "passes": all(checks.values()),
        }
    time_count = _timepoint_favorable_count(
        losses, "classical_time_conditioned_ridge_poisson"
    )
    return {
        "comparisons": comparisons,
        "favorable_timepoints_vs_poisson": time_count,
        "minimum_three_timepoints_passes": time_count >= 3,
        "passes": all(item["passes"] for item in comparisons.values())
        and time_count >= 3,
    }


def pilot_gate(
    losses: Mapping[str, Mapping[str, Any]], patient_order: Sequence[str]
) -> dict[str, Any]:
    if len(tuple(patient_order)) != 3:
        raise ValueError("pilot gate requires three physical patients")
    vectors = _loss_vectors(losses, patient_order)
    primary = vectors["primary"]
    comparisons = {}
    for method in MANDATORY_METHODS[1:]:
        difference = primary - vectors[method]
        reduction = 1.0 - float(primary.mean() / vectors[method].mean())
        checks = {"primary_mean_below": float(difference.mean()) < 0.0}
        if method in {"classical_time_conditioned_ridge_poisson", "destroyed_link"}:
            checks["all_three_patients_favorable"] = bool(np.all(difference < 0.0))
            checks["relative_reduction_at_least_five_percent"] = reduction >= 0.05
        comparisons[method] = {
            "mean_difference": float(difference.mean()),
            "relative_reduction": reduction,
            "favorable_patients": int(np.count_nonzero(difference < 0.0)),
            "checks": checks,
            "passes": all(checks.values()),
        }
    time_count = _timepoint_favorable_count(
        losses, "classical_time_conditioned_ridge_poisson"
    )
    return {
        "comparisons": comparisons,
        "favorable_timepoints_vs_poisson": time_count,
        "minimum_three_timepoints_passes": time_count >= 3,
        "passes": all(item["passes"] for item in comparisons.values())
        and time_count >= 3,
    }


def _fixed_panel_sign_p(favorable: int, total: int) -> float:
    return float(
        sum(math.comb(total, successes) for successes in range(favorable, total + 1))
        / (2**total)
    )


def held_gate(
    losses: Mapping[str, Mapping[str, Any]],
    patient_order: Sequence[str],
    completeness: Mapping[str, str],
) -> dict[str, Any]:
    patients = tuple(str(value) for value in patient_order)
    if len(patients) != 8:
        raise ValueError("held gate requires eight physical patients")
    if {completeness.get(patient) for patient in patients} != {"complete", "partial"}:
        raise ValueError("held patients must carry complete or partial strata")
    vectors = _loss_vectors(losses, patients)
    primary = vectors["primary"]
    strata = {
        label: np.asarray(
            [
                index
                for index, patient in enumerate(patients)
                if completeness[patient] == label
            ],
            dtype=int,
        )
        for label in ("complete", "partial")
    }
    if tuple(map(len, strata.values())) != (5, 3):
        raise ValueError(
            "held completeness strata must contain five and three patients"
        )
    complete_draws = strata["complete"][
        np.indices((5,) * 5, dtype=int).reshape(5, -1).T
    ]
    partial_draws = strata["partial"][
        np.indices((3,) * 3, dtype=int).reshape(3, -1).T
    ]
    resamples = np.concatenate(
        (
            np.repeat(complete_draws, len(partial_draws), axis=0),
            np.tile(partial_draws, (len(complete_draws), 1)),
        ),
        axis=1,
    )
    expected_resamples = 5**5 * 3**3
    if resamples.shape != (expected_resamples, len(patients)):
        raise AssertionError("exhaustive stratified resample construction failed")
    comparisons = {}
    for method in HELD_PAIRWISE_METHODS:
        comparator = vectors[method]
        difference = primary - comparator
        resampled_primary = np.mean(primary[resamples], axis=1)
        resampled_comparator = np.mean(comparator[resamples], axis=1)
        if np.any(resampled_comparator <= 0.0):
            raise ValueError("relative reduction is undefined for zero comparator loss")
        resampled_difference = resampled_primary - resampled_comparator
        resampled_reduction = 1.0 - resampled_primary / resampled_comparator
        difference_interval = np.quantile(
            resampled_difference, (0.025, 0.975), method="linear"
        )
        reduction_interval = np.quantile(
            resampled_reduction, (0.025, 0.975), method="linear"
        )
        favorable = int(np.count_nonzero(difference < 0.0))
        reduction = 1.0 - float(primary.mean() / comparator.mean())
        sign_p = _fixed_panel_sign_p(favorable, len(patients))
        checks = {
            "primary_mean_below": float(difference.mean()) < 0.0,
            "additive_difference_upper_95_below_zero": (
                float(difference_interval[1]) < 0.0
            ),
            "at_least_seven_of_eight_favorable": favorable >= 7,
            "one_sided_fixed_panel_sign_p_at_most_0_05": sign_p <= 0.05,
            "complete_mean_favorable": float(difference[strata["complete"]].mean())
            < 0.0,
            "partial_mean_favorable": float(difference[strata["partial"]].mean()) < 0.0,
        }
        if method in {"classical_time_conditioned_ridge_poisson", "destroyed_link"}:
            checks["relative_reduction_at_least_five_percent"] = reduction >= 0.05
        comparisons[method] = {
            "mean_difference": float(difference.mean()),
            "relative_reduction": reduction,
            "additive_difference_95_interval": list(
                map(float, difference_interval)
            ),
            "relative_reduction_95_interval": list(
                map(float, reduction_interval)
            ),
            "favorable_patients": favorable,
            "one_sided_fixed_panel_sign_p": sign_p,
            "checks": checks,
            "passes": all(checks.values()),
        }
    time_count = _timepoint_favorable_count(
        losses, "classical_time_conditioned_ridge_poisson"
    )
    independence_pass = float(primary.mean()) < float(
        vectors["fixed_margin_independence"].mean()
    )
    return {
        "exact_stratified_resample_count": expected_resamples,
        "comparisons": comparisons,
        "favorable_timepoints_vs_poisson": time_count,
        "minimum_three_timepoints_passes": time_count >= 3,
        "primary_mean_below_independence": independence_pass,
        "passes": (
            all(item["passes"] for item in comparisons.values())
            and time_count >= 3
            and independence_pass
        ),
    }


def serialize_fit(fit: CouplingFit) -> dict[str, Any]:
    return {
        "population_log_odds": fit.population_log_odds.tolist(),
        "population_log_odds_sha256": _array_sha256(fit.population_log_odds),
        "objective": fit.objective,
        "gradient_norm": fit.gradient_norm,
        "iterations": fit.iterations,
        "configuration": asdict(fit.configuration),
        "optimizer": fit.optimizer,
    }


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()
