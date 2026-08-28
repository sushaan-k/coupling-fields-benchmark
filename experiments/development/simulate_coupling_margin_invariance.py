"""Deterministic population comparison under separable margin shifts.

The source and every target table share one log-linear interaction field. Only
the row and column margins change. Each method receives the exact target
margins, so the comparison isolates whether its transferred interaction
representation is invariant to those changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mapreg.classical_residuals import poisson_independence_residuals
from mapreg.coupling_fields import association_coordinates, association_field
from mapreg.table_prediction import (
    field_coordinates_to_table,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/coupling_margin_invariance_simulation.json"
TOTAL = 100_000.0
INTERACTION_COORDINATES = 3.0 * np.array(
    [[0.22, -0.10], [0.08, 0.18]], dtype=float
)
SOURCE_ROW_PROPORTIONS = np.array([0.34, 0.38, 0.28], dtype=float)
SOURCE_COLUMN_PROPORTIONS = np.array([0.31, 0.36, 0.33], dtype=float)
TARGET_ROW_PROPORTIONS = np.array([0.66, 0.22, 0.12], dtype=float)
TARGET_COLUMN_PROPORTIONS = np.array([0.14, 0.24, 0.62], dtype=float)


def _residual_transfer(
    source_table: np.ndarray,
    row_margin: np.ndarray,
    column_margin: np.ndarray,
    *,
    residual: str,
) -> np.ndarray:
    """Apply the complete source residual matrix at new independence means."""

    source_residual = poisson_independence_residuals(
        source_table, residual=residual
    )
    return residual_coordinates_to_table(
        source_residual,
        row_margin,
        column_margin,
        residual=residual,
    )


def _nonnegative_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    value = max(0.0, multinomial_deviance_per_observation(observed, predicted))
    return 0.0 if value < 1e-14 else value


def _numerical_zero(value: float) -> float:
    """Canonicalize solver-level noise in exact-recovery diagnostics."""

    return 0.0 if abs(value) < 1e-10 else value


def binary_fixed_odds_ratio_example() -> dict[str, object]:
    """Return an explicit fixed-odds-ratio Pearson non-invariance witness."""

    odds_ratio = 9.0

    def solve(row_one: float, column_one: float) -> dict[str, float]:
        lower = max(0.0, row_one + column_one - 1.0)
        upper = min(row_one, column_one)
        for _ in range(120):
            joint_one = 0.5 * (lower + upper)
            numerator = joint_one * (1.0 - row_one - column_one + joint_one)
            denominator = (row_one - joint_one) * (column_one - joint_one)
            if numerator / denominator < odds_ratio:
                lower = joint_one
            else:
                upper = joint_one
        joint_one = 0.5 * (lower + upper)
        delta = joint_one - row_one * column_one
        phi = delta / np.sqrt(
            row_one * (1.0 - row_one) * column_one * (1.0 - column_one)
        )
        return {
            "row_one": row_one,
            "column_one": column_one,
            "joint_one_one": joint_one,
            "pearson_phi": phi,
            "coupling_coordinate": 0.5 * np.log(odds_ratio),
        }

    return {
        "odds_ratio": odds_ratio,
        "balanced_margins": solve(0.5, 0.5),
        "shifted_margins": solve(0.8, 0.3),
    }


def run_simulation() -> dict[str, object]:
    source_rows = TOTAL * SOURCE_ROW_PROPORTIONS
    source_columns = TOTAL * SOURCE_COLUMN_PROPORTIONS
    source_table = field_coordinates_to_table(
        INTERACTION_COORDINATES, source_rows, source_columns
    )
    transferred_field = association_coordinates(association_field(source_table))

    records = []
    for shift in np.linspace(0.0, 1.0, 6):
        row_proportions = (
            (1.0 - shift) * SOURCE_ROW_PROPORTIONS
            + shift * TARGET_ROW_PROPORTIONS
        )
        column_proportions = (
            (1.0 - shift) * SOURCE_COLUMN_PROPORTIONS
            + shift * TARGET_COLUMN_PROPORTIONS
        )
        rows = TOTAL * row_proportions
        columns = TOTAL * column_proportions
        truth = field_coordinates_to_table(INTERACTION_COORDINATES, rows, columns)
        field_prediction = field_coordinates_to_table(
            transferred_field, rows, columns
        )
        pearson_prediction = _residual_transfer(
            source_table, rows, columns, residual="pearson"
        )
        deviance_prediction = _residual_transfer(
            source_table, rows, columns, residual="deviance"
        )
        records.append(
            {
                "shift_fraction": float(shift),
                "row_proportions": row_proportions.tolist(),
                "column_proportions": column_proportions.tolist(),
                "field_maximum_absolute_count_error": _numerical_zero(
                    float(np.max(np.abs(field_prediction - truth)))
                ),
                "field_multinomial_deviance_per_observation": (
                    _nonnegative_deviance(truth, field_prediction)
                ),
                "pearson_residual_multinomial_deviance_per_observation": (
                    _nonnegative_deviance(truth, pearson_prediction)
                ),
                "deviance_residual_multinomial_deviance_per_observation": (
                    _nonnegative_deviance(truth, deviance_prediction)
                ),
            }
        )

    return {
        "schema_version": "coupling-margin-invariance-simulation-v1",
        "estimand": (
            "positive 3x3 population tables with a fixed log-linear interaction "
            "and changing row/column margins"
        ),
        "comparison": (
            "all methods receive exact target margins; residual comparators transfer "
            "the complete source Pearson or signed-deviance residual matrix"
        ),
        "total_count": TOTAL,
        "interaction_coordinates": INTERACTION_COORDINATES.tolist(),
        "source_row_proportions": SOURCE_ROW_PROPORTIONS.tolist(),
        "source_column_proportions": SOURCE_COLUMN_PROPORTIONS.tolist(),
        "terminal_row_proportions": TARGET_ROW_PROPORTIONS.tolist(),
        "terminal_column_proportions": TARGET_COLUMN_PROPORTIONS.tolist(),
        "binary_fixed_odds_ratio_witness": binary_fixed_odds_ratio_example(),
        "margin_shift_sweep": records,
        "interpretation_boundary": (
            "The field is the interaction term of a saturated Poisson log-linear "
            "model under centered constraints. It is not distinct from transferring "
            "that correctly parameterized classical interaction; the demonstrated "
            "advantage is only over transferring independence-model residuals."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run_simulation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
