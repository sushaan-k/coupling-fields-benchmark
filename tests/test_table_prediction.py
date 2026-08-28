import numpy as np
import pytest

from mapreg.classical_residuals import poisson_independence_residuals
from mapreg.coupling_fields import (
    association_coordinates,
    association_field,
)
from mapreg.table_prediction import (
    TablePredictionRefusal,
    field_coordinates_to_table,
    ipf_to_margins,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


def test_ipf_matches_both_margins():
    result = ipf_to_margins(
        np.array([[1.0, 4.0, 2.0], [5.0, 2.0, 3.0], [2.0, 1.0, 6.0]]),
        np.array([20.0, 30.0, 50.0]),
        np.array([25.0, 35.0, 40.0]),
    )
    np.testing.assert_allclose(result.sum(axis=1), [20.0, 30.0, 50.0])
    np.testing.assert_allclose(result.sum(axis=0), [25.0, 35.0, 40.0])
    assert np.all(result > 0.0)


def test_field_reconstruction_preserves_margins_and_association():
    table = np.array([[12.0, 3.0, 5.0], [4.0, 20.0, 6.0], [9.0, 7.0, 34.0]])
    coordinates = association_coordinates(association_field(table))
    result = field_coordinates_to_table(
        coordinates, table.sum(axis=1), table.sum(axis=0)
    )
    np.testing.assert_allclose(result, table, atol=1e-8)


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_residual_reconstruction_uses_common_margins(residual):
    table = np.array([[12.0, 3.0, 5.0], [4.0, 20.0, 6.0], [9.0, 7.0, 34.0]])
    residual_values = poisson_independence_residuals(table, residual=residual)
    result = residual_coordinates_to_table(
        residual_values,
        table.sum(axis=1),
        table.sum(axis=0),
        residual=residual,
    )
    np.testing.assert_allclose(result.sum(axis=1), table.sum(axis=1))
    np.testing.assert_allclose(result.sum(axis=0), table.sum(axis=0))
    assert np.isfinite(result).all() and np.all(result > 0.0)


def test_zero_residual_reconstructs_independence_table():
    rows = np.array([20.0, 30.0, 50.0])
    columns = np.array([25.0, 35.0, 40.0])
    result = residual_coordinates_to_table(
        np.zeros((3, 3)), rows, columns, residual="pearson"
    )
    np.testing.assert_allclose(result, np.outer(rows, columns) / rows.sum())


def test_multinomial_deviance_is_zero_only_at_truth():
    truth = np.array([[20.0, 5.0], [10.0, 15.0]])
    assert multinomial_deviance_per_observation(truth, truth) == pytest.approx(0.0)
    prediction = ipf_to_margins(
        np.ones((2, 2)), truth.sum(axis=1), truth.sum(axis=0)
    )
    assert multinomial_deviance_per_observation(truth, prediction) > 0.0


def test_invalid_structural_support_refuses():
    with pytest.raises(TablePredictionRefusal):
        ipf_to_margins(np.eye(2), np.ones(2), np.ones(2))
    with pytest.raises(TablePredictionRefusal):
        field_coordinates_to_table(np.zeros((1, 1)), [1.0, 0.0], [0.5, 0.5])
def test_projected_residuals_are_rejected_as_an_incomplete_classical_baseline():
    with pytest.raises(ValueError, match="residual shape"):
        residual_coordinates_to_table(
            np.zeros((2, 2)), np.ones(3), np.ones(3), residual="pearson"
        )


def test_sparse_residual_seed_converges_to_exact_common_margins():
    residuals = np.array(
        [[-8.0, -8.0, 12.0], [10.0, -2.0, -8.0], [-8.0, 11.0, -8.0]]
    )
    rows = np.array([11.0, 10.0, 10.0])
    columns = np.array([5.0, 15.0, 11.0])
    result = residual_coordinates_to_table(
        residuals, rows, columns, residual="pearson"
    )
    np.testing.assert_allclose(result.sum(axis=1), rows, atol=1e-6)
    np.testing.assert_allclose(result.sum(axis=0), columns, atol=1e-6)
