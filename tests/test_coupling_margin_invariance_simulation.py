import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/development/simulate_coupling_margin_invariance.py"
RESULT = ROOT / "results/coupling_margin_invariance_simulation.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("margin_invariance_simulation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_field_is_exact_and_residuals_fail_after_margin_shift():
    payload = _load_script().run_simulation()
    records = payload["margin_shift_sweep"]

    assert max(row["field_maximum_absolute_count_error"] for row in records) < 1e-8
    assert max(
        row["field_multinomial_deviance_per_observation"] for row in records
    ) < 1e-12
    assert records[0][
        "pearson_residual_multinomial_deviance_per_observation"
    ] < 1e-12
    assert records[0][
        "deviance_residual_multinomial_deviance_per_observation"
    ] < 1e-12
    assert records[-1][
        "pearson_residual_multinomial_deviance_per_observation"
    ] > 0.005
    assert records[-1][
        "deviance_residual_multinomial_deviance_per_observation"
    ] > 0.004


def test_binary_witness_holds_odds_ratio_but_changes_pearson_association():
    witness = _load_script().binary_fixed_odds_ratio_example()
    balanced = witness["balanced_margins"]
    shifted = witness["shifted_margins"]

    assert balanced["coupling_coordinate"] == pytest.approx(
        shifted["coupling_coordinate"]
    )
    assert balanced["pearson_phi"] == pytest.approx(0.5)
    assert shifted["pearson_phi"] == pytest.approx(0.26307233602404273)


def test_committed_result_matches_the_deterministic_runner():
    expected = _load_script().run_simulation()
    observed = json.loads(RESULT.read_text())
    assert observed == expected
