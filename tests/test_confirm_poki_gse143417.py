import hashlib
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.confirm_poki_gse143417_conditional_fields import (
    CONTROL_CONSTRUCTS,
    DEVELOPMENT_DONOR,
    EXPECTED_METADATA_EXCLUSION,
    HELD_DONOR,
    MARKER_GENES,
    STIM,
    TGFB,
    _eligibility,
    _fit_states,
    _implementation_sha256,
    _module_scores,
    _read_construct_calls,
    _require_pretruth_designation,
    _require_authorized_lock,
    analyze_cache,
    fit_pretruth_predictions,
)


def _synthetic_cache() -> dict[str, np.ndarray]:
    special = [
        *CONTROL_CONSTRUCTS,
        EXPECTED_METADATA_EXCLUSION,
        "TNGFR",
        "TGFBR2DN",
        "TGFBR241BB",
        "TGFBR2MYD88",
    ]
    roster = special + [f"GENE{index}" for index in range(29)]
    assert len(roster) == 36
    rng = np.random.default_rng(19)
    cell_id = []
    donor_values = []
    context_values = []
    replicate_values = []
    construct_values = []
    counts = []
    library = []
    for donor_index, donor in enumerate((DEVELOPMENT_DONOR, HELD_DONOR)):
        for context_index, context in enumerate((STIM, TGFB)):
            for construct_index, construct in enumerate(roster):
                cells = (
                    29
                    if construct == EXPECTED_METADATA_EXCLUSION
                    and donor == HELD_DONOR
                    and context == TGFB
                    else 31
                )
                for cell in range(cells):
                    first_state = cell % 3
                    shift = (context_index * (construct_index % 3) + donor_index) % 3
                    second_state = (first_state + shift) % 3
                    level = np.array([1.0, 5.0, 20.0])
                    marker = np.concatenate(
                        (
                            np.full(3, level[first_state]),
                            np.full(3, level[second_state]),
                        )
                    )
                    marker += rng.binomial(1, 0.1, size=6)
                    cell_id.append(f"{donor}:{context}:{construct}:{cell}")
                    donor_values.append(donor)
                    context_values.append(context)
                    replicate_values.append(cell % 3 + 1)
                    construct_values.append(construct)
                    counts.append(marker)
                    library.append(1_000.0)
    return {
        "cell_id": np.asarray(cell_id),
        "donor": np.asarray(donor_values),
        "context": np.asarray(context_values),
        "replicate": np.asarray(replicate_values),
        "construct": np.asarray(construct_values),
        "marker_counts": np.asarray(counts),
        "library_size": np.asarray(library),
        "marker_names": np.asarray(MARKER_GENES),
        "construct_roster": np.asarray(roster),
        "source_sha256": np.asarray("synthetic"),
    }


def test_metadata_gate_is_frozen_before_scores():
    eligible, support = _eligibility(_synthetic_cache())
    assert len(eligible) == 35
    assert EXPECTED_METADATA_EXCLUSION not in eligible
    assert min(support["TNGFR"].values()) >= 30


def test_construct_call_parser_keeps_one_unique_call_per_terminal_barcode():
    text = "barcode\tconstruct\n"
    text += "sample_AAACCCGGGTTTAAAA-1\tGFP\n"
    text += "sample_AAACCCGGGTTTAAAA-1\tGFP\n"
    text += "sample_TTTTGGGGCCCCAAAA-1\tGFP;mCherry\n"
    calls, roster = _read_construct_calls(gzip.compress(text.encode()))
    assert calls == {"AAACCCGGGTTTAAAA": "GFP"}
    assert {value.upper() for value in roster} == {"GFP", "MCHERRY"}


def test_state_thresholds_use_only_donor1_stim_controls():
    data = _synthetic_cache()
    first, second = _module_scores(data)
    _, _, thresholds = _fit_states(data, first, second)
    changed = {name: values.copy() for name, values in data.items()}
    held = changed["donor"].astype(str) == HELD_DONOR
    changed["marker_counts"][held] = 0.0
    changed_first, changed_second = _module_scores(changed)
    _, _, changed_thresholds = _fit_states(changed, changed_first, changed_second)
    assert thresholds == changed_thresholds
    assert thresholds["calibration"] == "Donor1 Stim pooled GFP+mCherry mono-construct cells"


def test_state_occupancy_failure_refuses_instead_of_dropping_target():
    data = _synthetic_cache()
    construct = data["construct"].astype(str)
    mask = (
        (data["donor"].astype(str) == HELD_DONOR)
        & (data["context"].astype(str) == TGFB)
        & (construct == "TNGFR")
    )
    data["marker_counts"][mask, :3] = 0.0
    with pytest.raises(ValueError, match="state-occupancy preflight"):
        fit_pretruth_predictions(data, permutations=4)


def test_synthetic_runner_uses_factorial_full_residual_benchmark():
    result = analyze_cache(_synthetic_cache(), permutations=4, bootstraps=20)
    assert result["support"]["query_constructs"] == 33
    assert "pooled GFP+mCherry" in result["design"]["factorial_target"]
    losses = result["common_held_table"]
    assert set(losses).issuperset({
        "field_zero",
        "field_direct",
        "field_destroyed",
        "field_fixed",
        "pearson_direct",
        "pearson_fixed",
        "deviance_direct",
        "deviance_fixed",
    })
    assert {
        "field_scalar",
        "field_nuclear",
        "field_hypergraph",
        "field_endpoint",
        "field_label_permuted",
    }.issubset(losses)
    assert len(losses["field_fixed"]["per_target"]) == 33
    assert len(result["primary_gate"]["best_classical_candidates"]) == 4
    assert "field_fixed_minus_best_matched_deviance_95_ci" in result["primary_gate"]
    assert all(
        component.startswith(("receptor:", "domain:"))
        for component in result["architecture_hypergraph"]["components"]
    )
    assert result["representation_space_secondary"]["pearson_fixed"][
        "metrics"
    ]["standardized_rmse"] >= 0.0


def test_pretruth_stage_never_forms_held_target_tgfb_tables(monkeypatch):
    import experiments.confirm_poki_gse143417_conditional_fields as runner

    data = _synthetic_cache()
    donor = data["donor"].astype(str)
    context = data["context"].astype(str)
    construct = np.asarray(
        [runner._canonical_construct(value) for value in data["construct"].astype(str)]
    )
    forbidden = (
        (donor == HELD_DONOR)
        & (context == TGFB)
        & ~np.isin(construct, CONTROL_CONSTRUCTS)
    )
    original = runner._table_statistics

    def guarded(first, second, mask, seed, permutations):
        assert not np.any(mask & forbidden)
        return original(first, second, mask, seed, permutations)

    monkeypatch.setattr(runner, "_table_statistics", guarded)
    record = fit_pretruth_predictions(data, permutations=4)
    assert record["stage"] == "PRETRUTH_PREDICTIONS_WRITTEN"
    assert np.asarray(record["coordinate_predictions"]["pearson_fixed"]).shape[1] == 9


def test_cli_lock_rejects_unsealed_or_modified_runner(tmp_path):
    runner = Path(
        "experiments/confirm_poki_gse143417_conditional_fields.py"
    ).resolve()
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    relative = str(runner.relative_to(Path.cwd()))
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "SEALED",
                "outcome_access_authorized": True,
                "public_freeze_commit": "b" * 40,
                "public_freeze_url": (
                    "https://github.com/example/benchmark/blob/"
                    + "b" * 40
                    + "/docs/protocol.md"
                ),
                "implementation_sha256": {relative: digest},
            }
        )
    )
    assert _require_authorized_lock(lock)["status"] == "SEALED"
    lock.write_text(
        json.dumps(
            {
                "status": "SEALED_PREOUTCOME",
                "outcome_access_authorized": False,
                "implementation_sha256": {relative: digest},
            }
        )
    )
    with pytest.raises(PermissionError):
        _require_authorized_lock(lock)


def test_lock_generator_binds_runner_protocol_and_classical_baseline(tmp_path):
    output = tmp_path / "preanalysis_lock.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_poki_gse143417_confirmation.py",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lock = json.loads(output.read_text())
    assert lock["status"] == "SEALED_PREOUTCOME"
    assert lock["outcome_access_authorized"] is False
    assert lock["public_freeze_commit"] is None
    assert lock["public_freeze_url"] is None
    assert "experiments/confirm_poki_gse143417_conditional_fields.py" in lock[
        "implementation_sha256"
    ]
    assert "mapreg/classical_residuals.py" in lock["implementation_sha256"]
    assert "scripts/authorize_poki_gse143417_scoring.py" in lock[
        "implementation_sha256"
    ]
    assert any("best-classical" in condition for condition in lock["primary_gate"])


def test_outcome_authorization_requires_and_records_public_protocol_commit(tmp_path):
    output = tmp_path / "authorized_lock.json"
    commit = "c" * 40
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_poki_gse143417_confirmation.py",
            "--authorize-outcome",
            "--public-freeze-commit",
            commit,
            "--public-freeze-url",
            f"https://github.com/example/benchmark/blob/{commit}/docs/protocol.md",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lock = json.loads(output.read_text())
    assert lock["status"] == "SEALED"
    assert lock["outcome_access_authorized"] is True
    assert lock["public_freeze_commit"] == commit


def test_score_requires_exact_pretruth_prediction_designation(tmp_path):
    predictions = tmp_path / "predictions.json"
    cache = tmp_path / "cache.npz"
    predictions.write_text('{"stage":"PRETRUTH_PREDICTIONS_WRITTEN"}\n')
    cache.write_bytes(b"sealed-cache")
    designation = tmp_path / "designation.json"
    designation.write_text(
        json.dumps(
            {
                "status": "SEALED_FOR_SCORING",
                "pretruth_predictions_sha256": hashlib.sha256(
                    predictions.read_bytes()
                ).hexdigest(),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "implementation_sha256": _implementation_sha256(),
                "pretruth_predictions_public_commit": "a" * 40,
                "pretruth_predictions_public_url": (
                    "https://github.com/example/benchmark/blob/"
                    + "a" * 40
                    + "/results/predictions.json"
                ),
            }
        )
    )
    assert (
        _require_pretruth_designation(designation, predictions, cache)["status"]
        == "SEALED_FOR_SCORING"
    )
    predictions.write_text('{"stage":"MUTATED"}\n')
    with pytest.raises(PermissionError, match="prediction bytes"):
        _require_pretruth_designation(designation, predictions, cache)


def test_score_refuses_a_prediction_without_public_commit_binding(tmp_path):
    predictions = tmp_path / "predictions.json"
    cache = tmp_path / "cache.npz"
    predictions.write_text('{"stage":"PRETRUTH_PREDICTIONS_WRITTEN"}\n')
    cache.write_bytes(b"sealed-cache")
    designation = tmp_path / "designation.json"
    designation.write_text(
        json.dumps(
            {
                "status": "SEALED_FOR_SCORING",
                "pretruth_predictions_sha256": hashlib.sha256(
                    predictions.read_bytes()
                ).hexdigest(),
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "implementation_sha256": _implementation_sha256(),
            }
        )
    )
    with pytest.raises(PermissionError, match="public commit"):
        _require_pretruth_designation(designation, predictions, cache)
