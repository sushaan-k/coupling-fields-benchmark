"""Zero-download checks of the completed, released prediction comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return json.loads((ROOT / "results/development/stephenson_predictive_reanalysis.json").read_text())


def test_all_exact_fits_and_original_donors_are_present(result):
    assert result["roles"] == {"calibration": 12, "pilot": 24, "held_site": 56}
    for phase in ("calibration", "source"):
        fitted = result["fits"][phase]["random_effects"]
        assert fitted["status"] == "FITTED"
        assert fitted["failed_pairs"] == []
        assert sorted(int(r["pair"]) for r in fitted["pair_fits"]) == list(range(81))
        assert all(r["status"] == "OK" for r in fitted["pair_fits"])
        for row in fitted["pair_fits"]:
            donors = set(row["informative_donors"].split(" | "))
            omitted = set(row["uninformative_donors"].split(" | ")) - {""}
            assert len(donors) == int(row["k"])
            assert not donors & omitted
            assert len(donors | omitted) == (12 if phase == "calibration" else 36)
    expected = [r["donor"] for r in result["methods"]["original_hierarchy"]["donor_results"]]
    assert len(expected) == len(set(expected)) == 56
    for method in result["methods"].values():
        assert method["status"] == "SCORED"
        assert [r["donor"] for r in method["donor_results"]] == expected


def test_every_reported_loss_recomputes_from_retained_pairs(result):
    for method in result["methods"].values():
        losses = []
        for donor in method["donor_results"]:
            assert len(donor["pair_losses"]) == 81
            pairs = [v for v in donor["pair_losses"] if v is not None]
            assert len(pairs) == donor["informative_pairs"]
            assert donor["loss"] == pytest.approx(np.mean(pairs), abs=1e-14)
            losses.append(donor["loss"])
        assert method["mean_loss"] == pytest.approx(np.mean(losses), abs=1e-14)


def test_original_loss_reproduces_the_frozen_experiment(result):
    frozen = json.loads((ROOT / "results/stephenson_citeseq_confirmation.json").read_text())
    expected = {r["donor"]: r for r in frozen["donor_results"]}
    for row in result["methods"]["original_hierarchy"]["donor_results"]:
        assert row["loss"] == pytest.approx(expected[row["donor"]]["losses"]["primary"], abs=2e-8)
        assert row["informative_pairs"] == expected[row["donor"]]["informative_pairs"]
    assert result["original_loss_max_absolute_error"] < 2e-8


def test_imported_fits_have_verifiable_archived_script_bytes(result):
    for phase in ("calibration", "source"):
        provenance = result["fits"][phase]["random_effects"].get("imported_fit_provenance")
        if provenance is None:
            continue
        for version in provenance.get("versions", [provenance]):
            archived = ROOT / version["script_path"]
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == version["binding"]["r_script_sha256"]
            assert set(version["included_pairs"]) <= set(range(81))


def test_qc_summary_uses_the_same_predictions_and_primary_intervals(result):
    path = ROOT / "results/development/stephenson_predictive_reanalysis.json"
    summary = json.loads(path.with_name("stephenson_predictive_reanalysis_summary.json").read_text())
    assert summary["bindings"]["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    full, qc = summary["cohorts"]["original_56"], summary["cohorts"]["assay_qc_54"]
    assert qc["excluded_donors"] == ["C-8914", "C-8939"]
    for key, comparison in result["comparisons"].items():
        assert full["comparisons"][key] == comparison
    for name, method in result["methods"].items():
        retained = [r for r in method["donor_results"] if r["donor"] not in qc["excluded_donors"]]
        assert len(retained) == qc["methods"][name]["donors"] == 54
        assert qc["methods"][name]["alpha"] == method["alpha"]
        assert qc["methods"][name]["loss"]["mean"] == pytest.approx(
            np.mean([r["loss"] for r in retained]), abs=1e-14)
