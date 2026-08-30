from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_gse239452_citeseq as confirmation


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/gse239452_citeseq_post_access_correction.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_signed_root_deviance_inversion_preserves_a_negative_coordinate() -> None:
    rows = np.asarray([184, 328])
    columns = np.asarray([256, 256])
    target = -4.024939692792063

    reconstructed = confirmation._classical_table(
        target, rows, columns, "deviance"
    )

    np.testing.assert_allclose(reconstructed.sum(axis=1), rows)
    np.testing.assert_allclose(reconstructed.sum(axis=0), columns)
    assert confirmation._fractional_deviance(reconstructed) == pytest.approx(
        target, abs=1e-10
    )
    assert not np.allclose(
        reconstructed, np.outer(rows, columns) / confirmation.CELL_BUDGET
    )


def test_post_access_correction_is_bound_and_reconstructs_every_coordinate() -> None:
    payload = json.loads(RESULT.read_text())
    audit = payload["held"]["residual_reconstruction_audit"]

    assert payload["schema"] == "gse239452-post-access-correction/1.0"
    assert payload["status"] == "POST_ACCESS_CORRECTION_COMPLETE"
    assert payload["outcome_blind"] is False
    assert payload["original_sealed_artifacts_overwritten"] is False
    assert payload["development"]["pilot_gate"]["passes"] is True
    assert payload["held"]["gate"]["passes"] is True
    assert audit["tables_checked"] == 729
    assert audit["original_coordinate_mismatches"] == 80
    assert audit["corrected_coordinate_mismatches"] == 0
    assert all(
        sample["primary_prediction_matches_original"]
        for sample in payload["held"]["samples"]
    )

    corrected_runner = payload["corrected_runner"]
    correction_runner = payload["correction_runner"]
    assert corrected_runner["sha256"] == _sha256(ROOT / corrected_runner["path"])
    assert correction_runner["sha256"] == _sha256(ROOT / correction_runner["path"])
    for binding in payload["original_sealed_artifacts"].values():
        assert binding["sha256"] == _sha256(ROOT / binding["path"])
