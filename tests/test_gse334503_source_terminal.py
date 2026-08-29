from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/development/gse334503_source_candidate_v1.json"
DECISION = ROOT / "results/development/gse334503_source_terminal_decision_v1.json"
RESULT_SHA256 = "f585df257b55b861e7d3efc2ae16fe7109ff4c78b8f02a67e2d7c10cd9f41bc3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def test_gse334503_source_failure_is_terminal_and_held_batches_remain_sealed() -> None:
    result = json.loads(RESULT.read_text())
    decision = json.loads(DECISION.read_text())
    _finite(result)
    _finite(decision)

    assert _sha256(RESULT) == RESULT_SHA256
    assert decision["result"]["sha256"] == RESULT_SHA256
    assert decision["status"] == "TERMINAL_SOURCE_GO_GATE_FAILED"
    assert decision["rerun_permitted"] is False
    assert decision["b3_numeric_access_authorized"] is False
    assert decision["candidate_frozen"] is False

    assert result["status"] == "SOURCE_GO_GATE_FAILED"
    assert result["numeric_batches_processed"] == ["Batch1", "Batch2"]
    assert result["forbidden_numeric_batches"] == ["Batch3", "Batch4", "Batch5"]
    assert result["candidate"] is None
    assert result["b3_numeric_access_gate_passed"] is False
    gate = result["development"]["source_go_gate"]
    assert gate["passes"] is False
    assert gate["favorable_donors"] == 6
    assert gate["fold_improvements"] == {"Batch1": True, "Batch2": False}
    assert gate["relative_loss_reduction_vs_matched_graph_zero"] < 0.0

    assert not (
        ROOT / "data/development/gse334503_source/reduced_batch_3_v1.npz"
    ).exists()
    assert not (
        ROOT / "data/development/gse334503_source/reduction_batch_3_manifest_v1.json"
    ).exists()

    for relative, expected in result["implementation"]["files_sha256"].items():
        assert _sha256(ROOT / relative) == expected
