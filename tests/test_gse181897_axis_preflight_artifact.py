from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "data/development/gse181897_source/axis_preflight_v1.json"
VERIFICATION = (
    ROOT / "data/development/gse181897_source/axis_preflight_verification_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tagged_sha256(tag: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tag}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def test_preflight_binds_input_axes_and_zero_numeric_access() -> None:
    preflight = json.loads(PREFLIGHT.read_text())
    assert preflight["status"] == "AXES_FROZEN_X_NUMERIC_UNREAD"
    assert preflight["source"]["archive_sha256"] == (
        "7fe58432f2f238319e81c9218eb35b5f7fbdae6f10f3d87ce9a6044ee851675b"
    )
    assert preflight["source"]["h5ad_sha256"] == (
        "183d7756c750fb0ca57f381512fe784df6249ec5a5478a9caf6a62df55cba56c"
    )
    assert preflight["source"]["acquisition"] == {
        "archive_present_at_start": True,
        "archive_removed_after_verification": True,
        "archive_verified_in_this_run": True,
        "h5ad_verified_in_this_run": True,
    }
    assert preflight["hdf5"]["matrix"]["shape"] == [136142, 20399]
    assert preflight["hdf5"]["obs"]["rows"] == 136142
    assert preflight["hdf5"]["var"]["rows"] == 20399
    assert preflight["source_plan"]["donor_count"] == 39
    assert preflight["source_plan"]["selected_rows"] == 39 * 128
    assert preflight["numeric_access"]["decoded_X_entries"] == 0
    assert preflight["numeric_access"]["matrix_datasets_indexed"] == []


def test_verification_binds_exact_artifact_and_implementation() -> None:
    verification = json.loads(VERIFICATION.read_text())
    assert verification["status"] == "VERIFIED_AXES_NUMERIC_X_UNREAD"
    assert verification["artifact"]["sha256"] == _sha256(PREFLIGHT)
    assert verification["axis_code_freeze"]["reducer_sha256"] == _tagged_sha256(
        verification["axis_code_freeze"]["tag"],
        verification["axis_code_freeze"]["reducer_path"],
    )
    assert verification["axis_code_freeze"]["test_sha256"] == _tagged_sha256(
        verification["axis_code_freeze"]["tag"],
        verification["axis_code_freeze"]["test_path"],
    )
    assert verification["axis_code_freeze"]["candidate_is_ancestor"] is True
    assert verification["numeric_access"]["source_reduction_authorized"] is False
    assert verification["numeric_access"]["held_batches_8_11_numeric_values_read"] == 0
    assert verification["numeric_access"]["non_control_numeric_values_read"] == 0
