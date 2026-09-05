import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.development import assay_qc_sensitivity as qc


ROOT = Path(__file__).resolve().parents[1]


def test_marker_diagnostics_describe_only_selected_panel_counts():
    counts = np.ones((4, 9, 8), dtype=np.int32)
    counts[2:] = 0
    counts[2, [0, 2, 8], 0] = 1
    data = {
        "rna_counts": np.ones_like(counts), "adt_counts": counts,
        "donor_ids": np.array(["source1", "source2", "C-8914", "C-8939"]),
        "sample_ids": np.array(["s1", "s2", "s3", "s4"]),
        "roles": np.array(["calibration", "pilot", "held_site", "held_site"]),
        "markers": np.array(qc.biology.stephenson.MARKERS),
        "cell_types": np.full((4, 8), "T"),
        "barcodes": np.tile([str(index) for index in range(8)], (4, 1)),
    }
    result = qc.summarize_counts(data)
    assert [row["total_adt_counts"] for row in result["donors"]] == [72, 72, 3, 0]
    assert result["donors"][2]["cells_with_any_panel_adt"] == 1
    assert result["donors"][3]["cells_with_any_panel_adt"] == 0
    assert len(result["donors"][2]["markers"]) == 9
    assert not set(result) & {"barcodes", "rna_counts", "adt_counts", "cell_types"}


def test_published_qc_sensitivity_replays_from_aggregate_diagnostics():
    diagnostic_path = ROOT / "data/development/stephenson_assay_qc_diagnostics.json"
    diagnostics = json.loads(diagnostic_path.read_text())
    confirmation = json.loads(qc.CONFIRMATION.read_text())
    classical = json.loads(qc.CLASSICAL.read_text())
    original = json.loads((ROOT / "results/development/stephenson_assay_qc_sensitivity.json").read_text())
    replay = qc.analyze(diagnostics, confirmation, classical)
    assert replay == {key: value for key, value in original.items() if key != "bindings"}
    assert original["bindings"]["diagnostics_sha256"] == hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    assert hashlib.sha256(qc.CONFIRMATION.read_bytes()).hexdigest() == (
        "5eb5fd2b41df7f4f7d822a92765ffe69854dcbe5f572f2db35cf433d7dd0adb1"
    )
    assert original["retained_held_donors"] == 54
    assert len(original["adt_diagnostics"]["donors"]) == 92
    assert original["next_lowest_panel_count_among_other_donors"] == 6560
    assert not original["models_refitted"]
    assert not original["predictions_retuned"]
    assert not original["confirmatory"]
    assert not set(qc.EXCLUDED) & {row["donor"] for row in original["donor_results"]}
