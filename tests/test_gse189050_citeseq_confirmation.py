import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import confirm_gse189050_citeseq as confirmation


def _synthetic_features() -> list[tuple[str, str, str]]:
    genes = {}
    for panel in confirmation.PANELS.values():
        for identifier, symbol in zip(panel["rna_ids"], panel["rna_symbols"]):
            genes[(identifier, symbol)] = (
                f"GRCh38_{identifier}",
                f"GRCh38_{symbol}",
                "Gene Expression",
            )
    rows = list(genes.values())
    rows.append(("mm10_ENSMUSG1", "mm10_Gene1", "Gene Expression"))
    antibodies = {
        antibody for panel in confirmation.PANELS.values() for antibody in panel["adt"]
    }
    rows.extend((value, value, "Antibody Capture") for value in sorted(antibodies))
    rows.extend(
        (value, value, "Antibody Capture") for value in ("HT-1", "HT-2", "HT-3", "HT-4")
    )
    return rows


def _matrix_bytes(
    features: list[tuple[str, str, str]],
    coordinates: list[tuple[int, int, str]],
    columns: int = 2,
) -> bytes:
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        "% synthetic",
        f"{len(features)} {columns} {len(coordinates)}",
    ]
    lines.extend(f"{row} {column} {value}" for row, column, value in coordinates)
    return ("\n".join(lines) + "\n").encode()


def test_frozen_split_panels_and_configuration_grid():
    assert set(confirmation.RUNS) == set(
        confirmation.CALIBRATION_POOLS
        + confirmation.PILOT_POOLS
        + confirmation.HELD_POOLS
    )
    assert len(confirmation.PRIMARY_MARKERS) == 12
    assert len(set(confirmation.PRIMARY_RNA_IDS)) == 12
    assert len(confirmation.LEGACY_MARKERS) == 9
    assert len(confirmation._primary_configs()) == 144
    assert confirmation.PANELS["primary"]["minimum_informative"] == 108


def test_metadata_authorities_bind_46_physical_subjects_and_disclosed_conflict():
    inventory = confirmation._metadata_inventory()
    assert len(inventory["geo_by_subject"]) == 46
    assert set(inventory["clinical_conflicts"]) == {"SUB235957", "SUB236000"}
    assert sum(value == "calibration" for value in inventory["roles"].values()) == 9
    assert sum(value == "pilot" for value in inventory["roles"].values()) == 9
    assert sum(value == "held" for value in inventory["roles"].values()) == 28
    assert {
        len(value) for run, value in inventory["hto_by_pool"].items() if run < "s3"
    } == {4}


def test_feature_normalization_is_suffix_robust_but_not_fuzzy():
    assert confirmation._canonical_adt("CD11c_protein") == "CD11C"
    assert confirmation._canonical_adt("CD11c TotalSeq-A") == "CD11C"
    assert confirmation._canonical_adt("HT-1") == "HT1"
    assert confirmation._canonical_adt("CD16") != confirmation._canonical_adt("CD64")


def test_feature_resolution_requires_12_direct_unique_cognate_mappings():
    rows = _synthetic_features()
    resolved = confirmation._resolve_features(
        rows, "s1a", ("HT-1", "HT-2", "HT-3", "HT-4")
    )
    assert len(resolved["panel_rows"]["primary"]["rna"]) == 12
    assert len(resolved["panel_rows"]["primary"]["adt"]) == 12
    assert len(resolved["panel_rows"]["primary"]["rna"]) ** 2 == 144
    assert len(resolved["hto_rows"]) == 4
    duplicate = rows + [("CD14_protein", "CD14_protein", "Antibody Capture")]
    with pytest.raises(PermissionError, match="uniquely resolve"):
        confirmation._resolve_features(
            duplicate, "s1a", ("HT-1", "HT-2", "HT-3", "HT-4")
        )


def test_held_rna_reader_skips_poison_biological_adt_without_conversion():
    features = _synthetic_features()
    row_by_name = {name: index + 1 for index, (_, name, _) in enumerate(features)}
    coordinates = [
        (row_by_name["GRCh38_CD14"], 1, "5"),
        (row_by_name["mm10_Gene1"], 1, "1"),
        (row_by_name["HT-1"], 1, "25"),
        (row_by_name["HT-2"], 1, "1"),
        (row_by_name["CD14"], 1, "HELD_ADT_POISON"),
    ]
    matrix = io.BytesIO(_matrix_bytes(features, coordinates))
    pool = confirmation._stream_pool_matrix(
        "s1a",
        matrix,
        ["AAAC-1", "AAAG-1"],
        features,
        {
            "HT-1": "D1",
            "HT-2": "D2",
            "HT-3": "D3",
            "HT-4": "D4",
        },
        read_biological_adt=False,
    )
    audit = pool["audit"]
    assert audit["numeric_values_converted"]["biological_adt"] == 0
    assert (
        audit["biological_adt_coordinate_lines_skipped_without_value_conversion"] == 1
    )
    with pytest.raises(ValueError):
        confirmation._stream_pool_matrix(
            "s1a",
            io.BytesIO(_matrix_bytes(features, coordinates)),
            ["AAAC-1", "AAAG-1"],
            features,
            {
                "HT-1": "D1",
                "HT-2": "D2",
                "HT-3": "D3",
                "HT-4": "D4",
            },
            read_biological_adt=True,
        )


def test_hto_rule_has_fixed_boundaries_and_no_rescue():
    pool = {
        "hto": np.asarray(
            [
                [20, 0, 0, 0],
                [19, 0, 0, 0],
                [20, 3, 0, 0],
                [20, 5, 0, 0],
                [10, 10, 0, 0],
            ]
        ),
        "human_total": np.asarray([90, 90, 90, 90, 90]),
        "mouse_total": np.asarray([10, 10, 10, 10, 11]),
    }
    result = confirmation._hto_classification(pool)
    np.testing.assert_array_equal(result["accepted"], [True, False, True, False, False])
    assert not result["human"][-1]


def test_support_gate_scales_to_fixed_role_thresholds():
    records = {
        f"D{index}": {
            "eligible": True,
            "pool": "s2a" if index < 4 else "s4a",
        }
        for index in range(7)
    }
    gate = confirmation._support_gate("pilot", records, confirmation.PILOT_POOLS)
    assert gate["passes"]
    records["D6"]["eligible"] = False
    assert not confirmation._support_gate("pilot", records, confirmation.PILOT_POOLS)[
        "passes"
    ]


def test_pool_qc_uses_the_declared_strict_bounds_and_exact_tag_set():
    accepted = np.zeros(100, dtype=bool)
    accepted[:60] = True
    negative = np.zeros(100, dtype=bool)
    negative[60:80] = True
    ambiguous = np.zeros(100, dtype=bool)
    ambiguous[80:] = True
    top_index = np.tile(np.arange(4), 25)
    classification = {
        "human": np.ones(100, dtype=bool),
        "accepted": accepted,
        "negative": negative,
        "ambiguous": ambiguous,
        "top_index": top_index,
    }
    pool = {
        "tags": ("HT-1", "HT-2", "HT-3", "HT-4"),
        "barcodes": [f"BC{index}" for index in range(100)],
    }
    mapping = {f"HT-{index}": f"D{index}" for index in range(1, 5)}
    result = confirmation._pool_qc("s1a", pool, classification, mapping)
    assert result["passes"]
    assert set(result["represented_subjects"]) == set(mapping.values())
    classification["top_index"][:56] = 0
    result = confirmation._pool_qc("s1a", pool, classification, mapping)
    assert not result["checks"]["positive_donor_yield_ratio_at_most_4"]


def test_refit_temporarily_configures_the_12_marker_core(monkeypatch):
    observed = {}

    def fake_fit(records, donors, selection):
        observed["markers"] = tuple(confirmation.model_core.MARKERS)
        return {"ok": True}

    monkeypatch.setattr(confirmation.model_core, "_fit_models", fake_fit)
    result = confirmation._refit_panel("primary", {}, [], {})
    assert result == {"ok": True}
    assert observed["markers"] == confirmation.PRIMARY_MARKERS


def test_held_inference_combines_s5_for_independent_blocks():
    subjects = [f"D{index}" for index in range(12)]
    pools = ("s1b", "s2b", "s3b", "s4b", "s5a", "s5b")
    subject_pools = {
        subject: pools[index % 6] for index, subject in enumerate(subjects)
    }
    primary = np.full(12, 0.8)
    comparator = np.full(12, 1.0)
    result = confirmation._comparison(
        subjects,
        subject_pools,
        primary,
        comparator,
        held=True,
        gating=True,
    )
    assert result["passes"]
    assert set(result["independent_run_block_mean_differences"]) == {
        "s1b",
        "s2b",
        "s3b",
        "s4b",
        "s5",
    }
    assert result["donor_exact_sign_test"]["one_sided_p"] == 1 / 2**12


def test_pooled_poisson_interaction_is_the_saturated_log_odds_ratio():
    tables = np.asarray([[[[[10, 20], [30, 40]]]], [[[[5, 10], [15, 20]]]]])
    value = confirmation._pooled_loglinear_interaction(tables)
    pooled = tables.sum(axis=0)[0, 0]
    expected = math.log(pooled[0, 0] * pooled[1, 1] / (pooled[0, 1] * pooled[1, 0]))
    assert value[0, 0] == pytest.approx(expected)


def test_schema_access_record_has_real_timestamp_and_zero_outcome_access():
    access = json.loads(confirmation.DEFAULT_ACCESS.read_text())
    assert access["created_at_utc"] != "2026-08-28T00:00:00Z"
    assert access["archive_access"]["barcodes_member_content_opened"] is False
    assert access["archive_access"]["matrix_member_content_opened"] is False
    assert set(access["assay_value_access"].values()) == {0}


def test_public_stage_tags_are_distinct():
    assert (
        len(
            {
                confirmation.PROTOCOL_TAG,
                confirmation.DEVELOPMENT_TAG,
                confirmation.PREDICTION_TAG,
            }
        )
        == 3
    )


def test_protocol_binds_complete_mapreg_runtime_and_environment():
    required = {
        "mapreg/__init__.py",
        "mapreg/classical_residuals.py",
        "mapreg/coupling_fields.py",
        "mapreg/factorial_coupling.py",
        "mapreg/heterogeneity_adaptive_coupling.py",
        "mapreg/hierarchical_conditional_coupling.py",
        "mapreg/table_prediction.py",
        "requirements.txt",
        "pyproject.toml",
    }
    assert required <= set(confirmation.PROTOCOL_BINDINGS)


def test_public_binding_rejects_a_changed_transitive_dependency(
    monkeypatch, tmp_path: Path
):
    paths = ("experiments/runner.py", "mapreg/coupling_fields.py")
    tagged = {
        "experiments/runner.py": b"runner\n",
        "mapreg/coupling_fields.py": b"dependency\n",
    }
    for relative, value in tagged.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    commit = "a" * 40

    def fake_run(args, **kwargs):
        del kwargs
        if args[1] == "rev-list":
            return SimpleNamespace(stdout=commit + "\n")
        if args[1] == "ls-remote":
            return SimpleNamespace(stdout=f"{commit}\trefs/tags/frozen^{{}}\n")
        if args[1] == "show":
            relative = args[-1].split(":", 1)[1]
            return SimpleNamespace(stdout=tagged[relative])
        raise AssertionError(args)

    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation.subprocess, "run", fake_run)
    assert confirmation._require_public_tag("frozen", paths) == commit
    (tmp_path / "mapreg/coupling_fields.py").write_bytes(b"changed\n")
    with pytest.raises(PermissionError, match="differs from public tag"):
        confirmation._require_public_tag("frozen", paths)


def test_frozen_preflight_is_count_blind():
    if not Path(confirmation.DEFAULT_PREFLIGHT).exists():
        pytest.skip(
            "preflight artifact is generated immediately before protocol freeze"
        )
    payload = json.loads(confirmation.DEFAULT_PREFLIGHT.read_text())
    assert payload["status"] == "PASS_BEFORE_BARCODE_OR_MATRIX_VALUE_ACCESS"
    audit = payload["access_audit"]
    assert audit["barcode_values_read"] == 0
    assert audit["matrix_members_opened"] == 0
    assert audit["adt_numeric_values_read"] == 0
