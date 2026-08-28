from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from experiments.build_paper_tables import build


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "variance_weighted_nuclear_scgpt_hypergraph"

FROZEN_JSON_SHA256 = {
    "results/public_coupling_atlas_benchmark_v4_final_estimator.json": (
        "eb250bd749c92b7278e7e8c54e89f5126f3367ed489a91b664f1d664ea083195"
    ),
    "results/multiperturb_conditional_fields.json": (
        "a59d3b82b914991fbb2e08195e5401b2dde249158e5f9c923decfcc1e1df4507"
    ),
    "results/development/perturbfate_conditional_fields.json": (
        "cdf0c2cf66facfbbdfb03ac7a20c97d4cdaf4fc0bc9c63025e825dbeb1715c26"
    ),
    "results/resistrace_conditional_fields.json": (
        "edef0dba1d1dd94f19829088cb1fcd00f72dbacbc0695516b9c25afee6b20ffb"
    ),
    "results/arce_gse278572_conditional_field_confirmation.json": (
        "65d4bf6097a8fafee8e22f352c2c6c14fa13c1d8082073a4589d3ee693ef8b57"
    ),
    "results/arce_gse278572_postlock_controls.json": (
        "66d3343db745ad338091397226a9b93ca222df759107fbe46eddfcbee4c1a612"
    ),
    "results/perturbsci_module_validation.json": (
        "68032c3bc05fe0702edd8600cf91c86d0850b1db44857e67a42845e0fc164fc2"
    ),
    "results/development/perturbsci_conditional_sensitivity.json": (
        "b79006842352df2a07722b772b8018c1400353e85b115c9986a94ecfd74b0690"
    ),
    "results/development/factorial_coupling_stress_v1.json": (
        "76f521decdb64945c0bf52e46dd60688eb9705224a7ff50bda027ed4fc981b48"
    ),
    "results/development/factorial_coupling_bootstrap_calibration_v1.json": (
        "744d79c3bc03321f15352d1aedda2b2308056106aa85cb841d0463396c89af62"
    ),
    "results/gse143417_pokiseq_preflight_refusal.json": (
        "24f7ad70fbbfd4e7482809db58bd94d1156c1e22c2dd94fa77d66b1d6acdcf24"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token}")


def _load(path: str | Path) -> dict:
    resolved = ROOT / path
    return json.loads(resolved.read_text(), parse_constant=_reject_nonfinite)


def _assert_finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def _metric_row_matches(row: dict[str, str], method: dict) -> None:
    metrics = method["metrics"]
    intervals = method["target_bootstrap_95_ci"]
    assert float(row["primary_r"]) == metrics["pooled_pearson"]
    assert float(row["primary_r_ci_low"]) == intervals["pooled_pearson"][0]
    assert float(row["primary_r_ci_high"]) == intervals["pooled_pearson"][1]
    assert float(row["primary_standardized_rmse"]) == metrics["standardized_rmse"]
    assert float(row["primary_rmse_ci_low"]) == intervals["standardized_rmse"][0]
    assert float(row["primary_rmse_ci_high"]) == intervals["standardized_rmse"][1]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _macros() -> dict[str, str]:
    pattern = re.compile(r"^\\newcommand\{\\([^}]+)\}\{(.*)\}$")
    values = {}
    for line in (ROOT / "paper/results_macros.tex").read_text().splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def _stress_rate(result: dict, scenario: str, world: str) -> float:
    record = next(
        item
        for item in result["operating_characteristics"]
        if item["scenario"] == scenario and item["world"] == world
    )
    return record["metrics"]["unconditional_global_rejection"]["rate"]


def test_frozen_json_artifacts_have_expected_bytes_and_finite_numbers():
    for relative_path, expected_sha256 in FROZEN_JSON_SHA256.items():
        path = ROOT / relative_path
        assert _sha256(path) == expected_sha256
        _assert_finite(_load(relative_path))


def test_public_benchmark_tsv_matches_results_and_provenance():
    table = ROOT / "results/final_public_benchmark_table.tsv"
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)

    assert len(reader.fieldnames or []) == 29
    assert len(rows) == 8
    by_panel = {row["panel"]: row for row in rows}
    assert len(by_panel) == 8
    assert all(None not in row for row in rows)

    estimator = ROOT / "mapreg/coupling_fields.py"
    runner_by_panel = {
        "PerturbSci-Kinetics": "experiments/benchmark_public_coupling_fields.py",
        "Frangieh Perturb-CITE-seq": "experiments/benchmark_public_coupling_fields.py",
        "Papalexi ECCITE-seq": "experiments/benchmark_public_coupling_fields.py",
        "MultiPerturb-seq RNA-ATAC": (
            "experiments/benchmark_multiperturb_conditional_fields.py"
        ),
        "PerturbFate": "experiments/benchmark_perturbfate_conditional_fields.py",
        "ReSisTrace": "experiments/validate_resistrace_conditional_fields.py",
        "Arce T-cell RNA-protein confirmation": (
            "experiments/confirm_arce_gse278572_conditional_fields.py"
        ),
        "PoKI-seq held-donor confirmation": (
            "experiments/confirm_poki_gse143417_conditional_fields.py"
        ),
    }
    for panel, row in by_panel.items():
        assert _sha256(ROOT / row["result_artifact"]) == row["result_sha256"]
        assert _sha256(estimator) == row["estimator_sha256"]
        assert _sha256(ROOT / runner_by_panel[panel]) == row["benchmark_sha256"]
        if row["input_sha256"] not in {"NA", "MULTIPLE"}:
            input_artifact = ROOT / row["input_artifact"]
            if input_artifact.is_file():
                assert _sha256(input_artifact) == row["input_sha256"]

    atlas = _load("results/public_coupling_atlas_benchmark_v4_final_estimator.json")
    assert atlas["implementation"]["classical_residuals_sha256"] == _sha256(
        ROOT / "mapreg/classical_residuals.py"
    )
    atlas_panels = {panel["panel"]: panel for panel in atlas["panels"]}
    for panel_name in (
        "PerturbSci-Kinetics",
        "Frangieh Perturb-CITE-seq",
        "Papalexi ECCITE-seq",
    ):
        row = by_panel[panel_name]
        panel = atlas_panels[panel_name]
        assert int(row["n_targets"]) == panel["targets"]
        _metric_row_matches(row, panel["methods"][row["primary_method"]])
        comparison = panel["primary_vs_best_matched_competitor"]
        assert row["strongest_declared_baseline"] == comparison["best_competitor"]
        assert (
            float(row["strongest_baseline_standardized_rmse"])
            == comparison["best_competitor_standardized_rmse"]
        )
        interval = panel["pairing_signal"][
            "primary_minus_destroyed_pearson_bootstrap_95_ci"
        ]
        assert float(row["pairing_control_ci_low"]) == interval[0]
        assert float(row["pairing_control_ci_high"]) == interval[1]
        classical = panel["coordinate_family_comparison"]
        assert set(classical) == {
            "protocol",
            "coupling_field",
            "poisson_deviance_residual",
            "fixed_structured_coupling_minus_poisson",
        }
        assert set(classical["poisson_deviance_residual"]) == {
            "direct",
            "fixed_structured",
            "destroyed_links",
        }

    extra_panels = {
        "MultiPerturb-seq RNA-ATAC": _load(
            "results/multiperturb_conditional_fields.json"
        ),
        "PerturbFate": _load("results/development/perturbfate_conditional_fields.json"),
    }
    for panel_name, result in extra_panels.items():
        row = by_panel[panel_name]
        assert int(row["n_targets"]) == result["targets"]
        _metric_row_matches(row, result["methods"][row["primary_method"]])
        baseline = row["strongest_declared_baseline"]
        assert (
            float(row["strongest_baseline_standardized_rmse"])
            == result["methods"][baseline]["metrics"]["standardized_rmse"]
        )

    resistrace = _load("results/resistrace_conditional_fields.json")
    assert resistrace["status"] == "REFUSE_TREATMENT_CONTRAST_NOT_REPRODUCED"
    assert len(resistrace["arm_fields"]) == 8
    for arm in resistrace["arm_fields"].values():
        assert arm["destroyed_link_p_value"] == 1 / 65
        assert arm["destroyed_link_bh_q_value"] == 1 / 65
    for treatment in resistrace["treatment_minus_control_reproducibility"].values():
        low, high = treatment["lineage_bootstrap_cosine_ci95"]
        assert low <= 0 <= high

    arce = _load("results/arce_gse278572_conditional_field_confirmation.json")
    arce_row = by_panel["Arce T-cell RNA-protein confirmation"]
    assert int(arce_row["n_targets"]) == len(arce["targets"])
    _metric_row_matches(arce_row, arce["methods"][arce_row["primary_method"]])
    assert not arce["confirmation_gate"]["passed"]

    poki = _load("results/gse143417_pokiseq_preflight_refusal.json")
    poki_row = by_panel["PoKI-seq held-donor confirmation"]
    assert poki["status"] == "REFUSE_PREFLIGHT_STATE_OCCUPANCY"
    assert poki["stage_reached"] == "PREDICT_STATE_OCCUPANCY_PREFLIGHT"
    assert not poki["prediction_written"]
    assert not poki["result_written"]
    assert not poki["outcome_scored"]
    assert poki["failure"]["exception_message"] == (
        "state-occupancy preflight failed for Donor1:Stim:41BB"
    )
    assert poki["failure"]["failing_arm"] == {
        "donor": "Donor1",
        "context": "Stim",
        "construct": "41BB",
    }
    lock = _load(poki["provenance"]["preanalysis_lock"]["path"])
    assert poki["provenance"]["raw_archive"]["sha256"] == lock["source"]["sha256"]
    assert poki["provenance"]["raw_archive"]["bytes"] == lock["source"]["bytes"]
    for key in ("prepared_cache", "preanalysis_lock", "runner"):
        record = poki["provenance"][key]
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]
    raw_record = poki["provenance"]["raw_archive"]
    raw_path = ROOT / raw_record["path"]
    if raw_path.is_file():
        assert raw_path.stat().st_size == raw_record["bytes"]
        assert _sha256(raw_path) == raw_record["sha256"]
    for expected_output in poki["expected_outputs"].values():
        assert not expected_output["exists_after_failure"]
        assert expected_output["sha256"] is None
        assert not (ROOT / expected_output["path"]).exists()
    assert poki["public_protocol_history"] == {
        "disabled_outcome_freeze": {
            "commit": "2e5f47a8676000c743be0459b9d979262e7eb147",
            "url": "https://github.com/sushaan-k/coupling-fields-benchmark/commit/2e5f47a8676000c743be0459b9d979262e7eb147",
        },
        "outcome_access_authorization": {
            "commit": "044478d35d46783eba9d91e2ab17925327af0f92",
            "url": "https://github.com/sushaan-k/coupling-fields-benchmark/commit/044478d35d46783eba9d91e2ab17925327af0f92",
        },
        "protocol_metadata_correction": {
            "commit": "580932ed9a429e38f1de43d340c224db31c1cf9b",
            "tag": "protocol-v1.0.1",
            "url": "https://github.com/sushaan-k/coupling-fields-benchmark/commit/580932ed9a429e38f1de43d340c224db31c1cf9b",
        },
    }
    assert poki_row["primary_metric"] == "not scored"
    assert poki_row["result_artifact"] == (
        "results/gse143417_pokiseq_preflight_refusal.json"
    )

    expected_decisions = {
        "PerturbSci-Kinetics": ("PROMOTE", "REFUSE", "PROMOTE"),
        "Frangieh Perturb-CITE-seq": ("REFUSE", "REFUSE", "REFUSE"),
        "Papalexi ECCITE-seq": ("REFUSE", "REFUSE", "REFUSE"),
        "MultiPerturb-seq RNA-ATAC": ("REFUSE", "REFUSE", "REFUSE"),
        "PerturbFate": ("REFUSE", "REFUSE", "REFUSE"),
        "ReSisTrace": ("PROMOTE", "NA", "REFUSE"),
        "Arce T-cell RNA-protein confirmation": ("REFUSE", "REFUSE", "REFUSE"),
        "PoKI-seq held-donor confirmation": (
            "NOT_EVALUATED",
            "NOT_EVALUATED",
            "REFUSE_PREFLIGHT",
        ),
    }
    for panel, expected in expected_decisions.items():
        row = by_panel[panel]
        assert (
            row["pairing_signal_decision"],
            row["estimator_superiority_decision"],
            row["panel_decision"],
        ) == expected


def test_paper_macros_match_frozen_results():
    atlas = _load("results/public_coupling_atlas_benchmark_v4_final_estimator.json")
    panels = {panel["panel"]: panel for panel in atlas["panels"]}
    psci = panels["PerturbSci-Kinetics"]
    frangieh = panels["Frangieh Perturb-CITE-seq"]
    papalexi = panels["Papalexi ECCITE-seq"]
    multi = _load("results/multiperturb_conditional_fields.json")
    fate = _load("results/development/perturbfate_conditional_fields.json")
    resis = _load("results/resistrace_conditional_fields.json")
    arce = _load("results/arce_gse278572_conditional_field_confirmation.json")
    sensitivity = _load("results/development/perturbsci_conditional_sensitivity.json")
    module = _load("results/perturbsci_module_validation.json")
    stress = _load("results/development/factorial_coupling_stress_v1.json")

    def metric_macros(
        prefix: str, result: dict, method: str = PRIMARY
    ) -> dict[str, str]:
        record = result["methods"][method]
        targets = result["targets"]
        return {
            f"{prefix}N": str(len(targets) if isinstance(targets, list) else targets),
            f"{prefix}R": _fmt(record["metrics"]["pooled_pearson"]),
            f"{prefix}RLo": _fmt(record["target_bootstrap_95_ci"]["pooled_pearson"][0]),
            f"{prefix}RHi": _fmt(record["target_bootstrap_95_ci"]["pooled_pearson"][1]),
            f"{prefix}RMSE": _fmt(record["metrics"]["standardized_rmse"]),
        }

    expected = {}
    expected.update(metric_macros("PSci", psci))
    psci_primary = psci["methods"][PRIMARY]
    expected.update(
        {
            "PSciRMSELo": _fmt(
                psci_primary["target_bootstrap_95_ci"]["standardized_rmse"][0]
            ),
            "PSciRMSEHi": _fmt(
                psci_primary["target_bootstrap_95_ci"]["standardized_rmse"][1]
            ),
            "PSciDirectR": _fmt(psci["methods"]["direct"]["metrics"]["pooled_pearson"]),
            "PSciDestroyedR": _fmt(
                psci["methods"]["destroyed_links"]["metrics"]["pooled_pearson"]
            ),
        }
    )
    four_state = next(
        item
        for item in sensitivity["configurations"]
        if item["states_per_assay"] == 4
        and item["permutations"] == 64
        and item["pseudocount"] == 0.5
    )
    expected["PSciFourStateR"] = _fmt(four_state["primary"]["pooled_pearson"])
    recovery = module["nearest_neighbor_validation"]["recovery"]
    for key, record in (
        ("PSciNeighbor", recovery["primary"]),
        ("PSciDestroyedNeighbor", recovery["destroyed_links"]),
        ("PSciNeighborDiff", recovery["primary_minus_destroyed"]),
    ):
        expected[key] = _fmt(record["mean"])
        expected[f"{key}Lo"] = _fmt(record["target_bootstrap_95_ci"][0])
        expected[f"{key}Hi"] = _fmt(record["target_bootstrap_95_ci"][1])
    expected["PSciNeighborP"] = _fmt(
        recovery["primary_vs_random_label_permutation"]["one_sided_p_value"]
    )
    expected.update(metric_macros("Frangieh", frangieh))
    expected.update(metric_macros("Papalexi", papalexi))
    expected.update(metric_macros("Multi", multi))
    expected.update(metric_macros("PerturbFate", fate))

    nk = resis["treatment_minus_control_reproducibility"]["NK"]
    expected.update(
        {
            "ReSisArmP": "1/65",
            "ReSisArmQ": _fmt(
                next(iter(resis["arm_fields"].values()))["destroyed_link_bh_q_value"]
            ),
            "ReSisNKCos": _fmt(nk["replicate_contrast_cosine"]),
            "ReSisNKLo": _fmt(nk["lineage_bootstrap_cosine_ci95"][0]),
            "ReSisNKHi": _fmt(nk["lineage_bootstrap_cosine_ci95"][1]),
        }
    )
    expected.update(metric_macros("Arce", arce, "fixed_structured"))
    arce_primary = arce["methods"]["fixed_structured"]
    expected.update(
        {
            "ArceRMSELo": _fmt(
                arce_primary["target_bootstrap_95_ci"]["standardized_rmse"][0]
            ),
            "ArceRMSEHi": _fmt(
                arce_primary["target_bootstrap_95_ci"]["standardized_rmse"][1]
            ),
            "ArceDecision": "refuse",
            "StressNull": f"{100 * _stress_rate(stress, 'well_specified', 'population_null'):.1f}\\%",
            "StressPower": f"{100 * _stress_rate(stress, 'well_specified', 'fixed_target_0_effect'):.1f}\\%",
            "StressHeavyNull": f"{100 * _stress_rate(stress, 'heavy_tailed_blocks', 'population_null'):.1f}\\%",
            "StressGuidePower": f"{100 * _stress_rate(stress, 'guide_channel_misspecified', 'fixed_target_0_effect'):.1f}\\%",
            "StressObsPower": f"{100 * _stress_rate(stress, 'observation_channel_misspecified', 'fixed_target_0_effect'):.1f}\\%",
        }
    )
    assert _macros() == expected

    assert (
        psci_primary["metrics"]["standardized_rmse"]
        > psci["methods"]["endpoint_ridge"]["metrics"]["standardized_rmse"]
    )
    assert module["decisions"] == {
        "nearest_neighbor_recovery": "REFUSE",
        "neighbor_annotation_enrichment": {
            "CORUM": "REFUSE",
            "Reactome": "REFUSE",
        },
        "strict_module_validation": "REFUSE",
    }


def test_comparator_table_is_generated_from_frozen_results(tmp_path: Path):
    generated = tmp_path / "generated_comparator_tables.tex"
    build(generated)
    assert (
        generated.read_bytes()
        == (ROOT / "paper/generated_comparator_tables.tex").read_bytes()
    )
