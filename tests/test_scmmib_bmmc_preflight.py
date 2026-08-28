import json
from pathlib import Path

from experiments.preflight_scmmib_bmmc import build_preflight


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/development/scmmib_bmmc_metadata_v1"
FROZEN_RESULT = ROOT / "results/development/scmmib_bmmc_metadata_preflight.json"


def _preflight():
    feature_file = DATA / "s1d1_filtered_feature_bc_matrix.h5"
    if feature_file.is_file():
        return build_preflight(
            DATA / "BMMC_RNA+ADT_p10_metadata.csv.gz",
            feature_file,
        )
    return json.loads(FROZEN_RESULT.read_text())


def test_bmmc_preflight_is_site_and_physical_donor_disjoint():
    result = _preflight()
    split = result["prospective_split"]
    assert split["fit"]["sites"] == ["site1", "site2"]
    assert split["development"]["sites"] == ["site4"]
    assert split["held"]["sites"] == ["site3"]
    assert split["held"]["donors"] == ["11466", "15078", "28045"]
    assert split["donor_disjoint"]
    assert split["site_disjoint"]
    assert result["metadata_snapshot"]["split"]["held"]["cells"] == 3203


def test_bmmc_preflight_has_a_locked_matched_marker_panel_without_outcome_access():
    result = _preflight()
    markers = result["marker_preflight"]
    assert markers["raw_antibody_features"] == 140
    assert markers["biological_antibody_features"] == 134
    assert markers["exact_rna_adt_name_matches"] == 37
    assert markers["locked_ordered_pairs"] == 100
    assert set(markers["locked_biology_only_panel"]).issubset(markers["exact_matches"])
    audit = result["access_audit"]
    assert "all site 3 feature-level RNA outcome counts" in audit["not_decoded"]
    assert "all site 3 feature-level ADT outcome counts" in audit["not_decoded"]
    assert "any held 2x2 table or held coupling statistic" in audit["not_decoded"]
    assert not audit["covid_sanger_touched"]
    assert not audit["large_scmmib_archive_downloaded"]


def test_bmmc_preflight_records_the_three_donor_inference_limit():
    result = _preflight()
    limit = result["inference_limit"]
    assert limit["held_physical_donors"] == 3
    assert limit["minimum_two_sided_exact_sign_p"] == 0.25
    assert limit["minimum_one_sided_exact_sign_p"] == 0.125


def test_bmmc_donor_powered_alternative_reaches_exact_sign_resolution():
    result = _preflight()
    split = result["donor_powered_alternative"]
    assert split["fit"]["donors"] == ["11466", "19593"]
    assert split["development"]["donors"] == ["15078"]
    assert split["held"]["donors"] == [
        "10886",
        "12710",
        "13272",
        "16710",
        "18303",
        "28045",
    ]
    assert split["physical_donor_disjoint"]
    assert not split["site_disjoint"]
    assert split["site_overlap_declared"]
    assert not split["original_partition_flags_used"]
    assert split["all_roles_have_all_broad_lineages"]
    assert split["minimum_two_sided_exact_sign_p"] == 0.03125
    assert split["minimum_one_sided_exact_sign_p"] == 0.015625
    assert split["development"]["sites"] == [
        "site1",
        "site2",
        "site3",
        "site4",
    ]
