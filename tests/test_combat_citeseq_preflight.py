from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.preflight_combat_citeseq import (
    ELIGIBLE_CELL_TYPES,
    MARKER_SPECS,
    MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE,
    _composition_contract,
    _frozen_sample_contract,
    _strict_json_loads,
    build_preflight,
    write_preflight,
)


OFFICIAL_COMPOSITION_CSV = (
    Path(__file__).resolve().parents[1] / "data/confirmation/combat_citeseq/"
    "COMBAT_CITEseq_Composition-PerSample_CellType_Counts_and_"
    "PercentFrequencies_out_of_all_PBMCs.csv"
)


def _legacy_dataframe(
    handle: h5py.File,
    name: str,
    index: list[str],
    columns: dict[str, list[str]],
    *,
    categorical: set[str],
) -> h5py.Group:
    group = handle.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = list(columns)
    string_dtype = h5py.string_dtype("utf-8")
    index_dataset = group.create_dataset(
        "_index", data=np.asarray(index, dtype=object), dtype=string_dtype
    )
    index_dataset.attrs["encoding-type"] = "string-array"
    index_dataset.attrs["encoding-version"] = "0.2.0"
    categories_group = group.create_group("__categories")
    for column, values in columns.items():
        if column not in categorical:
            dataset = group.create_dataset(
                column, data=np.asarray(values, dtype=object), dtype=string_dtype
            )
            dataset.attrs["encoding-type"] = "string-array"
            dataset.attrs["encoding-version"] = "0.2.0"
            continue
        categories = list(dict.fromkeys(values))
        lookup = {value: index for index, value in enumerate(categories)}
        category_dataset = categories_group.create_dataset(
            column,
            data=np.asarray(categories, dtype=object),
            dtype=string_dtype,
        )
        category_dataset.attrs["ordered"] = False
        codes = group.create_dataset(
            column,
            data=np.asarray([lookup[value] for value in values], dtype=np.int16),
        )
        codes.attrs["categories"] = category_dataset.ref
    return group


def _sparse_matrix(
    parent: h5py.File | h5py.Group,
    name: str,
    shape: tuple[int, int],
) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = shape
    group.create_dataset("data", data=np.asarray([], dtype=np.float32))
    group.create_dataset("indices", data=np.asarray([], dtype=np.int32))
    group.create_dataset("indptr", data=np.zeros(shape[0] + 1, dtype=np.int32))
    return group


def _synthetic_h5ad(
    path: Path,
    *,
    feature_override: tuple[str, str, str] | None = None,
    duplicate_feature: str | None = None,
    drop_combat_id: str | None = None,
    obs_override: tuple[str, str, str] | None = None,
    extra_sample_for_combat_id: str | None = None,
    replace_nan_cell_type: tuple[str, str] | None = None,
    reduce_eligible_pool: str | None = None,
) -> None:
    composition, official_pairs = _composition_contract(OFFICIAL_COMPOSITION_CSV)
    assert composition["valid"]
    expected_samples, _ = _frozen_sample_contract(official_pairs)
    obs = {
        "COMBAT_ID": [],
        "scRNASeq_sample_ID": [],
        "Source": [],
        "Institute": [],
        "Annotation_cell_type": [],
    }
    obs_names = []
    for combat_id, contract in expected_samples.items():
        if combat_id == drop_combat_id:
            continue
        for cell_index in range(MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE):
            obs_names.append(f"{combat_id}_cell_{cell_index:04d}")
            obs["COMBAT_ID"].append(combat_id)
            obs["scRNASeq_sample_ID"].append(contract["scRNASeq_sample_ID"])
            obs["Source"].append(contract["source"])
            obs["Institute"].append(contract["institute"])
            obs["Annotation_cell_type"].append(
                ELIGIBLE_CELL_TYPES[cell_index % len(ELIGIBLE_CELL_TYPES)]
            )
        obs_names.append(f"{combat_id}_cell_nan")
        obs["COMBAT_ID"].append(combat_id)
        obs["scRNASeq_sample_ID"].append(contract["scRNASeq_sample_ID"])
        obs["Source"].append(contract["source"])
        obs["Institute"].append(contract["institute"])
        obs["Annotation_cell_type"].append("nan")

    if obs_override is not None:
        combat_id, field, value = obs_override
        for index, observed_id in enumerate(obs["COMBAT_ID"]):
            if observed_id == combat_id:
                obs[field][index] = value
    if extra_sample_for_combat_id is not None:
        contract = expected_samples[extra_sample_for_combat_id]
        for cell_index in range(MINIMUM_ELIGIBLE_CELLS_PER_SAMPLE + 1):
            obs_names.append(f"{extra_sample_for_combat_id}_outside_{cell_index:04d}")
            obs["COMBAT_ID"].append(extra_sample_for_combat_id)
            obs["scRNASeq_sample_ID"].append(
                f"{extra_sample_for_combat_id}-OUTSIDE-PBCa"
            )
            obs["Source"].append("OUTSIDE_SOURCE")
            obs["Institute"].append(contract["institute"])
            obs["Annotation_cell_type"].append("OUTSIDE_CELL_TYPE")
    if replace_nan_cell_type is not None:
        combat_id, value = replace_nan_cell_type
        index = obs_names.index(f"{combat_id}_cell_nan")
        obs["Annotation_cell_type"][index] = value
    if reduce_eligible_pool is not None:
        index = obs["COMBAT_ID"].index(reduce_eligible_pool)
        obs["Annotation_cell_type"][index] = "nan"

    features = [
        {
            "name": marker,
            "gene_id": f"{ensembl_id}.12",
            "feature_type": "Gene Expression",
        }
        for marker, ensembl_id, _ in MARKER_SPECS
    ]
    features.extend(
        {
            "name": adt_name,
            "gene_id": "",
            "feature_type": "Antibody Capture",
        }
        for _, _, adt_name in MARKER_SPECS
    )
    features.append(
        {
            "name": "OTHER",
            "gene_id": "ENSG99999999999",
            "feature_type": "Gene Expression",
        }
    )
    if feature_override is not None:
        target_name, field, value = feature_override
        feature = next(row for row in features if row["name"] == target_name)
        feature[field] = value
    if duplicate_feature is not None:
        feature = next(row for row in features if row["name"] == duplicate_feature)
        features.append(dict(feature))

    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        _legacy_dataframe(
            handle,
            "obs",
            obs_names,
            obs,
            categorical=set(obs),
        )
        _legacy_dataframe(
            handle,
            "var",
            [row["name"] for row in features],
            {
                "gene_ids": [row["gene_id"] for row in features],
                "feature_types": [row["feature_type"] for row in features],
            },
            categorical={"feature_types"},
        )
        shape = (len(obs_names), len(features))
        _sparse_matrix(handle, "X", shape)
        layers = handle.create_group("layers")
        layers.attrs["encoding-type"] = "dict"
        layers.attrs["encoding-version"] = "0.1.0"
        _sparse_matrix(layers, "raw", shape)


def test_preflight_passes_exact_feature_and_frozen_sample_contracts(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source)

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    assert result["schema_version"] == "combat_citeseq_metadata_preflight_v3"
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_PASS"
    assert result["input"] == {
        "filename": source.name,
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert result["dataframes"]["obs"]["rows"] == 97 * 513
    assert result["dataframes"]["var"]["rows"] == 19
    assert result["matrices"]["X"]["shape"] == [97 * 513, 19]
    assert result["matrices"]["layers"]["raw"]["encoding_type"] == "csr_matrix"
    composition = result["composition_contract"]
    assert composition["valid"]
    assert composition["input"] == {
        "filename": OFFICIAL_COMPOSITION_CSV.name,
        "bytes": 33391,
        "md5": "c4e635d7f16b3f7e3e66571a3359de3b",
        "sha256": ("2ad7e92ab122ee52986d5748dbb23c335c02ec1f1f244943ce46fff94c585157"),
    }
    assert composition["data_rows"] == 873
    assert composition["unique_sample_ids"] == 97
    assert composition["official_pairs_by_combat_id"]["S00024"] == {
        "scRNASeq_sample_ID": "S00024-Ja003E-PBCa",
        "total_pbmc_count": 4176,
    }

    features = result["marker_feature_candidates"]
    assert features["complete_exact_panel"]
    assert features["missing_exact_names"] == []
    assert features["duplicate_exact_features"] == []
    assert not features["fuzzy_fallback_used"]
    cd44 = next(row for row in features["markers"] if row["marker"] == "CD44")
    assert cd44["rna_expected"] == {
        "symbol": "CD44",
        "version_stripped_ensembl_id": "ENSG00000026508",
        "feature_type": "Gene Expression",
    }
    assert cd44["adt_expected"] == {
        "name": "AB_humanCD44",
        "feature_type": "Antibody Capture",
    }
    assert all(len(row["rna_exact_matches"]) == 1 for row in features["markers"])
    assert all(len(row["adt_exact_matches"]) == 1 for row in features["markers"])

    samples = result["frozen_sample_contract"]
    assert samples["complete_frozen_cohorts"]
    assert samples["expected_role_counts"] == {
        "calibration": 12,
        "pilot_adaptive_development": 24,
        "oxford_held_confirmatory": 51,
        "st_georges_held_confirmatory": 10,
    }
    assert samples["present_role_counts"] == samples["expected_role_counts"]
    assert samples["expected_metadata_by_combat_id"]["S00024"] == {
        "role": "calibration",
        "source": "COVID_CRIT",
        "institute": "Oxford",
        "scRNASeq_sample_ID": "S00024-Ja003E-PBCa",
        "total_pbmc_count": 4176,
    }
    assert samples["expected_metadata_by_combat_id"]["U00501"] == {
        "role": "st_georges_held_confirmatory",
        "source": "Flu",
        "institute": "St_Georges",
        "scRNASeq_sample_ID": "U00501-Ua005E-PBUa",
        "total_pbmc_count": 1171,
    }
    assert samples["missing_combat_ids"] == []
    assert samples["missing_exact_pairs"] == []
    assert samples["source_institute_mismatches"] == []
    assert samples["outside_universe_rows_for_designated_combat_ids"] == {}
    assert samples["insufficient_eligible_cells"] == []
    assert samples["designated_sample_counts"]["S00024"] == {
        "raw_rows": 513,
        "eligible_cells": 512,
        "composition_total_pbmc_count": 4176,
    }
    assert not samples["total_pbmc_count_equality_asserted"]

    cell_types = result["cell_type_contract"]
    assert cell_types["observed_values_are_allowed_subset"]
    assert cell_types["observed_nonmissing_values"] == sorted(ELIGIBLE_CELL_TYPES)
    assert cell_types["unexpected_values"] == []
    assert "MNP|PLT" not in cell_types["exact_allowed_values"]
    assert result["warnings"] == []
    assert result["access_audit"]["matrix_payload_reads"] == 0
    assert "/X/data" in result["access_audit"]["matrix_payload_paths_not_read"]
    assert "/layers/raw/data" in result["access_audit"]["matrix_payload_paths_not_read"]


def test_preflight_never_indexes_a_matrix_payload(tmp_path, monkeypatch):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source)
    original_getitem = h5py.Dataset.__getitem__
    matrix_reads: list[str] = []

    def guarded_getitem(dataset, key):
        name = dataset.name
        if name == "/X" or name.startswith("/X/") or name.startswith("/layers/"):
            matrix_reads.append(name)
            raise AssertionError(f"matrix payload read: {name}")
        if name == "/raw/X" or name.startswith("/raw/X/"):
            matrix_reads.append(name)
            raise AssertionError(f"matrix payload read: {name}")
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_PASS"
    assert matrix_reads == []


@pytest.mark.parametrize(
    ("feature_override", "missing_name"),
    [
        (("CD4", "name", "cd4"), "CD4"),
        (("CD7", "gene_id", "ENSG00000173763.1"), "CD7"),
        (("CD14", "feature_type", "gene expression"), "CD14"),
        (("AB_humanCD44", "name", "AB_CD44"), "AB_humanCD44"),
        (("AB_CD47", "feature_type", "Gene Expression"), "AB_CD47"),
    ],
)
def test_feature_contract_has_no_fuzzy_fallback(
    tmp_path, feature_override, missing_name
):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, feature_override=feature_override)

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    features = result["marker_feature_candidates"]
    assert not features["complete_exact_panel"]
    assert features["missing_exact_names"] == [missing_name]
    assert not features["fuzzy_fallback_used"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"
    assert "the exact RNA/ADT feature contract failed" in result["warnings"]


def test_duplicate_exact_feature_is_a_contract_failure(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, duplicate_feature="CD7")

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    features = result["marker_feature_candidates"]
    assert features["missing_exact_names"] == []
    assert features["duplicate_exact_features"] == ["CD7"]
    assert not features["complete_exact_panel"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"


@pytest.mark.parametrize(
    ("fixture_kwargs", "report_key", "combat_id"),
    [
        ({"drop_combat_id": "S00024"}, "missing_combat_ids", "S00024"),
        (
            {"obs_override": ("S00024", "Source", "COVID_MILD")},
            "source_institute_mismatches",
            "S00024",
        ),
        (
            {"obs_override": ("U00501", "Institute", "Oxford")},
            "source_institute_mismatches",
            "U00501",
        ),
        (
            {
                "obs_override": (
                    "S00024",
                    "scRNASeq_sample_ID",
                    "S00024-OUTSIDE-PBCa",
                )
            },
            "missing_exact_pairs",
            "S00024",
        ),
    ],
)
def test_frozen_sample_metadata_discrepancies_fail(
    tmp_path, fixture_kwargs, report_key, combat_id
):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, **fixture_kwargs)

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    samples = result["frozen_sample_contract"]
    assert not samples["complete_frozen_cohorts"]
    assert any(
        item == combat_id or item.get("combat_id") == combat_id
        for item in samples[report_key]
    )
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"
    assert "the frozen sample metadata contract failed" in result["warnings"]


def test_each_designated_sample_requires_512_exactly_eligible_cells(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, reduce_eligible_pool="S00024")

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    samples = result["frozen_sample_contract"]
    assert samples["designated_sample_counts"]["S00024"] == {
        "raw_rows": 513,
        "eligible_cells": 511,
        "composition_total_pbmc_count": 4176,
    }
    assert samples["insufficient_eligible_cells"] == [
        {"combat_id": "S00024", "eligible_cells": 511, "required_minimum": 512}
    ]
    assert not samples["complete_frozen_cohorts"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"


def test_cell_type_matching_is_exact_and_mnp_plt_is_not_an_h5ad_alias(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, replace_nan_cell_type=("S00024", "MNP|PLT"))

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    cell_types = result["cell_type_contract"]
    assert not cell_types["observed_values_are_allowed_subset"]
    assert cell_types["unexpected_values"] == ["MNP|PLT"]
    assert not cell_types["fuzzy_fallback_used"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"
    assert (
        "the exact Annotation_cell_type allowlist contract failed" in result["warnings"]
    )


def test_extra_h5ad_timepoint_for_a_designated_donor_is_outside_the_universe(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source, extra_sample_for_combat_id="S00045")

    result = build_preflight(source, OFFICIAL_COMPOSITION_CSV)

    samples = result["frozen_sample_contract"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_PASS"
    assert samples["complete_frozen_cohorts"]
    assert samples["designated_sample_counts"]["S00045"] == {
        "raw_rows": 513,
        "eligible_cells": 512,
        "composition_total_pbmc_count": 6289,
    }
    assert samples["outside_universe_rows_for_designated_combat_ids"] == {
        "S00045": {
            "rows": 513,
            "scRNASeq_sample_ID_values": ["S00045-OUTSIDE-PBCa"],
        }
    }
    assert result["cell_type_contract"]["unexpected_values"] == []


def test_output_is_written_and_duplicate_json_keys_are_rejected(tmp_path):
    source = tmp_path / "combat.h5ad"
    output = tmp_path / "reports" / "preflight.json"
    _synthetic_h5ad(source)

    expected = write_preflight(source, OFFICIAL_COMPOSITION_CSV, output)

    assert _strict_json_loads(output.read_text()) == expected
    with pytest.raises(ValueError, match="duplicate JSON key: status"):
        _strict_json_loads('{"status":"one","status":"two"}')


def test_input_cannot_be_overwritten(tmp_path):
    source = tmp_path / "combat.h5ad"
    _synthetic_h5ad(source)

    with pytest.raises(ValueError, match="input and output paths must differ"):
        write_preflight(source, OFFICIAL_COMPOSITION_CSV, source)


def test_composition_csv_content_is_cryptographically_frozen(tmp_path):
    source = tmp_path / "combat.h5ad"
    composition = tmp_path / "composition.csv"
    _synthetic_h5ad(source)
    composition.write_bytes(OFFICIAL_COMPOSITION_CSV.read_bytes() + b"\n")

    result = build_preflight(source, composition)

    assert not result["composition_contract"]["valid"]
    assert result["status"] == "PREFLIGHT_METADATA_CONTRACT_FAIL"
    assert "the frozen composition CSV contract failed" in result["warnings"]
