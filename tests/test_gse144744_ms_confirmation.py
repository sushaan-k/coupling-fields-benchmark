from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_gse144744_ms as confirmation
from mapreg.streamed_matrix_market import MatrixMarketAccessAudit, TarAxes


EXPECTED_HELD_LIBRARIES = (
    (("HH-OX-43", "HH-OX-44"), 6, (1601, 2410), "HASH-HH-OX-43-44-61-62_O"),
    (("HH-OX-45", "HH-OX-46"), 4, (2888, 1998), "HASH-HH-OX-45-46-49-50_E"),
    (("HH-OX-47", "HH-OX-48"), 6, (2086, 2302), "HASH-HH-OX-47-48-53-54_C"),
    (("HH-OX-49", "HH-OX-50"), 2, (1267, 2655), "HASH-HH-OX-45-46-49-50_D"),
    (("HH-OX-51", "HH-OX-52"), 1, (2184, 3183), "HASH-HH-OX-59-60-51-52_G"),
    (("HH-OX-53", "HH-OX-54"), 1, (2004, 2286), "HASH-HH-OX-47-48-53-54_A"),
    (("HH-OX-55", "HH-OX-56"), 4, (2380, 2050), "HASH-HH-OX-55-56-57-58_K"),
    (("HH-OX-57", "HH-OX-58"), 3, (2920, 2436), "HASH-HH-OX-55-56-57-58_K"),
    (("HH-OX-59", "HH-OX-60"), 3, (2614, 2169), "HASH-HH-OX-59-60-51-52_H"),
    (("HH-OX-61", "HH-OX-62"), 1, (2150, 2056), "HASH-HH-OX-43-44-61-62_M"),
)

HELD_EXPERIMENT_BY_PAIR = {
    ("HH-OX-47", "HH-OX-48"): "HE-MK-015",
    ("HH-OX-53", "HH-OX-54"): "HE-MK-015",
    ("HH-OX-45", "HH-OX-46"): "HE-MK-016",
    ("HH-OX-49", "HH-OX-50"): "HE-MK-016",
    ("HH-OX-51", "HH-OX-52"): "HE-MK-017",
    ("HH-OX-59", "HH-OX-60"): "HE-MK-017",
    ("HH-OX-43", "HH-OX-44"): "HE-MK-018",
    ("HH-OX-55", "HH-OX-56"): "HE-MK-018",
    ("HH-OX-57", "HH-OX-58"): "HE-MK-018",
    ("HH-OX-61", "HH-OX-62"): "HE-MK-018",
}


def _source_experiment(donor: str) -> str:
    for pair, experiment in confirmation.SOURCE_EXPERIMENT_BY_PAIR.items():
        if donor in pair:
            return experiment
    raise AssertionError(donor)


def _held_annotations() -> dict[str, dict[str, str]]:
    rows = {}
    for left, right in confirmation.HELD_PAIRS:
        experiment = HELD_EXPERIMENT_BY_PAIR[(left, right)]
        rows[left] = {"donor": left, "group": "PPMS", "exp_name": experiment}
        rows[right] = {"donor": right, "group": "HI3", "exp_name": experiment}
    return rows


def test_frozen_constants_marker_axis_and_source_experiment_folds():
    assert confirmation.CELL_BUDGET == 512
    assert confirmation.CANDIDATE_MARKER_COUNT == 29
    assert confirmation.MINIMUM_LOCKED_MARKERS == 12
    assert confirmation.MINIMUM_VALID_SOURCE_DONORS == 18
    assert confirmation.MINIMUM_VALID_HELD_DONORS == 20
    assert confirmation.MINIMUM_PAIR_FRACTION == 1.0
    assert len(confirmation.SOURCE_DONORS) == 18
    assert len(confirmation.HELD_DONORS) == 20

    markers = confirmation._markers()
    assert len(markers) == len({row["rna"] for row in markers}) == 29
    assert len({row["adt"] for row in markers}) == 29
    assert {row["adt"] for row in markers}.isdisjoint(
        {"ADT-CD45RA", "ADT-CD57", "ADT-HLA-DR"}
    )

    records = {
        donor: {"exp_name": _source_experiment(donor)}
        for donor in confirmation.SOURCE_DONORS
    }
    folds = confirmation._source_folds(records)
    assert tuple(folds) == confirmation.SOURCE_EXPERIMENTS
    assert set().union(*(set(donors) for donors in folds.values())) == set(
        confirmation.SOURCE_DONORS
    )
    assert sum(len(donors) for donors in folds.values()) == 18
    assert {experiment: len(donors) for experiment, donors in folds.items()} == {
        "HE-MK-002": 2,
        "HE-MK-003": 4,
        "HE-MK-004": 4,
        "HE-MK-005": 4,
        "HE-MK-006": 4,
    }


def test_frozen_held_library_selection_uses_one_exact_library_per_pair():
    metadata = {}
    expected = {}
    for pair, suffix, counts, hash_pool in EXPECTED_HELD_LIBRARIES:
        for donor, count, group in zip(pair, counts, ("PPMS", "HI3")):
            sample = f"{donor}_{suffix}"
            expected[donor] = sample
            for index in range(count):
                metadata[f"{donor}:{index}"] = {
                    "donor": donor,
                    "cohort": "PPMS_HI",
                    "V_10X": "V3",
                    "group": group,
                    "nCount_ADT": "1",
                    "sample_10X": sample,
                    "HASH": hash_pool,
                }

    assert confirmation._selected_libraries(metadata, "held") == expected
    assert all(
        expected[left].rsplit("_", 1)[1] == expected[right].rsplit("_", 1)[1]
        for left, right in confirmation.HELD_PAIRS
    )

    metadata.pop("HH-OX-43:0")
    with pytest.raises(
        confirmation.ConfirmationRefusal, match="HELD_LIBRARY_SELECTION_DIFFERS"
    ):
        confirmation._selected_libraries(metadata, "held")


def test_source_lock_subsets_both_modalities_and_masks_unsupported_pairs(monkeypatch):
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 8)
    monkeypatch.setattr(confirmation, "MINIMUM_BINARY_MARGIN", 2)
    monkeypatch.setattr(confirmation, "MINIMUM_LOCKED_MARKERS", 2)

    markers = [
        {"rna": f"RNA-{index}", "adt": f"ADT-{index}"} for index in range(3)
    ]
    selected = {
        donor: [f"{donor}:cell-{index}" for index in range(8)]
        for donor in confirmation.SOURCE_DONORS
    }
    metadata = {
        cell: {"nCount_RNA": "100", "exp_name": _source_experiment(donor)}
        for donor, cells in selected.items()
        for cell in cells
    }
    rna_panels = []
    adt_panels = []
    for donor_index, _ in enumerate(confirmation.SOURCE_DONORS):
        rna = np.asarray(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        if donor_index == 0:
            rna[:, 2] = 0
        adt = np.repeat(np.arange(4, dtype=np.int64), 2)[:, None]
        adt = np.repeat(adt, 3, axis=1)
        rna_panels.append(rna)
        adt_panels.append(adt)

    records, locked, support = confirmation._records_from_counts(
        selected,
        metadata,
        np.vstack(rna_panels),
        np.vstack(adt_panels),
        markers,
    )
    assert locked == markers[:2]
    assert support["locked_candidate_indices"] == [0, 1]
    assert support["rna_valid_source_donors"] == [18, 18, 17]
    assert support["adt_valid_source_donors"] == [18, 18, 18]
    assert support["minimum_informative_pairs_per_donor"] == 4
    for record in records.values():
        assert record["tables"].shape == (2, 2, 2, 2)
        assert record["rna_profile"].shape == (2,)
        assert record["adt_profile"].shape == (2,)
        np.testing.assert_array_equal(record["informative"], np.ones((2, 2), bool))

    original = np.arange(16, dtype=np.int64).reshape(2, 2, 2, 2) + 1
    mask_records = {
        "donor": {
            "tables": original.copy(),
            "informative": np.asarray([[True, False], [False, True]]),
        }
    }
    masked = confirmation._masked_tables(mask_records, ("donor",))[0]
    np.testing.assert_array_equal(masked[0, 0], original[0, 0])
    np.testing.assert_array_equal(masked[1, 1], original[1, 1])
    np.testing.assert_array_equal(masked[0, 1], np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(masked[1, 0], np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(mask_records["donor"]["tables"], original)


def test_matrix_market_axis_mapping_and_access_audit_are_forwarded_without_io(
    monkeypatch, tmp_path: Path
):
    archive = tmp_path / "must-not-be-opened.tar.gz"
    axes = TarAxes(
        features=(("id-1", "G1"), ("id-2", "G2"), ("id-3", "G3")),
        barcodes=("cell-1", "cell-2"),
        matrix_shape=(3, 2),
    )
    audit = MatrixMarketAccessAudit(
        declared_entries=4,
        entries_seen=4,
        authorized_column_entries=4,
        unauthorized_column_entries=0,
        authorized_column_unrequested_row_entries=0,
        selected_entries_materialized=4,
        value_tokens_lexically_validated=4,
        value_tokens_converted=4,
        unauthorized_value_tokens_converted=0,
        row_major_monotone=False,
        column_major_monotone=True,
    )
    observed = {}

    def fake_axes(*args):
        observed["axes_args"] = args
        return axes

    def fake_subset(path, received_axes, member, requested_rows, authorized_cells):
        observed["subset_args"] = (
            path,
            received_axes,
            member,
            requested_rows,
            authorized_cells,
        )
        return np.asarray([[20, 10], [40, 30]]), audit

    monkeypatch.setattr(confirmation, "read_tar_axes", fake_axes)
    monkeypatch.setattr(confirmation, "read_tar_matrix_subset", fake_subset)
    counts, access = confirmation._axis_and_counts(
        archive,
        "matrix.mtx",
        "genes.tsv",
        "barcodes.tsv",
        ["G2", "G1"],
        ["cell-2", "cell-1"],
    )

    assert observed["axes_args"] == (
        archive,
        "matrix.mtx",
        "genes.tsv",
        "barcodes.tsv",
    )
    assert observed["subset_args"][3] == {"G2": 1, "G1": 0}
    assert list(observed["subset_args"][3]) == ["G2", "G1"]
    assert observed["subset_args"][4] == ("cell-2", "cell-1")
    np.testing.assert_array_equal(counts, [[20, 10], [40, 30]])
    assert access["selected_entries_materialized"] == 4
    assert access["unauthorized_value_tokens_converted"] == 0
    assert access["value_tokens_converted"] == 4


def test_exact_pair_sign_gate_uses_ten_matched_pairs(monkeypatch):
    monkeypatch.setattr(confirmation, "BOOTSTRAPS", 2_000)
    metadata = _held_annotations()
    comparator = {donor: 1.0 for donor in confirmation.HELD_DONORS}

    primary = {}
    for pair_index, pair in enumerate(confirmation.HELD_PAIRS):
        value = 0.8 if pair_index < 9 else 1.0001
        primary.update({donor: value for donor in pair})
    passing = confirmation._comparison(
        primary, comparator, metadata, require_full_gate=True
    )
    assert passing["favorable_pairs"] == 9
    assert passing["exact_matched_pair_sign_p"] == pytest.approx(11 / 1024)
    assert passing["bootstrap_unit"] == "author-matched donor pair"
    assert passing["passes"]

    for pair_index, pair in enumerate(confirmation.HELD_PAIRS):
        value = 0.8 if pair_index < 8 else 1.0001
        primary.update({donor: value for donor in pair})
    failing = confirmation._comparison(
        primary, comparator, metadata, require_full_gate=True
    )
    assert failing["favorable_pairs"] == 8
    assert failing["exact_matched_pair_sign_p"] == pytest.approx(56 / 1024)
    assert not failing["passes"]


def test_adt_stage_requires_the_common_full_held_map(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 8)
    monkeypatch.setattr(confirmation, "HELD_DONORS", ("D1", "D2"))
    monkeypatch.setattr(confirmation, "HELD_PAIRS", (("D1", "D2"),))
    monkeypatch.setattr(confirmation, "MINIMUM_VALID_HELD_DONORS", 2)
    monkeypatch.setattr(confirmation, "MINIMUM_PAIR_FRACTION", 1.0)
    monkeypatch.setattr(confirmation, "_private", lambda path: Path(path))
    monkeypatch.setattr(confirmation, "_sha256", lambda path: "frozen-hash")
    monkeypatch.setattr(
        confirmation,
        "_manifest_file",
        lambda name: {
            "name": name,
            "url": f"https://invalid.example/{name}",
            "bytes": 1,
            "sha256": "frozen-hash",
        },
    )
    monkeypatch.setattr(
        confirmation, "_download", lambda url, destination, size, stage: destination
    )
    monkeypatch.setattr(confirmation, "_metadata", lambda path: {})
    selected = {
        donor: [f"{donor}:cell-{index}" for index in range(8)]
        for donor in ("D1", "D2")
    }
    monkeypatch.setattr(confirmation, "_selected_cells", lambda metadata, role: selected)
    monkeypatch.setattr(
        confirmation,
        "_selected_libraries",
        lambda metadata, role: {"D1": "D1_1", "D2": "D2_1"},
    )
    markers = [
        {"rna": "RNA-0", "adt": "ADT-0"},
        {"rna": "RNA-1", "adt": "ADT-1"},
    ]
    monkeypatch.setattr(
        confirmation, "_load_source_models", lambda: (markers, {}, "classical")
    )
    panel = np.repeat(np.arange(4, dtype=np.int64), 2)[:, None]
    counts = np.vstack([np.repeat(panel, 2, axis=1)] * 2)
    monkeypatch.setattr(
        confirmation,
        "_stage_axis_and_counts",
        lambda *args: (counts, {"synthetic": True}),
    )
    rna_valid = {"value": np.ones((2, 2), dtype=bool)}
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda path: (
            {"status": "PREDICTIONS_FROZEN"}
            if path == confirmation.PREDICTION_RESULT
            else {"valid": rna_valid["value"].tolist()}
        ),
    )

    result = confirmation.adt_stage(tmp_path / "scratch", tmp_path / "adt.npz")
    assert result["valid_donor_counts"] == [2, 2]
    assert result["informative_pair_counts"] == [4, 4]
    assert result["minimum_pairs"] == 4

    rna_valid["value"][0, 0] = False
    with pytest.raises(
        confirmation.ConfirmationRefusal,
        match="HELD_LOCKED_MAP_IS_NOT_FULLY_SUPPORTED",
    ):
        confirmation.adt_stage(tmp_path / "scratch", tmp_path / "refused.npz")


def test_runtime_identity_binds_executable_packages_and_thread_environment(
    monkeypatch,
):
    expected = json.loads(confirmation.RUNTIME.read_text())
    assert expected["python_executable"] == (
        "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/"
        "Python3.framework/Versions/3.9/bin/python3.9"
    )
    assert (expected["python_version"], expected["numpy_version"], expected["scipy_version"]) == (
        "3.9.6",
        "2.0.2",
        "1.13.1",
    )
    observed = {
        key: expected[key]
        for key in (
            "python_executable",
            "python_version",
            "numpy_version",
            "scipy_version",
        )
    }
    monkeypatch.setattr(confirmation, "_runtime", lambda: observed.copy())
    for key, value in expected["thread_environment"].items():
        monkeypatch.setenv(key, value)
    confirmation._require_runtime()

    observed["python_executable"] = "/different/python"
    with pytest.raises(PermissionError, match="runtime differs at python_executable"):
        confirmation._require_runtime()


def test_rna_stage_rejects_metadata_hash_before_parsing_or_matrix_access(
    monkeypatch, tmp_path: Path
):
    records = {
        "GSE144744_metadata_per_cell.csv.gz": {
            "name": "metadata.csv.gz",
            "url": "https://invalid.example/metadata",
            "bytes": 10,
            "sha256": "expected-metadata",
        },
        "GSE144744_RNA_counts.tar.gz": {
            "name": "rna.tar.gz",
            "url": "https://invalid.example/rna",
            "bytes": 20,
            "sha256": "expected-rna",
        },
    }
    monkeypatch.setattr(confirmation, "_manifest_file", records.__getitem__)
    monkeypatch.setattr(
        confirmation, "_download", lambda url, destination, size, stage: destination
    )
    monkeypatch.setattr(
        confirmation,
        "_sha256",
        lambda path: "wrong-metadata" if path.name == "metadata.csv.gz" else "expected-rna",
    )
    monkeypatch.setattr(
        confirmation,
        "_metadata",
        lambda path: (_ for _ in ()).throw(AssertionError("metadata was parsed")),
    )
    monkeypatch.setattr(
        confirmation,
        "_axis_and_counts",
        lambda *args: (_ for _ in ()).throw(AssertionError("matrix was opened")),
    )
    monkeypatch.setattr(
        confirmation, "_load_source_models", lambda: ([], {}, "classical")
    )

    with pytest.raises(
        confirmation.ConfirmationRefusal, match="FROZEN_METADATA_SHA256_MISMATCH"
    ):
        confirmation.rna_stage(tmp_path, tmp_path / "private.npz")


def test_failed_prediction_refuses_adt_before_download(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda path: {"status": "TERMINAL_REFUSAL"},
    )
    monkeypatch.setattr(
        confirmation,
        "_download",
        lambda *args: (_ for _ in ()).throw(AssertionError("download began")),
    )
    with pytest.raises(
        PermissionError, match="prediction did not authorize held ADT access"
    ):
        confirmation.adt_stage(tmp_path / "scratch", tmp_path / "adt.npz")


def test_score_authorization_rejects_post_authorization_prediction_change(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    public = {
        "source": tmp_path / "source.json",
        "source_journal": tmp_path / "source.jsonl",
        "rna": tmp_path / "rna.json",
        "rna_journal": tmp_path / "rna.jsonl",
        "prediction": tmp_path / "prediction.json",
        "adt": tmp_path / "adt.json",
        "adt_journal": tmp_path / "adt.jsonl",
    }
    for path in public.values():
        path.write_text("frozen\n")
    rna_private, adt_private = tmp_path / "rna.npz", tmp_path / "adt.npz"
    rna_private.write_bytes(b"rna-state")
    adt_private.write_bytes(b"adt-state")
    monkeypatch.setattr(confirmation, "SOURCE_RESULT", public["source"])
    monkeypatch.setattr(confirmation, "RNA_RESULT", public["rna"])
    monkeypatch.setattr(confirmation, "PREDICTION_RESULT", public["prediction"])
    monkeypatch.setattr(confirmation, "ADT_RESULT", public["adt"])
    monkeypatch.setattr(
        confirmation,
        "JOURNALS",
        {
            "source": public["source_journal"],
            "rna": public["rna_journal"],
            "adt": public["adt_journal"],
        },
    )
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(confirmation, "SCORE_AUTHORIZATION", authorization)
    authorization.write_text(json.dumps({
        "status": "SCORE_AUTHORIZED",
        "public_artifacts": {
            str(path.relative_to(tmp_path)): confirmation._sha256(path)
            for path in confirmation._score_public_paths()
        },
        "private_state_sha256": {
            "rna": confirmation._sha256(rna_private),
            "adt": confirmation._sha256(adt_private),
        },
    }))
    confirmation._validate_score_authorization(rna_private, adt_private)
    public["prediction"].write_text("changed-after-authorization\n")
    with pytest.raises(PermissionError, match="public hashes differ"):
        confirmation._validate_score_authorization(rna_private, adt_private)


def test_stage_order_and_score_authorization_bind_public_and_private_state(
    monkeypatch, tmp_path: Path
):
    assert confirmation._stage_prerequisites("source") == ()
    assert confirmation._stage_prerequisites("rna") == (
        (
            confirmation.COMPLETION_TAGS["source"],
            (confirmation.SOURCE_RESULT, confirmation.JOURNALS["source"]),
        ),
    )
    assert confirmation._stage_prerequisites("prediction") == (
        (
            confirmation.COMPLETION_TAGS["source"],
            (confirmation.SOURCE_RESULT, confirmation.JOURNALS["source"]),
        ),
        (
            confirmation.COMPLETION_TAGS["rna"],
            (confirmation.RNA_RESULT, confirmation.JOURNALS["rna"]),
        ),
    )
    assert confirmation._stage_prerequisites("adt") == (
        (
            confirmation.COMPLETION_TAGS["source"],
            (confirmation.SOURCE_RESULT, confirmation.JOURNALS["source"]),
        ),
        (
            confirmation.COMPLETION_TAGS["rna"],
            (confirmation.RNA_RESULT, confirmation.JOURNALS["rna"]),
        ),
        (
            confirmation.COMPLETION_TAGS["prediction"],
            (confirmation.PREDICTION_RESULT,),
        ),
    )
    assert confirmation._stage_prerequisites("score") == (
        (
            confirmation.COMPLETION_TAGS["source"],
            (confirmation.SOURCE_RESULT, confirmation.JOURNALS["source"]),
        ),
        (
            confirmation.COMPLETION_TAGS["rna"],
            (confirmation.RNA_RESULT, confirmation.JOURNALS["rna"]),
        ),
        (
            confirmation.COMPLETION_TAGS["prediction"],
            (confirmation.PREDICTION_RESULT,),
        ),
        (
            confirmation.COMPLETION_TAGS["adt"],
            (confirmation.ADT_RESULT, confirmation.JOURNALS["adt"]),
        ),
        (confirmation.SCORE_AUTHORIZATION_TAG, (confirmation.SCORE_AUTHORIZATION,)),
    )

    tag_commits = {
        confirmation.PROTOCOL_TAG: "protocol-commit",
        confirmation.COMPLETION_TAGS["source"]: "source-commit",
        confirmation.COMPLETION_TAGS["rna"]: "rna-commit",
        confirmation.COMPLETION_TAGS["prediction"]: "prediction-commit",
        confirmation.COMPLETION_TAGS["adt"]: "adt-commit",
        confirmation.SCORE_AUTHORIZATION_TAG: "authorization-commit",
    }
    ancestry = []
    monkeypatch.setattr(
        confirmation, "_remote_tag_commit", lambda tag: tag_commits[tag]
    )
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda tag, paths: tag_commits[tag],
    )
    monkeypatch.setattr(
        confirmation, "_require_ancestor", lambda old, new: ancestry.append((old, new))
    )
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda path: {
            "status": {
                confirmation.SOURCE_RESULT: "SOURCE_PASS",
                confirmation.RNA_RESULT: "HELD_RNA_PASS",
                confirmation.PREDICTION_RESULT: "PREDICTIONS_FROZEN",
                confirmation.ADT_RESULT: "HELD_ADT_PASS",
                confirmation.SCORE_AUTHORIZATION: "SCORE_AUTHORIZED",
            }[path]
        },
    )
    assert confirmation._require_stage_prerequisites("score") == (
        "source-commit",
        "rna-commit",
        "prediction-commit",
        "adt-commit",
        "authorization-commit",
    )
    assert ancestry == [
        ("protocol-commit", "source-commit"),
        ("source-commit", "rna-commit"),
        ("rna-commit", "prediction-commit"),
        ("prediction-commit", "adt-commit"),
        ("adt-commit", "authorization-commit"),
    ]

    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    paths = {
        "source": tmp_path / "source.json",
        "source_journal": tmp_path / "source.jsonl",
        "rna": tmp_path / "rna.json",
        "rna_journal": tmp_path / "rna.jsonl",
        "prediction": tmp_path / "prediction.json",
        "adt": tmp_path / "adt.json",
        "adt_journal": tmp_path / "adt.jsonl",
        "authorization": tmp_path / "authorization.json",
    }
    paths["source"].write_text(json.dumps({"status": "SOURCE_PASS"}))
    paths["rna"].write_text(
        json.dumps({"status": "HELD_RNA_PASS", "private_state_sha256": "rna-private"})
    )
    paths["prediction"].write_text(json.dumps({"status": "PREDICTIONS_FROZEN"}))
    paths["adt"].write_text(
        json.dumps({"status": "HELD_ADT_PASS", "private_state_sha256": "adt-private"})
    )
    for key in ("source_journal", "rna_journal", "adt_journal"):
        paths[key].write_text("{}\n")
    monkeypatch.setattr(confirmation, "SOURCE_RESULT", paths["source"])
    monkeypatch.setattr(confirmation, "RNA_RESULT", paths["rna"])
    monkeypatch.setattr(confirmation, "PREDICTION_RESULT", paths["prediction"])
    monkeypatch.setattr(confirmation, "ADT_RESULT", paths["adt"])
    monkeypatch.setattr(confirmation, "SCORE_AUTHORIZATION", paths["authorization"])
    monkeypatch.setattr(
        confirmation,
        "JOURNALS",
        {
            "source": paths["source_journal"],
            "rna": paths["rna_journal"],
            "adt": paths["adt_journal"],
        },
    )
    ancestry.clear()
    authorization_commits = {
        confirmation.COMPLETION_TAGS["source"]: "source-commit",
        confirmation.COMPLETION_TAGS["rna"]: "rna-commit",
        confirmation.COMPLETION_TAGS["prediction"]: "prediction-commit",
        confirmation.COMPLETION_TAGS["adt"]: "adt-commit",
    }
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda tag, files: authorization_commits[tag],
    )
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda path: json.loads(Path(path).read_text()),
    )

    authorization = confirmation.authorize_score()
    assert ancestry == [
        ("source-commit", "rna-commit"),
        ("rna-commit", "prediction-commit"),
        ("prediction-commit", "adt-commit"),
    ]
    assert authorization["status"] == "SCORE_AUTHORIZED"
    assert authorization["private_state_sha256"] == {
        "rna": "rna-private",
        "adt": "adt-private",
    }
    assert set(authorization["public_artifacts"]) == {
        "source.json",
        "source.jsonl",
        "rna.json",
        "rna.jsonl",
        "prediction.json",
        "adt.json",
        "adt.jsonl",
    }
    assert json.loads(paths["authorization"].read_text()) == authorization

    refused = tmp_path / "refused-authorization.json"
    monkeypatch.setattr(confirmation, "SCORE_AUTHORIZATION", refused)
    paths["adt"].write_text(
        json.dumps(
            {"status": "COMPLETED_NEGATIVE_RESULT", "private_state_sha256": "adt-private"}
        )
    )
    with pytest.raises(PermissionError, match="score prerequisites did not pass"):
        confirmation.authorize_score()
    assert not refused.exists()
