from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_gse279451_sepsis as runner
from experiments import acquire_gse279451_nonheld as acquisition
from experiments import reduce_gse279451_sepsis as reducer


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "data/confirmation/gse279451_sepsis"


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _patch_runner_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    paths = {
        "PROTOCOL": "docs/protocol.md",
        "PREFLIGHT": "data/development/preflight.json",
        "DESIGNATION": "data/confirmation/designation.json",
        "FAMILY_POLICY": "data/confirmation/family.json",
        "SOURCE_TEMPLATE": "data/confirmation/source-template.json",
        "SOURCE_MANIFEST": "data/confirmation/source.json",
        "REDUCER": "experiments/reducer.py",
        "EVALUATOR": "experiments/evaluator.py",
        "REDUCED_DEVELOPMENT": "data/development/reduced.json",
        "DEVELOPMENT_ATTEMPT": "data/development/development-attempt.json",
        "EVALUATION_ATTEMPT": "data/development/evaluation-attempt.json",
        "DEVELOPMENT_RESULT": "results/development.json",
        "PREDICTION": "results/prediction.json",
        "AUTH_TEMPLATE": "data/confirmation/auth-template.json",
        "AUTHORIZATION": "data/confirmation/auth.json",
        "SCORE_ATTEMPT": "data/confirmation/attempt.json",
        "OUTPUT": "results/score.json",
        "REFUSAL": "results/refusal.json",
        "BMMC_TERMINAL": "results/bmmc-terminal.json",
        "HELD_MEMBER_DIR": "data/confirmation/held-work",
        "HELD_PREDICTION_DIR": "data/confirmation/held-predictions",
    }
    for name, relative in paths.items():
        monkeypatch.setattr(runner, name, tmp_path / relative)
    runner_file = tmp_path / "experiments/confirm_gse279451_sepsis.py"
    runner_file.parent.mkdir(parents=True, exist_ok=True)
    runner_file.write_text("# frozen runner\n")
    monkeypatch.setattr(runner, "__file__", str(runner_file))
    for path in (
        runner.PROTOCOL,
        runner.PREFLIGHT,
        runner.DESIGNATION,
        runner.FAMILY_POLICY,
        runner.SOURCE_TEMPLATE,
        runner.REDUCER,
        runner.EVALUATOR,
        runner.REDUCED_DEVELOPMENT,
        runner.DEVELOPMENT_ATTEMPT,
        runner.EVALUATION_ATTEMPT,
        runner.DEVELOPMENT_RESULT,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    _json(
        runner.AUTH_TEMPLATE,
        {
            "schema": "gse279451-sepsis-score-authorization/1.0",
            "status": "OUTCOME_ACCESS_DISABLED",
            "prediction_path": runner._relative(runner.PREDICTION),
            "prediction_sha256": None,
            "runner_sha256": None,
            "reducer_sha256": None,
            "protocol_sha256": None,
            "candidate_designation_sha256": None,
            "family_policy_sha256": None,
            "source_manifest_sha256": None,
            "development_result_sha256": None,
            **{key: None for key in runner._transitive_bindings()},
            "public_prediction_commit": None,
            "public_prediction_url": None,
        },
    )
    monkeypatch.setattr(
        runner,
        "SANGER_TERMINAL_ARTIFACTS",
        (tmp_path / "sanger-attempt.json", tmp_path / "sanger-score.json"),
    )
    monkeypatch.setattr(
        runner,
        "BMMC_REVIVAL_ARTIFACTS",
        (tmp_path / "bmmc-revival.json",),
    )


def _valid_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    _patch_runner_paths(tmp_path, monkeypatch)
    source = {"accession": "GSE279451", "status": "fixture"}
    _json(runner.SOURCE_MANIFEST, source)
    certificate = {
        "converged": True,
        "scaled_gradient_norm": 0.0,
        "gradient_tolerance": 1e-8,
        "schur_condition_number": 1.0,
        "theta_curvature_condition_number": 1.0,
    }
    methods = {}
    for name in runner.REQUIRED_METHODS:
        if name == "independence":
            methods[name] = {"kind": "independence"}
        elif name == "best_residual":
            methods[name] = {
                "kind": "classical_residual",
                "family": "pearson",
                "centered": False,
                "source_coordinate": [0.0] * 81,
                "sample_size_normalized": True,
                "normalization": "source/sqrt(n), recipient*sqrt(m)",
            }
        else:
            methods[name] = {
                "kind": "conditional_log_odds",
                "source_coordinate": [0.0] * 81,
                "numerical_certificate": certificate,
            }
    development = {
        "selection": {"fixture": "locked"},
        "gate": {"passes_all": True, "comparisons": {}},
        "frozen_source_model": {"methods": methods},
    }
    monkeypatch.setattr(runner, "_validated_source", lambda: source)
    monkeypatch.setattr(
        runner, "_validated_development", lambda source_hash, actual_source: development
    )
    prediction = {
        **runner._expected_prediction_semantics(
            development, runner._sha256(runner.SOURCE_MANIFEST), source
        ),
        "created_at_utc": "2026-08-28T00:00:00+00:00",
    }
    _json(runner.PREDICTION, prediction)
    commit = "b" * 40
    authorization = {
        "schema": "gse279451-sepsis-score-authorization/1.0",
        "status": "OUTCOME_ACCESS_AUTHORIZED",
        "prediction_path": runner._relative(runner.PREDICTION),
        "prediction_sha256": runner._sha256(runner.PREDICTION),
        "runner_sha256": runner._sha256(Path(runner.__file__)),
        "reducer_sha256": runner._sha256(runner.REDUCER),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "candidate_designation_sha256": runner._sha256(runner.DESIGNATION),
        "family_policy_sha256": runner._sha256(runner.FAMILY_POLICY),
        "source_manifest_sha256": runner._sha256(runner.SOURCE_MANIFEST),
        "development_result_sha256": runner._sha256(runner.DEVELOPMENT_RESULT),
        **runner._transitive_bindings(),
        "public_prediction_commit": commit,
        "public_prediction_url": (
            f"https://github.com/o/r/blob/{commit}/{runner._relative(runner.PREDICTION)}"
        ),
    }
    _json(runner.AUTHORIZATION, authorization)
    monkeypatch.setattr(runner, "_assert_family_available", lambda: None)
    frozen_bytes = runner.PREDICTION.read_bytes()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return frozen_bytes

    monkeypatch.setattr(
        runner.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    return development


def test_designation_locks_split_panel_budget_and_gates() -> None:
    designation = json.loads(
        (CONFIRMATION / "candidate_designation_v1.json").read_text()
    )
    design = designation["design"]
    assert len(design["development_accessions"]) == 19
    assert len(design["held_accessions"]) == 21
    assert not set(design["development_accessions"]) & set(design["held_accessions"])
    assert design["markers"] == list(reducer.MARKERS)
    assert design["ordered_entities"] == 81
    assert design["primary_cell_budget_per_donor"] == 1024
    assert "GSE279451-CELL-BUDGET-v1" in design["cell_selection_rule"]
    assert designation["gates"]["development"]["minimum_favorable_donors"] == 15
    assert designation["gates"]["held"]["minimum_favorable_donors"] == 16
    assert designation["gates"]["held"]["maximum_exact_one_sided_sign_flip_p"] == 0.025


def test_source_template_is_disabled_and_binds_all_member_sizes() -> None:
    source = json.loads((CONFIRMATION / "source_manifest_template_v1.json").read_text())
    assert source["status"] == "SOURCE_UNAVAILABLE"
    assert source["members"] == []
    assert len(source["donors"]) == 40
    assert sum(record["matrix_bytes"] for record in source["donors"]) == 2319077199
    assert source["raw_tar"]["bytes"] == 2334412800
    assert source["primary_cell_budget"]["cells_per_donor"] == 1024
    assert source["primary_cell_budget"]["selection_uses_matrix_values"] is False
    assert source["access_audit"]["matrix_entries_decoded_before_template_freeze"] == 0
    assert sum(record["barcode_cells"] for record in source["donors"]) == 330112
    assert all(record["barcode_sha256"] for record in source["donors"])
    identity = source["donor_identity_contract"]
    assert identity["unique_adult_cite_seq_donor_ids"] == 40
    assert identity["donor_id_set_equals_geo_sample_name_set"] is True


def test_acquisition_plan_contains_all_axes_and_only_development_matrices() -> None:
    source = json.loads((CONFIRMATION / "source_manifest_template_v1.json").read_text())
    plan = acquisition._member_plan(source["donors"])
    keys = [(donor["accession"], kind) for donor, kind in plan]
    assert len(keys) == 99
    assert sum(kind == "matrix" for _, kind in keys) == 19
    assert all(
        donor["role"] == "development" for donor, kind in plan if kind == "matrix"
    )


def test_nonheld_manifest_rejects_even_a_bound_held_matrix_member() -> None:
    source = json.loads((CONFIRMATION / "source_manifest_template_v1.json").read_text())
    source["status"] = "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    source["members"] = [
        {"accession": accession, "kind": kind}
        for accession in (*reducer.DEVELOPMENT_DONORS, *reducer.HELD_DONORS)
        for kind in ("barcodes", "features")
    ] + [
        {"accession": accession, "kind": "matrix"}
        for accession in (*reducer.DEVELOPMENT_DONORS, reducer.HELD_DONORS[0])
    ]
    with pytest.raises(PermissionError, match="only development matrices"):
        reducer._validate_manifest_shape(source)


def test_active_manifest_cannot_mutate_frozen_sample_or_axis_contract() -> None:
    source = json.loads((CONFIRMATION / "source_manifest_template_v1.json").read_text())
    source["status"] = "NONHELD_SOURCE_ACCESS_AUTHORIZED"
    source["donors"][0]["sample"] = "MUTATED-SAMPLE"
    with pytest.raises(PermissionError, match="frozen source template"):
        reducer._validate_manifest_shape(source)


def test_budgeted_cell_axis_is_exact_deterministic_and_value_free() -> None:
    barcodes = [f"cell-{index:04d}" for index in range(1500)]
    selected, selected_barcodes = reducer._budgeted_cells(
        barcodes, "GSM8571042", "Abd-S111"
    )
    assert len(selected) == len(set(selected)) == 1024
    assert selected_barcodes == [barcodes[index] for index in selected]
    expected = sorted(
        range(len(barcodes)),
        key=lambda index: hashlib.sha256(
            (
                reducer.CELL_SELECTION_SALT
                + "GSM8571042"
                + "Abd-S111"
                + barcodes[index]
            ).encode()
        ).hexdigest(),
    )[:1024]
    assert selected == expected


def test_midrank_is_balanced_and_every_table_uses_1024_cells() -> None:
    barcodes = [f"cell-{index:04d}" for index in range(1024)]
    adt_counts = np.zeros((9, 1024), dtype=np.int64)
    adt = reducer._adt_states(adt_counts, barcodes, "GSM8571042")
    assert np.all(adt.sum(axis=1) == 512)
    rna = np.vstack([(np.arange(1024) + marker) % 3 == 0 for marker in range(9)])
    tables = reducer._ordered_tables(rna.astype(np.uint8), adt)
    assert tables.shape == (81, 2, 2)
    assert np.all(tables.sum(axis=(1, 2)) == 1024)


def test_destroyed_link_is_deterministic_and_preserves_every_margin() -> None:
    generator = np.random.default_rng(41)
    rna = generator.integers(0, 2, size=(9, 1024), dtype=np.uint8)
    adt = generator.integers(0, 2, size=(9, 1024), dtype=np.uint8)
    observed = reducer._ordered_tables(rna, adt)
    first = reducer._destroyed_adt(adt, "GSM8571043")
    second = reducer._destroyed_adt(adt, "GSM8571043")
    np.testing.assert_array_equal(first, second)
    destroyed = reducer._ordered_tables(rna, first)
    np.testing.assert_array_equal(observed.sum(axis=-1), destroyed.sum(axis=-1))
    np.testing.assert_array_equal(observed.sum(axis=-2), destroyed.sum(axis=-2))


def test_graph_profiles_are_marginal_and_adt_profile_is_not_constant() -> None:
    rna_counts = np.vstack(
        [np.arange(1024) % (marker + 2) == 0 for marker in range(9)]
    ).astype(np.int64)
    adt_counts = np.vstack(
        [(marker + 1) * (np.arange(1024) % (marker + 3)) for marker in range(9)]
    )
    rna = (rna_counts > 0).astype(np.uint8)
    panel_total = adt_counts.sum(axis=0, keepdims=True)
    adt_composition = np.divide(
        100.0 * adt_counts,
        panel_total,
        out=np.zeros_like(adt_counts, dtype=float),
        where=panel_total > 0,
    )
    rna_profile = rna.mean(axis=1)
    adt_profile = np.log1p(adt_composition).mean(axis=1)
    assert rna_profile.shape == adt_profile.shape == (9,)
    assert np.ptp(rna_profile) > 0
    assert np.ptp(adt_profile) > 0


def test_stream_reader_decodes_only_selected_rows_and_cells(tmp_path: Path) -> None:
    matrix = tmp_path / "fixture.mtx.gz"
    with gzip.open(matrix, "wt") as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n")
        handle.write("% synthetic poison fixture\n")
        handle.write("4 5 5\n")
        handle.write("1 1 3\n")
        handle.write("2 2 4\n")
        handle.write("3 3 5\n")
        handle.write("4 4 6\n")
        handle.write("1 5 7\n")
    values = reducer._stream_selected_rows(
        matrix,
        [0, 3],
        [4, 0],
        expected_features=4,
        expected_cells=5,
    )
    np.testing.assert_array_equal(values, np.array([[7, 3], [0, 0]]))


def test_held_accession_poison_refuses_before_member_lookup() -> None:
    class PoisonMembers:
        def __iter__(self):
            raise AssertionError("held source members were inspected")

    source = {"members": PoisonMembers()}
    with pytest.raises(PermissionError, match="forbids every held accession"):
        reducer._validated_member(source, "GSM8571042", "matrix", phase="development")


def test_held_score_phase_without_permit_refuses_before_member_lookup() -> None:
    class PoisonMembers:
        def __iter__(self):
            raise AssertionError("held source members were inspected")

    source = {"members": PoisonMembers()}
    with pytest.raises(PermissionError, match="public prediction authorization"):
        reducer._validated_member(
            source, "GSM8571042", "matrix", phase="held_score_authorized"
        )


def test_held_truth_requires_materialized_predictions_before_axes() -> None:
    class PoisonDonors:
        def __iter__(self):
            raise AssertionError("held axes were inspected")

    with pytest.raises(PermissionError, match="materialized prediction binding"):
        reducer.reduce_donor(
            {"donors": PoisonDonors()},
            "GSM8571042",
            phase="held_score_authorized",
        )


def test_path_traversal_and_symlink_escape_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reducer, "ROOT", tmp_path / "repo")
    reducer.ROOT.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("poison\n")
    with pytest.raises(PermissionError, match="repository-relative"):
        reducer._bound_path("../outside", "poison")
    link = reducer.ROOT / "link"
    link.symlink_to(outside)
    with pytest.raises(PermissionError, match="escapes the repository"):
        reducer._bound_path("link", "poison")


def test_wrong_member_hash_refuses_before_gzip_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reducer, "ROOT", tmp_path)
    path = tmp_path / "data/member.gz"
    path.parent.mkdir()
    path.write_bytes(b"not a matrix")
    source = {
        "members": [
            {
                "accession": "GSM8571043",
                "kind": "matrix",
                "local_path": "data/member.gz",
                "bytes": path.stat().st_size,
                "sha256": "0" * 64,
                "retained": True,
            }
        ]
    }
    with pytest.raises(PermissionError, match="SHA-256 differs"):
        reducer._validated_member(source, "GSM8571043", "matrix", phase="development")


def test_acquisition_family_and_artifact_checks_precede_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    monkeypatch.setattr(acquisition, "OUTPUT", tmp_path / "source.json")
    monkeypatch.setattr(acquisition, "REDUCED_OUTPUT", tmp_path / "reduced.json")
    monkeypatch.setattr(acquisition, "DEVELOPMENT_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(acquisition, "DEVELOPMENT_REFUSAL", tmp_path / "refusal.json")
    monkeypatch.setattr(acquisition, "EVALUATOR", tmp_path / "missing-evaluator.py")
    monkeypatch.setattr(
        acquisition,
        "_download",
        lambda *args: (_ for _ in ()).throw(AssertionError("network opened")),
    )
    monkeypatch.setattr(
        acquisition,
        "_assert_family_available",
        lambda: (_ for _ in ()).throw(PermissionError("family closed")),
    )
    with pytest.raises(PermissionError, match="family closed"):
        acquisition.acquire()
    monkeypatch.setattr(acquisition, "_assert_family_available", lambda: None)
    with pytest.raises(FileNotFoundError, match="before network access"):
        acquisition.acquire()


def test_development_attempt_precedes_first_matrix_and_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    monkeypatch.setattr(acquisition, "OUTPUT", tmp_path / "source.json")
    monkeypatch.setattr(acquisition, "REDUCED_OUTPUT", tmp_path / "reduced.json")
    monkeypatch.setattr(acquisition, "DEVELOPMENT_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(acquisition, "DEVELOPMENT_REFUSAL", tmp_path / "refusal.json")
    monkeypatch.setattr(acquisition, "MEMBER_DIR", tmp_path / "members")
    monkeypatch.setattr(acquisition, "_assert_family_available", lambda: None)
    monkeypatch.setattr(acquisition, "_artifact_bindings", lambda: {"test": "a" * 64})
    monkeypatch.setattr(
        acquisition.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 10**12})(),
    )

    def fake_download(url, destination, expected_bytes, expected_sha256):
        del url, expected_sha256
        if destination.name.endswith("matrix.mtx.gz"):
            assert acquisition.DEVELOPMENT_ATTEMPT.is_file()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"matrix poison")
        return expected_bytes, "b" * 64

    monkeypatch.setattr(acquisition, "_download", fake_download)
    monkeypatch.setattr(
        reducer,
        "reduce_donor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("support poison")),
    )
    with pytest.raises(ValueError, match="support poison"):
        acquisition.acquire()
    assert acquisition.DEVELOPMENT_ATTEMPT.is_file()
    refusal = json.loads(acquisition.DEVELOPMENT_REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL"
    assert refusal["rerun_permitted"] is False
    assert not list(acquisition.MEMBER_DIR.glob("*.matrix.mtx.gz"))
    with pytest.raises(FileExistsError, match="development acquisition artifact"):
        acquisition.acquire()


def test_strict_json_and_certificates_reject_nonfinite_before_write(
    tmp_path: Path,
) -> None:
    poison = tmp_path / "poison.json"
    poison.write_text('{"value": NaN}\n')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        runner._read_json(poison)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError):
        runner._write_json_exclusive(output, {"value": float("nan")})
    assert not output.exists()
    certificate = {
        "converged": True,
        "scaled_gradient_norm": float("nan"),
        "gradient_tolerance": 1e-8,
        "schur_condition_number": 1.0,
        "theta_curvature_condition_number": 1.0,
    }
    with pytest.raises(ValueError, match="certificate"):
        runner._validate_certificate(certificate, "poison")


def test_authorization_template_is_validated_before_prediction_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner_paths(tmp_path, monkeypatch)
    runner.AUTH_TEMPLATE.unlink()
    monkeypatch.setattr(runner, "_assert_family_available", lambda: None)
    monkeypatch.setattr(
        runner,
        "_validated_source",
        lambda: (_ for _ in ()).throw(AssertionError("source inspected")),
    )
    with pytest.raises(PermissionError, match="authorization template"):
        runner.predict()
    assert not runner.PREDICTION.exists()


def test_missing_source_refuses_prediction_before_development_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner_paths(tmp_path, monkeypatch)
    runner.SOURCE_MANIFEST.unlink(missing_ok=True)
    monkeypatch.setattr(runner, "_assert_family_available", lambda: None)
    monkeypatch.setattr(
        runner,
        "_validated_development",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("development result was accessed")
        ),
    )
    with pytest.raises(PermissionError, match="source manifest is absent"):
        runner.predict()
    assert not runner.PREDICTION.exists()


def _development_binding_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict]:
    _patch_runner_paths(tmp_path, monkeypatch)
    source = {
        "bindings": {"fixture": "bound"},
        "members": [
            {
                "accession": accession,
                "kind": "matrix",
                "sha256": f"{index + 1:064x}",
            }
            for index, accession in enumerate(runner.DEVELOPMENT_DONORS)
        ],
    }
    _json(runner.SOURCE_MANIFEST, source)
    attempt = {
        "status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
        "source_template_sha256": runner._sha256(runner.SOURCE_TEMPLATE),
        "artifact_bindings": source["bindings"],
        "axis_members_sha256": runner._json_sha256([]),
    }
    _json(runner.DEVELOPMENT_ATTEMPT, attempt)
    reduced = {
        "status": "NONHELD_REDUCTION_COMPLETE",
        "development_attempt_sha256": runner._sha256(runner.DEVELOPMENT_ATTEMPT),
        "source_manifest_sha256": runner._sha256(runner.SOURCE_MANIFEST),
        "development_donors": list(runner.DEVELOPMENT_DONORS),
        "held_donors": list(runner.HELD_DONORS),
        "primary_cells_per_donor": 1024,
        "all_cells_sensitivity_included": False,
        "access_audit": {"held_matrix_members_opened": 0},
        "donors": [
            {
                "accession": accession,
                "matrix_sha256": source["members"][index]["sha256"],
            }
            for index, accession in enumerate(runner.DEVELOPMENT_DONORS)
        ],
    }
    _json(runner.REDUCED_DEVELOPMENT, reduced)
    evaluation = {
        "status": "TERMINAL_DEVELOPMENT_EVALUATION_STARTED",
        "reduced_development_sha256": runner._sha256(runner.REDUCED_DEVELOPMENT),
        "development_attempt_sha256": runner._sha256(runner.DEVELOPMENT_ATTEMPT),
        "evaluator_sha256": runner._sha256(runner.EVALUATOR),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "candidate_designation_sha256": runner._sha256(runner.DESIGNATION),
        "family_policy_sha256": runner._sha256(runner.FAMILY_POLICY),
        "transitive_bindings": runner._transitive_bindings(),
    }
    _json(runner.EVALUATION_ATTEMPT, evaluation)
    _json(runner.DEVELOPMENT_RESULT, {})
    monkeypatch.setattr(runner, "_validate_reduced_donors", lambda reduced: None)
    return source, reduced


def test_altered_evaluation_attempt_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _development_binding_fixture(tmp_path, monkeypatch)
    evaluation = json.loads(runner.EVALUATION_ATTEMPT.read_text())
    evaluation["evaluator_sha256"] = "0" * 64
    _json(runner.EVALUATION_ATTEMPT, evaluation)
    with pytest.raises(PermissionError, match="access seal"):
        runner._validated_development(runner._sha256(runner.SOURCE_MANIFEST), source)


def test_reduced_matrix_hash_must_match_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, reduced = _development_binding_fixture(tmp_path, monkeypatch)
    reduced["donors"][0]["matrix_sha256"] = "f" * 64
    _json(runner.REDUCED_DEVELOPMENT, reduced)
    evaluation = json.loads(runner.EVALUATION_ATTEMPT.read_text())
    evaluation["reduced_development_sha256"] = runner._sha256(
        runner.REDUCED_DEVELOPMENT
    )
    _json(runner.EVALUATION_ATTEMPT, evaluation)
    with pytest.raises(PermissionError, match="matrix hash differs"):
        runner._validated_development(runner._sha256(runner.SOURCE_MANIFEST), source)


def test_sanger_or_bmmc_revival_poison_closes_family_before_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner_paths(tmp_path, monkeypatch)
    runner.SANGER_TERMINAL_ARTIFACTS[0].write_text("attempt\n")
    with pytest.raises(PermissionError, match="Sanger terminal artifact"):
        runner._assert_family_available()
    runner.SANGER_TERMINAL_ARTIFACTS[0].unlink()
    runner.BMMC_REVIVAL_ARTIFACTS[0].write_text("revival\n")
    with pytest.raises(PermissionError, match="BMMC revival artifact"):
        runner._assert_family_available()


def test_bad_public_authorization_refuses_before_held_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _valid_authorization(tmp_path, monkeypatch)
    authorization = json.loads(runner.AUTHORIZATION.read_text())
    authorization["prediction_sha256"] = "0" * 64
    _json(runner.AUTHORIZATION, authorization)
    monkeypatch.setattr(
        runner,
        "_score_held_once",
        lambda *args: (_ for _ in ()).throw(AssertionError("held engine ran")),
    )
    with pytest.raises(PermissionError, match="prediction_sha256"):
        runner.score()
    assert not runner.SCORE_ATTEMPT.exists()


@pytest.mark.parametrize("poison", ["model", "bindings", "transitive"])
def test_authorized_fabricated_prediction_refuses_before_held_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str
) -> None:
    _valid_authorization(tmp_path, monkeypatch)
    prediction = json.loads(runner.PREDICTION.read_text())
    if poison == "model":
        prediction["frozen_source_model"]["methods"]["primary"]["source_coordinate"][
            0
        ] = 0.25
    elif poison == "bindings":
        prediction["bindings"]["protocol_sha256"] = "0" * 64
    else:
        prediction["bindings"]["coupling_fields_sha256"] = "0" * 64
    _json(runner.PREDICTION, prediction)
    authorization = json.loads(runner.AUTHORIZATION.read_text())
    authorization["prediction_sha256"] = runner._sha256(runner.PREDICTION)
    _json(runner.AUTHORIZATION, authorization)
    with pytest.raises(PermissionError, match="prediction .* differs"):
        runner.score()
    assert not runner.SCORE_ATTEMPT.exists()


@pytest.mark.parametrize("failure", ["mismatch", "network"])
def test_remote_prediction_failure_refuses_before_held_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _valid_authorization(tmp_path, monkeypatch)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not the committed prediction"

    if failure == "mismatch":
        monkeypatch.setattr(
            runner.urllib.request, "urlopen", lambda *args, **kwargs: Response()
        )
        match = "bytes differ"
    else:
        monkeypatch.setattr(
            runner.urllib.request,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
        )
        match = "fetch failed"
    with pytest.raises(PermissionError, match=match):
        runner.score()
    assert not runner.SCORE_ATTEMPT.exists()


def test_terminal_attempt_precedes_held_engine_and_retains_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _valid_authorization(tmp_path, monkeypatch)

    def poison_after_marker(*args):
        assert runner.SCORE_ATTEMPT.is_file()
        assert not runner.REFUSAL.exists()
        raise RuntimeError("deliberate post-marker stop")

    monkeypatch.setattr(runner, "_score_held_once", poison_after_marker)
    with pytest.raises(RuntimeError, match="post-marker"):
        runner.score()
    assert runner.SCORE_ATTEMPT.is_file()
    refusal = json.loads(runner.REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_SCORE_REFUSAL"
    assert refusal["error_type"] == "RuntimeError"
    assert refusal["sanitized_error_message"] == "deliberate post-marker stop"
    assert refusal["partial_audit"]["held_donors_completed"] == []
    assert refusal["partial_audit"]["current_source_matrix_deleted"] is True


def test_held_scorer_materializes_predictions_before_truth_and_deletes_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner_paths(tmp_path, monkeypatch)
    accession = "GSM8571042"
    sample = "Abd-S111"
    monkeypatch.setattr(runner, "HELD_DONORS", (accession,))
    monkeypatch.setattr(
        runner,
        "HELD_MEMBER_DIR",
        tmp_path / "data/confirmation/held-work",
    )
    monkeypatch.setattr(
        runner,
        "HELD_PREDICTION_DIR",
        tmp_path / "data/confirmation/held-predictions",
    )
    donor = {
        "accession": accession,
        "sample": sample,
        "matrix_bytes": 7,
        "barcode_sha256": "a" * 64,
        "feature_sha256": "b" * 64,
    }
    _json(runner.SOURCE_TEMPLATE, {"donors": [donor]})
    _json(runner.AUTHORIZATION, {"status": "fixture"})
    _json(runner.SOURCE_MANIFEST, {"status": "fixture"})
    monkeypatch.setattr(
        runner,
        "_validated_source",
        lambda: {"donors": [donor], "members": []},
    )
    matrix_path = runner.HELD_MEMBER_DIR / f"{accession}_{sample}.matrix.mtx.gz"

    def fake_download(url: str, destination: Path, expected_bytes: int) -> str:
        del url
        assert runner.SCORE_ATTEMPT.is_file()
        assert expected_bytes == 7
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return hashlib.sha256(b"fixture").hexdigest()

    monkeypatch.setattr(runner, "_download_held_matrix", fake_download)
    barcodes = [f"cell-{index}" for index in range(1024)]
    monkeypatch.setattr(
        reducer,
        "_axes",
        lambda *args, **kwargs: (barcodes, [("id", "name", "kind")]),
    )
    monkeypatch.setattr(
        reducer,
        "_marker_rows",
        lambda features: {"rna": list(range(9)), "adt": list(range(9, 18))},
    )
    monkeypatch.setattr(
        reducer, "_validated_member", lambda *args, **kwargs: matrix_path
    )
    rna = np.vstack([(np.arange(1024) + index) % 2 for index in range(9)])
    adt_counts = np.zeros((9, 1024), dtype=np.int64)
    calls = []

    def fake_stream(
        path: Path, selected_rows: list[int], *args, **kwargs
    ) -> np.ndarray:
        del path, args, kwargs
        calls.append(len(selected_rows))
        if len(calls) == 1:
            return rna
        if len(calls) == 2:
            return adt_counts
        materialized = runner.HELD_PREDICTION_DIR / f"{accession}.json"
        assert materialized.is_file()
        payload = json.loads(materialized.read_text())
        embedded = payload.pop("materialized_content_sha256")
        assert runner._json_sha256(payload) == embedded
        return np.vstack((rna, adt_counts))

    monkeypatch.setattr(reducer, "_stream_selected_rows", fake_stream)

    def fake_predictions(methods, margins):
        del methods
        values = np.asarray(margins, dtype=float)
        rows = values.sum(axis=-1)
        columns = values.sum(axis=-2)
        independence = rows[:, :, None] * columns[:, None, :] / 1024
        return {name: independence.copy() for name in runner.REQUIRED_METHODS}

    monkeypatch.setattr(runner, "_predict_from_margins", fake_predictions)
    _json(
        runner.SCORE_ATTEMPT,
        {
            "status": "TERMINAL_ATTEMPT_STARTED",
            "prediction_sha256": "c" * 64,
            "public_prediction_commit": "d" * 40,
        },
    )
    prediction = {
        "frozen_source_model": {
            "methods": {name: {} for name in runner.REQUIRED_METHODS}
        }
    }
    result = runner._score_held_once(
        prediction,
        runner._ScorePermit(
            "c" * 64,
            "d" * 40,
            "e" * 64,
            "https://github.com/o/r/blob/" + "d" * 40 + "/results/prediction.json",
            "c" * 64,
        ),
    )
    assert calls == [9, 9, 18]
    assert result["access_audit"]["predictions_hashed_before_truth_per_donor"] is True
    assert runner._LAST_HELD_AUDIT["held_donors_completed"] == [accession]
    assert len(runner._LAST_HELD_AUDIT["matrix_members_hashed"]) == 1
    assert len(runner._LAST_HELD_AUDIT["prediction_materializations"]) == 1
    assert runner._LAST_HELD_AUDIT["current_source_matrix_deleted"] is True
    donor_result = result["donor_results"][0]
    assert donor_result["informative_entity_mask"] == [True] * 81
    assert donor_result["excluded_entities"] == []
    assert not matrix_path.exists()


def test_exact_sign_flip_inclusive_tail_preserves_zero_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "HELD_DONORS", ("a", "b", "c"))
    assert runner._exact_sign_flip_p(np.zeros(3)) == 1.0
    assert runner._exact_sign_flip_p(-np.ones(3)) == 1 / 8


def test_entity_support_report_binds_mask_and_ordered_exclusions() -> None:
    informative = np.ones(81, dtype=bool)
    informative[[1, 9, 80]] = False
    report = runner._entity_support_report(informative)
    assert report["informative_entities"] == 78
    assert report["informative_entity_mask"] == informative.tolist()
    assert report["excluded_entities"] == [
        {"entity_index": 1, "rna_marker": "CD4", "adt_marker": "CD7"},
        {"entity_index": 9, "rna_marker": "CD7", "adt_marker": "CD4"},
        {"entity_index": 80, "rna_marker": "CD52", "adt_marker": "CD52"},
    ]


def test_refusal_message_redacts_local_paths_and_email() -> None:
    message = runner._sanitized_error_message(
        RuntimeError("/Users/private/person/file user@example.org")
    )
    assert "/Users/" not in message
    assert "user@example.org" not in message
    assert "<local-path>" in message
    assert "<email>" in message


def test_protocol_locks_lodo_residual_gates_family_and_secondary_depth() -> None:
    protocol = (
        ROOT / "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
    ).read_text()
    normalized = " ".join(protocol.split())
    assert "19-fold leave-one-development-donor-out" in normalized
    assert "divided by `sqrt(n)`" in normalized
    assert "at least 15 of 19" in normalized
    assert "at least 16 of 21" in normalized
    assert "exact one-sided sign-flip `p <= 0.025`" in normalized
    assert "GSE279451-CELL-BUDGET-v1" in normalized
    assert "all-deposited-cell analysis" in normalized
    assert "BMMC is closed and cannot be" in normalized
    assert "Sanger" in normalized
