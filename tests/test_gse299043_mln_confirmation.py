from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from experiments import confirm_gse299043_mln as runner
from experiments import reduce_gse299043_mln as reducer


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _model() -> dict[str, object]:
    certificate = {
        "converged": True,
        "scaled_gradient_norm": 0.0,
        "gradient_tolerance": 1e-8,
        "schur_condition_number": 1.0,
        "theta_curvature_condition_number": 1.0,
    }
    methods: dict[str, object] = {
        name: {
            "kind": "conditional_log_odds",
            "source_coordinate": [0.0] * 81,
            "numerical_certificate": certificate,
        }
        for name in (
            "primary",
            "destroyed_link",
            "hierarchical_ridge_only",
        )
    }
    methods["best_residual"] = {
        "kind": "classical_residual",
        "family": "deviance",
        "centered": True,
        "source_coordinate": [0.0] * 81,
        "sample_size_normalized": True,
        "normalization": "source/sqrt(n), recipient*sqrt(m)",
        "donor_equal_pooling": True,
        "target_margin_inversion": True,
        "target_null_restored": True,
    }
    methods["independence"] = {"kind": "independence"}
    return {"methods": methods}


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    paths = {
        "PROTOCOL": "docs/protocol.md",
        "DESIGNATION": "data/confirmation/designation.json",
        "FAMILY_POLICY": "data/confirmation/family.json",
        "SOURCE_TEMPLATE": "data/confirmation/source-template.json",
        "SOURCE_MANIFEST": "data/confirmation/source.json",
        "REDUCER": "experiments/reducer.py",
        "EVALUATOR": "experiments/evaluator.py",
        "DEVELOPMENT_ATTEMPT": "data/development/development-attempt.json",
        "EVALUATION_ATTEMPT": "data/development/evaluation-attempt.json",
        "REDUCED_DEVELOPMENT": "data/development/reduced.json",
        "DEVELOPMENT_RESULT": "results/development.json",
        "DEVELOPMENT_REFUSAL": "results/development-refusal.json",
        "DEVELOPMENT_ACQUISITION_REFUSAL": "results/acquisition-refusal.json",
        "PREDICTION": "results/prediction.json",
        "AUTH_TEMPLATE": "data/confirmation/auth-template.json",
        "AUTHORIZATION": "data/confirmation/auth.json",
        "AUTH_PUBLICATION_TEMPLATE": "data/confirmation/auth-publication-template.json",
        "AUTH_PUBLICATION": "data/confirmation/auth-publication.json",
        "SCORE_ATTEMPT": "data/confirmation/score-attempt.json",
        "OUTPUT": "results/confirmation.json",
        "REFUSAL": "results/refusal.json",
        "HELD_MEMBER_DIR": "data/confirmation/held-work",
        "HELD_PREDICTION_DIR": "data/confirmation/held-predictions",
    }
    for name, relative in paths.items():
        monkeypatch.setattr(runner, name, tmp_path / relative)
    runner_file = tmp_path / "experiments/confirm_gse299043_mln.py"
    runner_file.parent.mkdir(parents=True, exist_ok=True)
    runner_file.write_text("# frozen runner\n")
    monkeypatch.setattr(runner, "__file__", str(runner_file))
    for path in (
        runner.PROTOCOL,
        runner.DESIGNATION,
        runner.FAMILY_POLICY,
        runner.SOURCE_TEMPLATE,
        runner.SOURCE_MANIFEST,
        runner.REDUCER,
        runner.EVALUATOR,
        runner.DEVELOPMENT_ATTEMPT,
        runner.EVALUATION_ATTEMPT,
        runner.REDUCED_DEVELOPMENT,
        runner.DEVELOPMENT_RESULT,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    transitive = {"numerical_core_sha256": "f" * 64}
    monkeypatch.setattr(runner, "_transitive_bindings", lambda: transitive)
    bound = {
        "prediction_sha256",
        "public_prediction_commit",
        "public_prediction_url",
        "runner_sha256",
        "reducer_sha256",
        "development_evaluator_sha256",
        "protocol_sha256",
        "candidate_designation_sha256",
        "family_policy_sha256",
        "authorization_publication_template_sha256",
        "source_manifest_sha256",
        "development_result_sha256",
        *transitive,
    }
    _json(
        runner.AUTH_TEMPLATE,
        {
            "schema": "gse299043-mln-score-authorization/1.0",
            "status": "OUTCOME_ACCESS_DISABLED",
            "prediction_path": runner._relative(runner.PREDICTION),
            **{key: None for key in bound},
        },
    )
    _json(
        runner.AUTH_PUBLICATION_TEMPLATE,
        {
            "schema": "gse299043-mln-score-authorization-publication/1.0",
            "status": "PUBLIC_AUTHORIZATION_UNAVAILABLE",
            "authorization_path": runner._relative(runner.AUTHORIZATION),
            "authorization_sha256": None,
            "public_authorization_commit": None,
            "public_authorization_url": None,
        },
    )


def _authorized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], runner._ScorePermit]:
    _patch_paths(tmp_path, monkeypatch)
    source = {
        "accession": "GSE299043",
        "members": [
            {"role": "development"},
            {"role": "held"},
        ],
    }
    development = {
        "selection": {"frozen": True},
        "gate": {"passes_all": True},
        "frozen_source_model": _model(),
    }
    monkeypatch.setattr(runner, "_validated_source", lambda: source)
    monkeypatch.setattr(
        runner, "_validated_development", lambda source_hash: development
    )
    prediction = {
        **runner._expected_prediction(
            development, runner._sha256(runner.SOURCE_MANIFEST), source
        ),
        "created_at_utc": "2026-08-28T00:00:00+00:00",
    }
    _json(runner.PREDICTION, prediction)
    commit = "b" * 40
    authorization = {
        "schema": "gse299043-mln-score-authorization/1.0",
        "status": "OUTCOME_ACCESS_AUTHORIZED",
        "prediction_path": runner._relative(runner.PREDICTION),
        "prediction_sha256": runner._sha256(runner.PREDICTION),
        "public_prediction_commit": commit,
        "public_prediction_url": (
            "https://github.com/"
            f"{runner.PUBLIC_GITHUB_OWNER}/{runner.PUBLIC_GITHUB_REPOSITORY}/blob/"
            f"{commit}/{runner._relative(runner.PREDICTION)}"
        ),
        "runner_sha256": runner._sha256(Path(runner.__file__)),
        "reducer_sha256": runner._sha256(runner.REDUCER),
        "development_evaluator_sha256": runner._sha256(runner.EVALUATOR),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "candidate_designation_sha256": runner._sha256(runner.DESIGNATION),
        "family_policy_sha256": runner._sha256(runner.FAMILY_POLICY),
        "authorization_publication_template_sha256": runner._sha256(
            runner.AUTH_PUBLICATION_TEMPLATE
        ),
        "source_manifest_sha256": runner._sha256(runner.SOURCE_MANIFEST),
        "development_result_sha256": runner._sha256(runner.DEVELOPMENT_RESULT),
        **runner._transitive_bindings(),
    }
    _json(runner.AUTHORIZATION, authorization)
    authorization_commit = "c" * 40
    _json(
        runner.AUTH_PUBLICATION,
        {
            "schema": "gse299043-mln-score-authorization-publication/1.0",
            "status": "PUBLIC_AUTHORIZATION_AVAILABLE",
            "authorization_path": runner._relative(runner.AUTHORIZATION),
            "authorization_sha256": runner._sha256(runner.AUTHORIZATION),
            "public_authorization_commit": authorization_commit,
            "public_authorization_url": (
                "https://github.com/"
                f"{runner.PUBLIC_GITHUB_OWNER}/{runner.PUBLIC_GITHUB_REPOSITORY}/blob/"
                f"{authorization_commit}/{runner._relative(runner.AUTHORIZATION)}"
            ),
        },
    )
    prediction_bytes = runner.PREDICTION.read_bytes()
    authorization_bytes = runner.AUTHORIZATION.read_bytes()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def __init__(self, payload: bytes):
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

    def urlopen(request, **kwargs):
        del kwargs
        payload = (
            authorization_bytes
            if request.full_url.endswith(runner._relative(runner.AUTHORIZATION))
            else prediction_bytes
        )
        return Response(payload)

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    validated, permit = runner._validated_authorization()
    assert validated == prediction
    return prediction, permit


def _held_permit() -> reducer.HeldAccessPermit:
    return reducer.HeldAccessPermit(
        prediction_sha256="a" * 64,
        public_commit="b" * 40,
        authorization_sha256="c" * 64,
        terminal_attempt_sha256="d" * 64,
    )


def _held_member(donor: str = "D512") -> dict[str, object]:
    filename = f"GSE299043_{donor}_001.CZINY-0161.v2.h5ad"
    return {
        "donor": donor,
        "role": "held",
        "filename": filename,
        "gex_library": "CZINY-0161",
        "url": f"https://ftp.ncbi.nlm.nih.gov/example/{filename}",
        "bytes": 7,
        "sha256": None,
        "local_path": None,
        "retained": False,
    }


def test_strict_json_and_exclusive_write_reject_nonfinite(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text('{"value": NaN}\n')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        runner._read_json(source)
    source.write_text('{"status": "first", "status": "second"}\n')
    with pytest.raises(ValueError, match="duplicate key 'status'"):
        runner._read_json(source)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError):
        runner._write_json_exclusive(output, {"value": float("nan")})
    assert not output.exists()
    runner._write_json_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(output, {"value": 2})


def test_exclusive_write_fsyncs_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(runner.os, "fsync", fsync_calls.append)
    runner._write_json_exclusive(tmp_path / "durable.json", {"value": 1})
    assert len(fsync_calls) == 2


def test_classical_residual_is_a_required_matched_head_to_head() -> None:
    methods = _model()["methods"]
    runner._validate_method_set(methods)
    methods["best_residual"]["sample_size_normalized"] = False
    with pytest.raises(ValueError, match="matched head-to-head"):
        runner._validate_method_set(methods)


def test_public_authorization_templates_and_designation_are_bound() -> None:
    assert runner.PUBLIC_GITHUB_OWNER == "sushaan-k"
    assert runner.PUBLIC_GITHUB_REPOSITORY == "coupling-fields-benchmark"
    confirmation = ROOT / "data/confirmation/gse299043_mln"
    publication_template = json.loads(
        (confirmation / "score_authorization_publication_template_v1.json").read_text()
    )
    authorization_template = json.loads(
        (confirmation / "score_authorization_template_v1.json").read_text()
    )
    designation = json.loads(
        (confirmation / "candidate_designation_v1.json").read_text()
    )
    assert publication_template == {
        "schema": "gse299043-mln-score-authorization-publication/1.0",
        "status": "PUBLIC_AUTHORIZATION_UNAVAILABLE",
        "authorization_path": (
            "data/confirmation/gse299043_mln/score_authorization_v1.json"
        ),
        "authorization_sha256": None,
        "public_authorization_commit": None,
        "public_authorization_url": None,
    }
    assert authorization_template["authorization_publication_template_sha256"] is None
    assert designation["authorization_publication_template"] == (
        "data/confirmation/gse299043_mln/"
        "score_authorization_publication_template_v1.json"
    )
    assert designation["authorization_publication"] == (
        "data/confirmation/gse299043_mln/score_authorization_publication_v1.json"
    )
    assert (
        designation["outcome_access"][
            "public_commit_bound_authorization_required_before_first_held_request"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("prediction_sha256", "0" * 64, "prediction_sha256"),
        ("public_prediction_commit", "main", "commit is not immutable"),
        (
            "public_prediction_url",
            "https://github.com/o/r/blob/main/results/prediction.json",
            "bound GitHub blob",
        ),
        (
            "public_prediction_url",
            "https://github.com/o/r/blob/" + "b" * 40 + "/results/prediction.json",
            "bound GitHub blob",
        ),
        ("numerical_core_sha256", "0" * 64, "numerical_core_sha256"),
    ],
)
def test_authorization_hash_commit_url_and_transitive_poison_refuse_before_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    match: str,
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)
    authorization = json.loads(runner.AUTHORIZATION.read_text())
    authorization[field] = value
    _json(runner.AUTHORIZATION, authorization)
    with pytest.raises(PermissionError, match=match):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


def test_remote_prediction_must_match_byte_for_byte_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"not the frozen prediction"

    monkeypatch.setattr(
        runner.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    with pytest.raises(PermissionError, match="bytes differ"):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("authorization_sha256", "0" * 64, "sidecar differs"),
        ("public_authorization_commit", "main", "sidecar differs"),
        (
            "public_authorization_url",
            "https://github.com/o/r/blob/main/data/confirmation/auth.json",
            "bound GitHub blob",
        ),
        (
            "public_authorization_url",
            "https://github.com/o/r/blob/" + "c" * 40 + "/data/confirmation/auth.json",
            "bound GitHub blob",
        ),
        ("status", "PUBLIC_AUTHORIZATION_UNAVAILABLE", "sidecar differs"),
    ],
)
def test_public_authorization_sidecar_poison_refuses_before_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    match: str,
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)
    publication = json.loads(runner.AUTH_PUBLICATION.read_text())
    publication[field] = value
    _json(runner.AUTH_PUBLICATION, publication)
    with pytest.raises(PermissionError, match=match):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


def test_public_authorization_must_be_byte_identical_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)
    prediction_bytes = runner.PREDICTION.read_bytes()

    class Response:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return self.payload

    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda request, **kwargs: Response(
            b"not the public authorization"
            if request.full_url.endswith(runner._relative(runner.AUTHORIZATION))
            else prediction_bytes
        ),
    )
    with pytest.raises(PermissionError, match="authorization bytes differ"):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


def test_public_authorization_fetch_failure_refuses_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)
    prediction_bytes = runner.PREDICTION.read_bytes()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return prediction_bytes

    def urlopen(request, **kwargs):
        del kwargs
        if request.full_url.endswith(runner._relative(runner.AUTHORIZATION)):
            raise OSError("offline")
        return Response()

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    with pytest.raises(PermissionError, match="authorization fetch failed"):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


def test_missing_public_authorization_sidecar_refuses_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_fixture(tmp_path, monkeypatch)
    runner.AUTH_PUBLICATION.unlink()
    with pytest.raises(PermissionError, match="publication sidecar is absent"):
        runner._validated_authorization()
    assert not runner.SCORE_ATTEMPT.exists()


def test_sidecar_and_attempt_swap_refuses_before_held_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, permit = _authorized_fixture(tmp_path, monkeypatch)
    _json(
        runner.SCORE_ATTEMPT,
        runner._score_attempt_payload(permit, "2026-08-28T00:00:01+00:00"),
    )
    publication = json.loads(runner.AUTH_PUBLICATION.read_text())
    replacement_commit = "d" * 40
    publication["public_authorization_commit"] = replacement_commit
    publication["public_authorization_url"] = (
        "https://github.com/"
        f"{runner.PUBLIC_GITHUB_OWNER}/{runner.PUBLIC_GITHUB_REPOSITORY}/blob/"
        f"{replacement_commit}/{runner._relative(runner.AUTHORIZATION)}"
    )
    _json(runner.AUTH_PUBLICATION, publication)
    attempt = json.loads(runner.SCORE_ATTEMPT.read_text())
    attempt["authorization_publication_sha256"] = runner._sha256(
        runner.AUTH_PUBLICATION
    )
    attempt["public_authorization_commit"] = replacement_commit
    attempt["public_authorization_url"] = publication["public_authorization_url"]
    _json(runner.SCORE_ATTEMPT, attempt)
    with pytest.raises(PermissionError, match="terminal score seal differs"):
        runner._validated_score_attempt(permit)


def test_live_permit_rejects_sidecar_mutation_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, permit = _authorized_fixture(tmp_path, monkeypatch)
    _json(
        runner.SCORE_ATTEMPT,
        runner._score_attempt_payload(permit, "2026-08-28T00:00:01+00:00"),
    )
    attempt_hash = runner._validated_score_attempt(permit)
    held_permit = reducer.HeldAccessPermit(
        prediction_sha256=permit.prediction_sha256,
        public_commit=permit.public_commit,
        authorization_sha256=permit.authorization_sha256,
        terminal_attempt_sha256=attempt_hash,
    )
    publication = json.loads(runner.AUTH_PUBLICATION.read_text())
    publication["public_authorization_commit"] = "d" * 40
    _json(runner.AUTH_PUBLICATION, publication)
    with pytest.raises(PermissionError, match="terminal seal"):
        runner._validate_live_held_permit("D512", held_permit)


def test_score_preauthorization_failure_cannot_reach_held_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_assert_family_available", lambda: None)
    monkeypatch.setattr(
        runner,
        "_validated_authorization",
        lambda: (_ for _ in ()).throw(PermissionError("authorization poison")),
    )
    monkeypatch.setattr(
        runner,
        "_score_held_once",
        lambda *args: (_ for _ in ()).throw(AssertionError("held engine ran")),
    )
    with pytest.raises(PermissionError, match="authorization poison"):
        runner.score()
    assert not runner.SCORE_ATTEMPT.exists()


def test_held_access_is_checked_before_source_member_iteration() -> None:
    class PoisonMembers:
        def __iter__(self):
            raise AssertionError("held member metadata was inspected")

    with pytest.raises(PermissionError, match="score authorization"):
        runner._held_members({"members": PoisonMembers()}, "D512", None)


@pytest.mark.parametrize("poison", ("role", "duplicate"))
def test_member_role_and_duplicate_member_poison_refuse_without_network(
    poison: str,
) -> None:
    member = _held_member()
    if poison == "role":
        member["role"] = "development"
        records = [member]
        match = "role or identity"
    else:
        records = [member, dict(member)]
        match = "role or identity"
    with pytest.raises(PermissionError, match=match):
        runner._held_members({"members": records}, "D512", _held_permit())


def test_terminal_attempt_must_precede_first_held_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    member = _held_member()
    destination = runner.HELD_MEMBER_DIR / member["filename"]
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network reached before attempt")
        ),
    )
    with pytest.raises(PermissionError, match="attempt must precede"):
        runner._download_held_member(member, destination, _held_permit())


def test_held_download_retries_transient_transport_and_deletes_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_validate_live_held_permit", lambda *args: None)
    content = b"fixture"
    calls = 0
    sleeps: list[int] = []

    class Response:
        headers = {"Content-Length": str(len(content))}

        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int) -> bytes:
            del size
            if self.fail:
                if not self.sent:
                    self.sent = True
                    return content[:3]
                raise ConnectionResetError("transient reset")
            if self.sent:
                return b""
            self.sent = True
            return content

    def urlopen(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return Response(calls < runner.DOWNLOAD_ATTEMPTS)

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    member = _held_member()
    destination = runner.HELD_MEMBER_DIR / member["filename"]
    digest = runner._download_held_member(member, destination, _held_permit())
    assert calls == runner.DOWNLOAD_ATTEMPTS == 3
    assert sleeps == [1, 2]
    assert digest == hashlib.sha256(content).hexdigest()
    assert destination.read_bytes() == content
    assert not destination.with_suffix(destination.suffix + ".part").exists()


def test_held_download_does_not_retry_byte_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_validate_live_held_permit", lambda *args: None)
    calls = 0

    class Response:
        headers = {"Content-Length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int) -> bytes:
            del size
            return b""

    def urlopen(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return Response()

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError(f"slept {seconds}")),
    )
    member = _held_member()
    destination = runner.HELD_MEMBER_DIR / member["filename"]
    with pytest.raises(PermissionError, match="remote byte count differs"):
        runner._download_held_member(member, destination, _held_permit())
    assert calls == 1
    assert not destination.exists()
    assert not destination.with_suffix(destination.suffix + ".part").exists()


def test_held_one_hto_member_refuses_before_count_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_validate_live_held_permit", lambda *args: None)
    member = _held_member()
    path = tmp_path / member["filename"]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.ones((1, 1), dtype=np.int64))
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray([b"cell-1"]))
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray([b"HTO-1"]))
        var.create_dataset("gene_ids", data=np.asarray([b"D512-MLN-1"]))
        var.create_dataset("feature_types", data=np.asarray([b"Antibody Capture"]))
    monkeypatch.setattr(
        reducer,
        "_matrix_columns",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("counts materialized before HTO refusal")
        ),
    )
    with pytest.raises(ValueError, match="frozen single-tissue exception"):
        runner._hto_candidates(path, member, "D512", _held_permit())


def test_member_is_deleted_when_a_sealed_pass_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    _json(runner.SCORE_ATTEMPT, {"status": "TERMINAL_ATTEMPT_STARTED"})
    member = _held_member()

    def download(
        record: dict[str, object],
        destination: Path,
        permit: reducer.HeldAccessPermit,
    ) -> str:
        del record, permit
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return hashlib.sha256(b"fixture").hexdigest()

    monkeypatch.setattr(runner, "_download_held_member", download)
    audit = {
        "current_member": None,
        "current_member_deleted": True,
        "member_content_sha256": {},
        "member_passes": [],
    }
    with pytest.raises(RuntimeError, match="reader poison"):
        runner._visit_member(
            member,
            "hto_census",
            audit,
            lambda path: (_ for _ in ()).throw(RuntimeError("reader poison")),
            _held_permit(),
        )
    assert audit["current_member_deleted"] is True
    assert not (runner.HELD_MEMBER_DIR / member["filename"]).exists()


@pytest.mark.parametrize("encoding", ("dense", "csr_matrix", "csc_matrix"))
def test_selected_matrix_reader_decodes_only_requested_cells_and_markers(
    tmp_path: Path, encoding: str
) -> None:
    values = np.arange(30, dtype=np.int64).reshape(5, 6)
    path = tmp_path / f"{encoding}.h5"
    with h5py.File(path, "w") as handle:
        if encoding == "dense":
            handle.create_dataset("X", data=values)
        else:
            encoded = (
                sparse.csr_matrix(values)
                if encoding == "csr_matrix"
                else sparse.csc_matrix(values)
            )
            group = handle.create_group("X")
            group.attrs["encoding-type"] = encoding
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=encoded.data)
            group.create_dataset("indices", data=encoded.indices)
            group.create_dataset("indptr", data=encoded.indptr)
    with h5py.File(path, "r") as handle:
        observed = runner._matrix_selected_values(
            handle["X"], [4, 1], [5, 0, 3], values.shape
        )
    np.testing.assert_array_equal(observed, values[[4, 1]][:, [5, 0, 3]])


def test_truth_refuses_before_any_cell_read_when_predictions_are_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "HELD_DONORS", ("D512",))
    monkeypatch.setattr(
        runner,
        "_marker_pass",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("paired truth read before prediction")
        ),
    )
    with pytest.raises(PermissionError, match="predictions must be materialized"):
        runner._truth_and_losses(
            "D512",
            [],
            [],
            "a" * 64,
            np.tile([256, 256], (81, 1)),
            np.tile([256, 256], (81, 1)),
            "b" * 64,
            tuple(_model()["methods"]),
            {},
            _held_permit(),
        )


def test_score_engine_materializes_every_donor_before_first_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    donors = ("D512", "D520")
    monkeypatch.setattr(runner, "HELD_DONORS", donors)
    _json(runner.SCORE_ATTEMPT, {"status": "TERMINAL_ATTEMPT_STARTED"})
    monkeypatch.setattr(
        runner,
        "_validated_score_attempt",
        lambda permit: runner._sha256(runner.SCORE_ATTEMPT),
    )
    _json(runner.AUTHORIZATION, {"status": "fixture"})
    _json(runner.AUTH_PUBLICATION, {"status": "fixture"})
    source = {"members": []}
    monkeypatch.setattr(runner, "_validated_source", lambda: source)
    monkeypatch.setattr(
        runner, "_held_members", lambda source, donor, permit: [_held_member(donor)]
    )
    selected = [
        {
            "filename": _held_member()["filename"],
            "barcode": f"cell-{index}",
            "assigned_mln_tag": "D512-MLN-1",
            "cell_selection_sha256": f"{index:064x}",
        }
        for index in range(512)
    ]

    def census(donor, members, audit, held_permit):
        del members, audit, held_permit
        current = [dict(record) for record in selected]
        for record in current:
            record["filename"] = _held_member(donor)["filename"]
            record["assigned_mln_tag"] = f"{donor}-MLN-1"
        return current, {
            "source_members": 1,
            "candidate_mln_singlets_retained": 512,
            "selected_cells": 512,
            "selected_cell_axis_sha256": hashlib.sha256(donor.encode()).hexdigest(),
            "member_census": [],
        }

    monkeypatch.setattr(runner, "_census_donor", census)
    rows = np.tile([256, 256], (81, 1))
    columns = np.tile([256, 256], (81, 1))
    monkeypatch.setattr(runner, "_rna_margins", lambda *args: rows.copy())
    monkeypatch.setattr(runner, "_adt_margins", lambda *args: columns.copy())
    events: list[str] = []

    def predictions(methods, row_margins, column_margins):
        donor = runner._LAST_HELD_AUDIT["current_donor"]
        events.append(f"predict:{donor}")
        independence = (
            row_margins[:, :, None] * column_margins[:, None, :] / runner.CELL_BUDGET
        )
        return {name: independence.copy() for name in methods}

    monkeypatch.setattr(runner, "_predict_from_margins", predictions)

    def truth(
        donor,
        members,
        selected_cells,
        axis_sha256,
        row_margins,
        column_margins,
        prediction_sha256,
        expected_methods,
        audit,
        held_permit,
    ):
        del (
            members,
            selected_cells,
            axis_sha256,
            row_margins,
            column_margins,
            prediction_sha256,
            audit,
            held_permit,
        )
        assert all(
            (runner.HELD_PREDICTION_DIR / f"{held}.json").is_file() for held in donors
        )
        events.append(f"truth:{donor}")
        return (
            {name: 1.0 for name in expected_methods},
            {
                "informative_entities": 81,
                "informative_entity_mask": [True] * 81,
                "excluded_entities": [],
                "truth_sha256": "e" * 64,
            },
        )

    monkeypatch.setattr(runner, "_truth_and_losses", truth)
    monkeypatch.setattr(
        runner,
        "_held_gate",
        lambda losses: {"comparisons": {}, "passes_all": False},
    )
    prediction = {"frozen_source_model": _model()}
    permit = runner._ScorePermit(
        prediction_sha256="a" * 64,
        public_commit="b" * 40,
        authorization_sha256="c" * 64,
        public_prediction_url="https://github.com/o/r/blob/" + "b" * 40,
        remote_prediction_sha256="a" * 64,
    )
    result = runner._score_held_once(prediction, permit)
    assert events == ["predict:D512", "predict:D520", "truth:D512", "truth:D520"]
    assert result["access_audit"]["all_predictions_hashed_before_any_paired_truth"]
    assert result["access_audit"]["cell_count_vectors_serialized"] is False


def test_terminal_marker_precedes_engine_and_failure_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_assert_family_available", lambda: None)
    prediction = {"frozen_source_model": _model()}
    permit = runner._ScorePermit(
        "a" * 64,
        "b" * 40,
        "c" * 64,
        "https://github.com/o/r/blob/" + "b" * 40,
        "a" * 64,
    )
    monkeypatch.setattr(
        runner, "_validated_authorization", lambda: (prediction, permit)
    )

    def refuse_after_marker(*args):
        assert runner.SCORE_ATTEMPT.is_file()
        raise RuntimeError("post-marker refusal")

    monkeypatch.setattr(runner, "_score_held_once", refuse_after_marker)
    with pytest.raises(RuntimeError, match="post-marker"):
        runner.score()
    assert runner.SCORE_ATTEMPT.is_file()
    refusal = json.loads(runner.REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_SCORE_REFUSAL"
    assert refusal["partial_audit"]["current_member_deleted"] is True
    monkeypatch.setattr(
        runner,
        "_validated_authorization",
        lambda: (_ for _ in ()).throw(AssertionError("authorization reran")),
    )
    with pytest.raises(FileExistsError, match="terminal"):
        runner.score()


def test_exact_sign_flip_inclusive_tail_preserves_zero_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "HELD_DONORS", ("a", "b", "c"))
    assert runner._exact_sign_flip_p(np.zeros(3)) == 1.0
    assert runner._exact_sign_flip_p(-np.ones(3)) == 1 / 8


def test_protocol_locks_held_site_residual_and_terminal_gates() -> None:
    protocol = runner.PROTOCOL.read_text()
    normalized = " ".join(protocol.split())
    assert "Cambridge" in normalized and "LiveOnNY" in normalized
    assert "signed-root Poisson-deviance" in normalized
    assert "strongest classical residual" in normalized
    assert "exactly 512 MLN singlets" in normalized
    assert "at most three total download attempts" in normalized
    assert (
        "Byte-count, decode, schema, and analysis failures are never retried"
        in normalized
    )
    assert "GSE299043_694B_001.CZI-IA11512689.v2.h5ad" in normalized
    assert (
        "every other panel with fewer than two donor HTOs is a terminal refusal"
        in normalized
    )
    assert "at least 8 of 10" in normalized
    assert "one-sided sign-flip `p <= 0.025`" in normalized
    assert "materializes and hashes every prediction" in normalized
