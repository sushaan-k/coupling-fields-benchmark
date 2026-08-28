from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_scmmib_bmmc as candidate


ROOT = Path(__file__).resolve().parents[1]
METADATA = (
    ROOT / "data/development/scmmib_bmmc_metadata_v1/BMMC_RNA+ADT_p10_metadata.csv.gz"
)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate, "ROOT", tmp_path)
    runner = tmp_path / "experiments/confirm_scmmib_bmmc.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# frozen runner\n")
    monkeypatch.setattr(candidate, "__file__", str(runner))
    paths = {
        "PROTOCOL": "docs/protocol.md",
        "PREFLIGHT": "results/development/preflight.json",
        "SOURCE_TEMPLATE": "data/confirmation/source-template.json",
        "SOURCE_MANIFEST": "data/confirmation/source.json",
        "DEVELOPMENT_RESULT": "results/development/development.json",
        "PREDICTION": "results/prediction.json",
        "AUTH_TEMPLATE": "data/confirmation/auth-template.json",
        "AUTHORIZATION": "data/confirmation/auth.json",
        "SCORE_ATTEMPT": "data/confirmation/attempt.json",
        "OUTPUT": "results/score.json",
        "REFUSAL": "results/refusal.json",
    }
    for name, relative in paths.items():
        monkeypatch.setattr(candidate, name, tmp_path / relative)
    monkeypatch.setattr(
        candidate,
        "SANGER_TERMINAL_ARTIFACTS",
        (
            tmp_path / "data/sanger/attempt.json",
            tmp_path / "results/sanger-score.json",
            tmp_path / "results/sanger-refusal.json",
        ),
    )
    candidate.PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    candidate.PROTOCOL.write_text("frozen protocol\n")
    _json(candidate.PREFLIGHT, {"status": "PREFLIGHT_ELIGIBLE_NOT_FROZEN"})
    _json(candidate.AUTH_TEMPLATE, {"status": "OUTCOME_ACCESS_DISABLED"})


def _certificate() -> dict[str, object]:
    return {
        "converged": True,
        "scaled_gradient_norm": 1e-9,
        "gradient_tolerance": 1e-8,
        "schur_condition_number": 12.0,
        "theta_curvature_condition_number": 5.0,
    }


def _development(source_hash: str) -> dict[str, object]:
    exact = {
        "kind": "conditional_log_odds",
        "alpha": 1.0,
        "source_coordinate": np.linspace(-0.2, 0.2, 100).tolist(),
        "numerical_certificate": _certificate(),
    }
    methods = {
        "primary": exact,
        "best_residual": {
            "kind": "classical_residual",
            "family": "deviance",
            "centered": True,
            "sample_size_normalized": True,
            "alpha": 1.0,
            "source_coordinate": np.zeros(100).tolist(),
        },
        "destroyed_link": exact,
        "hierarchical_ridge_only": exact,
        "common_effect_graph": exact,
        "common_effect_ridge_only": exact,
        "label_permuted_graph": exact,
        "independence": {"kind": "independence"},
    }
    return {
        "status": "DEVELOPMENT_PASS",
        "gate": {"passes_all": True},
        "source_manifest_sha256": source_hash,
        "split": {
            "fit_donors": list(candidate.FIT_DONORS),
            "development_donors": list(candidate.DEVELOPMENT_DONORS),
            "held_donors": list(candidate.HELD_DONORS),
            "original_is_train_used": False,
            "physical_donor_disjoint": True,
            "site_disjoint": False,
        },
        "markers": list(candidate.MARKERS),
        "entity_count": 100,
        "access_audit": {
            "held_feature_rows_read": 0,
            "held_tables_formed": 0,
            "raw_x_opened": False,
        },
        "software_audit": {"hierarchical_tests_passed": True},
        "frozen_source_model": {
            "methods": methods,
            "graph": {
                "built_from_fit_donors_only": True,
                "development_or_held_outcomes_used": False,
            },
        },
    }


def test_locked_donor_roles_partition_metadata_and_ignore_original_labels() -> None:
    rows, roles = candidate._metadata_roles(METADATA)
    assert len(rows) == 9026
    assert len(roles["fit"]) == 1540
    assert len(roles["development"]) == 3067
    assert len(roles["held"]) == 4419
    assert set().union(*(set(value) for value in roles.values())) == {
        row["barcode"] for row in rows
    }
    assert {row["is_train"] for row in rows} == {"train", "test", "iid_holdout"}


def test_disabled_source_refuses_before_any_hdf5_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("predict opened an assay before source authorization")

    monkeypatch.setattr(candidate.h5py, "File", forbidden_open)
    with pytest.raises(PermissionError, match="source manifest is absent"):
        candidate.predict()
    assert not candidate.PREDICTION.exists()
    assert not candidate.SCORE_ATTEMPT.exists()


def test_any_sanger_terminal_artifact_disables_backup_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    sanger_attempt = candidate.SANGER_TERMINAL_ARTIFACTS[0]
    _json(sanger_attempt, {"status": "TERMINAL_ATTEMPT_STARTED"})
    monkeypatch.setattr(
        candidate,
        "_validated_source",
        lambda: (_ for _ in ()).throw(AssertionError("BMMC source was accessed")),
    )
    with pytest.raises(PermissionError, match="Sanger terminal artifact"):
        candidate.predict()
    assert not candidate.PREDICTION.exists()


def test_development_with_any_held_feature_access_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    _json(candidate.SOURCE_MANIFEST, {"status": "fixture"})
    payload = _development(candidate._sha256(candidate.SOURCE_MANIFEST))
    payload["access_audit"]["held_feature_rows_read"] = 1
    _json(candidate.DEVELOPMENT_RESULT, payload)
    with pytest.raises(PermissionError, match="held feature access"):
        candidate._validated_development(candidate._sha256(candidate.SOURCE_MANIFEST))


def test_predict_packages_only_audited_nonheld_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_paths(tmp_path, monkeypatch)
    _json(candidate.SOURCE_MANIFEST, {"status": "fixture"})
    source_hash = candidate._sha256(candidate.SOURCE_MANIFEST)
    development = _development(source_hash)
    _json(candidate.DEVELOPMENT_RESULT, development)
    metadata = tmp_path / "data/metadata.csv.gz"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_bytes(METADATA.read_bytes())
    dummy = tmp_path / "data/dummy.h5ad"
    dummy.write_text("axis fixture\n")
    monkeypatch.setattr(
        candidate,
        "_validated_source",
        lambda: (
            {
                "complete_cite_h5ad": {"sha256": "a" * 64},
                "combined_assay": {},
            },
            {
                "metadata": metadata,
                "preflight": candidate.PREFLIGHT,
                "complete_cite_h5ad": dummy,
            },
        ),
    )
    rows, roles = candidate._metadata_roles(metadata)
    barcodes = [row["barcode"] for row in rows]
    axis = {
        "barcodes": barcodes,
        "features": list(candidate.MARKERS),
        "marker_indices": {
            "rna": list(range(10)),
            "adt": list(range(10)),
        },
        "shape": (len(rows), 10),
    }
    monkeypatch.setattr(candidate, "_axis", lambda path, assay: axis)

    result = candidate.predict()

    assert result["status"] == "FROZEN_OUTCOME_ACCESS_DISABLED"
    assert result["held_access_audit"]["held_feature_rows_decoded"] == 0
    assert result["held_access_audit"]["held_tables_formed"] == 0
    assert result["design"]["held_donors"] == list(candidate.HELD_DONORS)
    assert result["design"]["site_disjoint"] is False
    assert candidate.PREDICTION.is_file()
    assert not candidate.SCORE_ATTEMPT.exists()


def _authorization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    _patch_paths(tmp_path, monkeypatch)
    _json(candidate.SOURCE_MANIFEST, {"status": "fixture"})
    _json(candidate.DEVELOPMENT_RESULT, {"status": "fixture"})
    prediction = {
        "status": "FROZEN_OUTCOME_ACCESS_DISABLED",
        "frozen_source_model": {},
    }
    _json(candidate.PREDICTION, prediction)
    commit = "b" * 40
    authorization = {
        "status": "OUTCOME_ACCESS_AUTHORIZED",
        "prediction_path": candidate._relative(candidate.PREDICTION),
        "prediction_sha256": candidate._sha256(candidate.PREDICTION),
        "runner_sha256": candidate._sha256(Path(candidate.__file__)),
        "protocol_sha256": candidate._sha256(candidate.PROTOCOL),
        "source_manifest_sha256": candidate._sha256(candidate.SOURCE_MANIFEST),
        "development_result_sha256": candidate._sha256(candidate.DEVELOPMENT_RESULT),
        "public_prediction_commit": commit,
        "public_prediction_url": (
            f"https://github.com/o/r/blob/{commit}/"
            f"{candidate._relative(candidate.PREDICTION)}"
        ),
    }
    _json(candidate.AUTHORIZATION, authorization)
    return authorization


def test_bad_authorization_refuses_before_held_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorization_fixture(tmp_path, monkeypatch)
    authorization["prediction_sha256"] = "0" * 64
    _json(candidate.AUTHORIZATION, authorization)
    monkeypatch.setattr(
        candidate,
        "_score_held_once",
        lambda *args: (_ for _ in ()).throw(AssertionError("held scorer ran")),
    )
    with pytest.raises(PermissionError, match="prediction_sha256"):
        candidate.score()
    assert not candidate.SCORE_ATTEMPT.exists()
    assert not candidate.REFUSAL.exists()


def test_terminal_marker_precedes_any_held_scoring_and_failure_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorization_fixture(tmp_path, monkeypatch)

    def stop_after_marker(prediction, permit):
        assert candidate.SCORE_ATTEMPT.is_file()
        assert not candidate.REFUSAL.exists()
        raise RuntimeError("deliberate post-marker stop")

    monkeypatch.setattr(candidate, "_score_held_once", stop_after_marker)
    with pytest.raises(RuntimeError, match="post-marker"):
        candidate.score()
    assert candidate.SCORE_ATTEMPT.is_file()
    refusal = json.loads(candidate.REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_SCORE_REFUSAL"


def test_protocol_locks_classical_residual_and_six_donor_gate() -> None:
    protocol = (
        ROOT / "docs/SCMMIB_BMMC_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md"
    ).read_text()
    normalized = " ".join(protocol.split())
    assert "signed Poisson-deviance" in normalized
    assert "divided by `sqrt(n)`" in normalized
    assert "All six donors favor" in normalized
    assert "0.015625" in normalized
    assert "They are not site-disjoint" in normalized
    assert "disabled if a Sanger score-attempt" in normalized
    assert "family-wise multiplicity adjustment" in normalized


def test_source_template_binds_versioned_complete_cite_object_but_is_disabled() -> None:
    template = json.loads(
        (
            ROOT / "data/confirmation/scmmib_bmmc/source_manifest_template_v1.json"
        ).read_text()
    )
    source = template["complete_cite_h5ad"]
    assert template["status"] == "SOURCE_UNAVAILABLE"
    assert source["bytes"] == candidate.COMPLETE_CITE_BYTES
    assert source["s3_bucket"] == candidate.COMPLETE_CITE_BUCKET
    assert source["s3_key"] == candidate.COMPLETE_CITE_KEY
    assert source["version_id"] == candidate.COMPLETE_CITE_VERSION
    assert source["etag"] == candidate.COMPLETE_CITE_ETAG
    assert source["local_path"] is None
    assert source["sha256"] is None
