from collections import Counter
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import confirm_combat_citeseq as combat
from mapreg.heterogeneity_adaptive_coupling import (
    signed_deviance_coordinate,
    signed_pearson_coordinate,
)


def _public_preflight() -> Path:
    relative = Path("results/development/combat_citeseq_metadata_preflight.json")
    candidates = (
        combat.ROOT / relative,
        combat.ROOT / "benchmark_release/coupling_fields_v1" / relative,
    )
    return next(path for path in candidates if path.is_file())


def _manifest_records() -> list[dict[str, str]]:
    payload = json.loads(_public_preflight().read_text())
    expected = payload["frozen_sample_contract"]["expected_metadata_by_combat_id"]
    return [
        {
            "sample": row["scRNASeq_sample_ID"],
            "combat_id": combat_id,
            "source": row["source"],
            "institute": row["institute"],
        }
        for combat_id, row in sorted(expected.items())
    ]


def _strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(
        name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8"))
    )


def _write_obs_h5ad(
    path: Path,
    *,
    combat_ids: list[str],
    samples: list[str],
    sources: list[str],
    institutes: list[str],
    cell_types: list[str],
) -> None:
    count = len(samples)
    assert all(
        len(values) == count for values in (combat_ids, sources, institutes, cell_types)
    )
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        _strings(obs, "_index", [f"cell-{index:05d}" for index in range(count)])
        _strings(obs, "COMBAT_ID", combat_ids)
        _strings(obs, "scRNASeq_sample_ID", samples)
        _strings(obs, "Source", sources)
        _strings(obs, "Institute", institutes)
        _strings(obs, "Annotation_cell_type", cell_types)


def _write_csr(group: h5py.Group, dense: np.ndarray) -> None:
    matrix = np.asarray(dense)
    rows, columns = np.nonzero(matrix)
    order = np.lexsort((columns, rows))
    rows = rows[order]
    columns = columns[order]
    indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = np.asarray(matrix.shape, dtype=np.int64)
    group.create_dataset("indptr", data=indptr)
    group.create_dataset("indices", data=columns.astype(np.int32))
    group.create_dataset("data", data=matrix[rows, columns])


def _repeat_entities(table: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(table), (len(combat.MARKERS), len(combat.MARKERS), 1, 1))


class _PublicBytes:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload


class _ObservedArray:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.requests = []

    def __getitem__(self, key):
        self.requests.append(key)
        return self.values[key]


class _FakeCsr:
    def __init__(self, indptr, indices, data, shape):
        self.attrs = {"encoding-type": "csr_matrix", "shape": shape}
        self.datasets = {
            "indptr": _ObservedArray(indptr),
            "indices": _ObservedArray(indices),
            "data": _ObservedArray(data),
        }

    def __getitem__(self, key):
        return self.datasets[key]


def _patch_designated_paths(root: Path, monkeypatch) -> dict[str, Path]:
    relative = {
        "DEFAULT_SOURCE_MANIFEST": (
            "data/confirmation/combat_citeseq/source_manifest_v1.json"
        ),
        "DEFAULT_REDUCED": "data/development/combat_citeseq/reduced_v1.json",
        "DEFAULT_PILOT": "results/development/combat_citeseq_development.json",
        "DEFAULT_PREDICTION": "results/combat_citeseq_predictions.json",
        "DEFAULT_DEVELOPMENT_AUTHORIZATION": (
            "data/confirmation/combat_citeseq/development_authorization_v1.json"
        ),
        "DEFAULT_MARGIN_AUTHORIZATION": (
            "data/confirmation/combat_citeseq/held_rna_margin_authorization_v1.json"
        ),
        "DEFAULT_PREDICTION_ATTEMPT": (
            "data/confirmation/combat_citeseq/prediction_attempt_v1.json"
        ),
        "DEFAULT_AUTHORIZATION": (
            "data/confirmation/combat_citeseq/score_authorization_v1.json"
        ),
        "DEFAULT_SCORE": "results/combat_citeseq_confirmation.json",
        "DEFAULT_SCORE_ATTEMPT": (
            "data/confirmation/combat_citeseq/score_attempt_v1.json"
        ),
        "DEFAULT_TERMINAL_REFUSAL": ("results/combat_citeseq_terminal_refusal.json"),
    }
    monkeypatch.setattr(combat, "ROOT", root)
    result = {name: root / value for name, value in relative.items()}
    for name, path in result.items():
        monkeypatch.setattr(combat, name, path)
    return result


def _development_authorization_fixture(
    root: Path, monkeypatch
) -> tuple[Path, Path, str, str, str]:
    freeze_commit = "b" * 40
    verification_commit = "c" * 40
    for label, relative in combat.DEVELOPMENT_BINDING_PATHS.items():
        if label == "fresh_clone_verification":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {label}\n")
    bindings_without_verification = {
        label: {
            "path": relative,
            "sha256": combat._sha256(root / relative),
        }
        for label, relative in combat.DEVELOPMENT_BINDING_PATHS.items()
        if label != "fresh_clone_verification"
    }
    verification_path = (
        root / combat.DEVELOPMENT_BINDING_PATHS["fresh_clone_verification"]
    )
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(
            {
                "schema": "combat-citeseq-public-freeze-verification/1.0",
                "status": "PASS",
                "fresh_clone": True,
                "origin": combat.PUBLIC_GITHUB_ORIGIN,
                "immutable_tag": "combat-citeseq-confirmation-v1",
                "verified_commit": freeze_commit,
                "designation_sha256": bindings_without_verification["designation"][
                    "sha256"
                ],
                "protocol_sha256": bindings_without_verification["protocol"]["sha256"],
                "artifact_bindings": bindings_without_verification,
                "source_h5ad_sha256": combat.OFFICIAL_H5AD_SHA256,
                "composition_csv_sha256": combat.OFFICIAL_COMPOSITION_SHA256,
                "matrix_payload_reads": 0,
                "all_bound_artifacts_match": True,
            },
            sort_keys=True,
        )
    )
    bindings = {
        label: {"path": relative, "sha256": combat._sha256(root / relative)}
        for label, relative in combat.DEVELOPMENT_BINDING_PATHS.items()
    }
    authorization_path = (
        root / "data/confirmation/combat_citeseq/development_authorization_v1.json"
    )
    authorization_path.write_text(
        json.dumps(
            {
                "schema": "combat-citeseq-development-authorization/1.0",
                "status": "OUTCOME_ACCESS_AUTHORIZED",
                "public_freeze_commit": freeze_commit,
                "public_verification_commit": verification_commit,
                "bindings": bindings,
            },
            sort_keys=True,
        )
    )
    _patch_designated_paths(root, monkeypatch)
    monkeypatch.setattr(
        combat, "__file__", str(root / "experiments/confirm_combat_citeseq.py")
    )
    return (
        authorization_path,
        root / combat.DEVELOPMENT_BINDING_PATHS["source_manifest"],
        "a" * 40,
        freeze_commit,
        verification_commit,
    )


def test_metadata_hash_split_recovers_the_frozen_12_24_51_10_roles():
    records = _manifest_records()
    roles = combat.assign_roles(records)
    by_id = {row["combat_id"]: row["sample"] for row in records}

    assert {by_id[value] for value in combat.CALIBRATION_IDS} == {
        sample for sample, role in roles.items() if role == "calibration"
    }
    assert {by_id[value] for value in combat.PILOT_IDS} == {
        sample for sample, role in roles.items() if role == "pilot"
    }
    assert Counter(roles.values()) == {
        "calibration": 12,
        "pilot": 24,
        "held_donor": 51,
        "held_site": 10,
    }


@pytest.mark.parametrize("field", ["sample", "source", "institute"])
def test_source_manifest_refuses_held_replacement_or_metadata_tamper(field):
    records = _manifest_records()
    target = next(row for row in records if row["institute"] == "St_Georges")
    target[field] = f"tampered-{field}"
    with pytest.raises(PermissionError, match="sample universe differs"):
        combat._sample_records({"samples": records})


@pytest.mark.parametrize(
    "tamper",
    ["matrix_path", "markers", "cell_types", "pool_type", "missing_pool"],
)
def test_validated_source_requires_the_frozen_assay_and_pool_contract(
    tmp_path, monkeypatch, tamper
):
    payload = json.loads(combat.DEFAULT_SOURCE_MANIFEST.read_text())
    if tamper == "matrix_path":
        payload["h5ad"]["raw_matrix_path"] = "X"
    elif tamper == "markers":
        payload["markers"][0]["rna_ensembl"] = "ENSG_TAMPERED"
    elif tamper == "cell_types":
        payload["eligible_cell_types"].append("UNKNOWN")
    elif tamper == "pool_type":
        payload["samples"][0]["eligible_pool_cells"] = "512"
    else:
        del payload["samples"][0]["official_total_pbmc_count"]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    h5ad = tmp_path / "COMBAT-CITESeq-DATA.h5ad"
    h5ad.write_bytes(b"metadata-only-test-placeholder")
    monkeypatch.setattr(combat, "_resolved_h5ad", lambda *args: h5ad)

    with pytest.raises(PermissionError):
        combat._validated_source(source, verify_hash=False)


@pytest.mark.parametrize("tamper", ["table", "profile", "authorization"])
def test_reduced_development_replays_authorized_source_outcomes(
    tmp_path, monkeypatch, tamper
):
    records = _manifest_records()
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    source_path = defaults["DEFAULT_SOURCE_MANIFEST"]
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}\n")
    roles = combat.assign_roles(records)
    by_id = {record["combat_id"]: record for record in records}
    development = sorted(
        [by_id[value]["sample"] for value in combat.CALIBRATION_IDS]
        + [by_id[value]["sample"] for value in combat.PILOT_IDS]
    )
    source_samples = [
        {"sample": record["sample"], "eligible_pool_cells": 600} for record in records
    ]
    source = {
        "payload": {"samples": source_samples},
        "records": records,
        "roles": roles,
        "h5ad": tmp_path / "COMBAT-CITESeq-DATA.h5ad",
        "source_manifest_sha256": combat._sha256(source_path),
        "h5ad_sha256": "d" * 64,
    }
    verify_hash_calls = []

    def validated_source(path, *, verify_hash):
        assert path == source_path
        verify_hash_calls.append(verify_hash)
        return source

    monkeypatch.setattr(combat, "_validated_source", validated_source)
    authorization = {
        "public_freeze_commit": "b" * 40,
        "public_verification_commit": "c" * 40,
        "public_authorization_commit": "a" * 40,
        "authorization_sha256": "e" * 64,
        "binding_sha256": {"protocol": "f" * 64},
    }
    monkeypatch.setattr(
        combat,
        "_validated_development_authorization",
        lambda path, source, commit: json.loads(json.dumps(authorization)),
    )
    table = np.asarray([[128, 128], [128, 128]], dtype=np.int64)
    table_map = _repeat_entities(table)
    flat_tables = table_map.reshape(len(combat.MARKERS) ** 2, 4).tolist()
    informative = combat._informative(table_map).reshape(-1).tolist()
    records_by_sample = {record["sample"]: record for record in records}
    reduced_records = []
    for index, sample in enumerate(development, start=1):
        source_record = records_by_sample[sample]
        reduced_records.append(
            {
                "sample": sample,
                "combat_id": source_record["combat_id"],
                "role": roles[sample],
                "cells": combat.CELL_BUDGET,
                "eligible_pool_cells": 600,
                "selected_barcode_sha256": f"{index:064x}",
                "strata": [
                    {
                        "cell_type": "B",
                        "cells": combat.CELL_BUDGET,
                        "rna_detection_prevalence": [0.5] * len(combat.MARKERS),
                        "adt_log_panel_fraction_mean": [1.0] * len(combat.MARKERS),
                    }
                ],
                "tables": json.loads(json.dumps(flat_tables)),
                "destroyed_tables": json.loads(json.dumps(flat_tables)),
                "informative": informative,
            }
        )
    replay = json.loads(json.dumps(reduced_records))
    monkeypatch.setattr(
        combat,
        "_development_records",
        lambda frozen_source: json.loads(json.dumps(replay)),
    )
    payload = {
        "schema": "combat-citeseq-reduced-development/1.0",
        "status": "DEVELOPMENT_REDUCTION_COMPLETE",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "source_manifest_sha256": source["source_manifest_sha256"],
        "h5ad_sha256": source["h5ad_sha256"],
        "development_authorization": json.loads(json.dumps(authorization)),
        "markers": list(combat.MARKERS),
        "cell_types": list(combat.CELL_TYPES),
        "cells_per_sample": combat.CELL_BUDGET,
        "cell_selection_salt": combat.CELL_SELECTION_SALT,
        "adt_tie_salt": combat.ADT_TIE_SALT,
        "samples": json.loads(json.dumps(reduced_records)),
        "access_audit": {
            "calibration_samples_read": 12,
            "pilot_samples_read": 24,
            "held_donor_matrix_rows_read": 0,
            "held_site_matrix_rows_read": 0,
            "modalities_read_sequentially": ["rna", "adt"],
            "matrix_values_decoded_only_for_frozen_features": 18,
        },
    }
    valid_path = defaults["DEFAULT_REDUCED"]
    combat._write_json(valid_path, payload)
    assert len(combat._validated_reduced(valid_path, source_path)["by_sample"]) == 36
    assert verify_hash_calls == [True]

    tampered = json.loads(json.dumps(payload))
    if tamper == "table":
        replacement = [129, 127, 127, 129]
        tampered["samples"][0]["tables"][0] = replacement
        tampered["samples"][0]["destroyed_tables"][0] = replacement
    elif tamper == "profile":
        tampered["samples"][0]["strata"][0]["rna_detection_prevalence"][0] = 0.51
    else:
        tampered["development_authorization"]["authorization_sha256"] = "0" * 64
    tampered_path = tmp_path / f"reduced-{tamper}.json"
    combat._write_json(tampered_path, tampered)
    with pytest.raises(PermissionError):
        combat._validated_reduced(tampered_path, source_path)


def test_source_seal_accepts_only_the_clean_metadata_pass_and_is_path_free(
    tmp_path, monkeypatch
):
    preflight_bytes = _public_preflight().read_bytes()
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    preflight = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["metadata_preflight_result"]
    preflight.parent.mkdir(parents=True)
    preflight.write_bytes(preflight_bytes)
    output = defaults["DEFAULT_SOURCE_MANIFEST"]

    payload = combat.seal_source(preflight, output)

    assert payload["status"] == "SOURCE_SEALED"
    assert len(payload["samples"]) == 97
    serialized = output.read_text()
    assert "/Users/" not in serialized
    assert "file://" not in serialized
    assert '"path"' not in serialized


def test_source_seal_refuses_a_preflight_that_read_matrix_payload(
    tmp_path, monkeypatch
):
    payload = json.loads(_public_preflight().read_text())
    payload["access_audit"]["matrix_payload_reads"] = 1
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    preflight = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["metadata_preflight_result"]
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps(payload))

    with pytest.raises(PermissionError, match="metadata-only v3 PASS"):
        combat.seal_source(preflight, defaults["DEFAULT_SOURCE_MANIFEST"])


def test_source_seal_refuses_malformed_pool_metadata(tmp_path, monkeypatch):
    payload = json.loads(_public_preflight().read_text())
    counts = payload["frozen_sample_contract"]["designated_sample_counts"]
    counts[next(iter(counts))]["eligible_cells"] = "512"
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    preflight = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["metadata_preflight_result"]
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="pool metadata is invalid"):
        combat.seal_source(preflight, defaults["DEFAULT_SOURCE_MANIFEST"])


def test_default_public_artifact_paths_match_the_designation():
    designation = json.loads(
        (
            combat.ROOT
            / "data/confirmation/combat_citeseq/candidate_designation_v1.json"
        ).read_text()
    )["artifacts"]
    assert combat.DEFAULT_PILOT == combat.ROOT / designation["development_result"]
    assert (
        combat.DEFAULT_PREDICTION_ATTEMPT
        == combat.ROOT / designation["held_prediction_attempt"]
    )
    assert combat.DEFAULT_PREDICTION == combat.ROOT / designation["held_predictions"]
    assert combat.DEFAULT_SCORE_ATTEMPT == combat.ROOT / designation["score_attempt"]
    assert combat.DEFAULT_SCORE == combat.ROOT / designation["score_result"]
    assert (
        combat.DEFAULT_TERMINAL_REFUSAL == combat.ROOT / designation["terminal_refusal"]
    )


def test_post_attempt_failure_writes_one_sanitized_terminal_refusal_and_blocks_rerun(
    tmp_path, monkeypatch
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    runner_path = tmp_path / "experiments/confirm_combat_citeseq.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner\n")
    monkeypatch.setattr(combat, "__file__", str(runner_path))

    authorization = defaults["DEFAULT_AUTHORIZATION"]
    attempt = defaults["DEFAULT_SCORE_ATTEMPT"]
    refusal = defaults["DEFAULT_TERMINAL_REFUSAL"]
    authorization.parent.mkdir(parents=True)
    authorization.write_text("authorization\n")
    attempt.write_text("attempt\n")
    calls = 0

    def failure():
        nonlocal calls
        calls += 1
        raise RuntimeError(f"{tmp_path}/private.tsv reviewer@example.org")

    with pytest.raises(RuntimeError):
        combat._run_terminal_phase(
            "held_score", attempt, authorization, refusal, failure
        )

    payload = json.loads(refusal.read_text())
    assert payload["status"] == "TERMINAL_REFUSAL"
    assert payload["phase"] == "held_score"
    assert payload["attempt_sha256"] == combat._sha256(attempt)
    assert payload["authorization_sha256"] == combat._sha256(authorization)
    assert str(tmp_path) not in payload["reason"]
    assert "reviewer@example.org" not in payload["reason"]
    first_bytes = refusal.read_bytes()

    with pytest.raises(FileExistsError, match="reruns are forbidden"):
        combat._run_terminal_phase(
            "held_score", attempt, authorization, refusal, failure
        )
    assert calls == 1
    assert refusal.read_bytes() == first_bytes


def test_pre_attempt_authorization_failure_is_nonterminal(tmp_path, monkeypatch):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    authorization = defaults["DEFAULT_MARGIN_AUTHORIZATION"]
    attempt = defaults["DEFAULT_PREDICTION_ATTEMPT"]
    refusal = defaults["DEFAULT_TERMINAL_REFUSAL"]

    def failure():
        raise PermissionError("designation remains outcome-access disabled")

    with pytest.raises(PermissionError):
        combat._run_terminal_phase(
            "held_prediction", attempt, authorization, refusal, failure
        )
    assert not attempt.exists()
    assert not refusal.exists()


@pytest.mark.parametrize(
    "secret",
    [
        "/Volumes/private-study/outcomes.tsv",
        r"C:\\Users\\analyst\\private-study\\outcomes.tsv",
    ],
)
def test_terminal_refusal_reason_never_persists_exception_text(secret):
    reason = combat._sanitized_error("held_score", RuntimeError(secret))
    assert secret not in reason
    assert reason == "held_score: an authorized runtime step failed"


def test_pilot_validation_reconstructs_frozen_sample_order_after_json_round_trip(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(combat, "ROOT", tmp_path)
    monkeypatch.setattr(combat, "BOOTSTRAPS", 100)
    runner_path = tmp_path / "experiments/confirm_combat_citeseq.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner\n")
    monkeypatch.setattr(combat, "__file__", str(runner_path))

    protocol_path = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["protocol"]
    designation_path = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["designation"]
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    designation_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("protocol\n")
    designation_path.write_text("designation\n")
    source_path = tmp_path / "data/confirmation/combat_citeseq/source_manifest_v1.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}\n")
    reduced_path = tmp_path / "data/development/combat_citeseq/reduced_v1.json"
    reduced_path.parent.mkdir(parents=True, exist_ok=True)
    reduced_path.write_text("reduced\n")

    calibration_samples = tuple(f"calibration-{index:02d}" for index in range(12))
    pilot_samples = tuple(f"pilot-{index:02d}" for index in range(24, 0, -1))
    monkeypatch.setattr(combat, "_sample_records", lambda payload: ["record"])
    monkeypatch.setattr(combat, "assign_roles", lambda records: None)

    def frozen_samples(records, identifiers):
        if identifiers == combat.CALIBRATION_IDS:
            return calibration_samples
        if identifiers == combat.PILOT_IDS:
            return pilot_samples
        raise AssertionError("unexpected identifier set")

    monkeypatch.setattr(combat, "_samples_for_ids", frozen_samples)
    monkeypatch.setattr(
        combat,
        "_validated_reduced",
        lambda path, source: {
            "by_sample": {},
            "calibration_samples": calibration_samples,
            "pilot_samples": pilot_samples,
            "payload": {
                "development_authorization": {"authorization_sha256": "a" * 64}
            },
        },
    )

    def frozen_models(records, samples, config, residual, ridge):
        return {
            name: {
                "kind": name,
                "source_coordinate": [1.0, 2.0],
                "configuration": list(config),
                "residual": list(residual),
                "unstructured_ridge": ridge,
            }
            for name in combat.METHODS
        }

    monkeypatch.setattr(combat, "_fit_method_panel", frozen_models)
    models = frozen_models(
        {},
        calibration_samples + pilot_samples,
        combat.CONFIG_GRID[0],
        combat.CLASSICAL_GRID[0],
        0.0,
    )
    primary = np.linspace(1.0, 2.0, len(pilot_samples))
    losses = {
        name: primary if name == "primary" else primary + 1.0 for name in combat.METHODS
    }
    comparisons = {
        name: combat._comparison(
            pilot_samples,
            losses["primary"],
            losses[name],
            favorable_required=(19 if name in combat.PROMOTION_COMPARATORS else None),
        )
        for name in combat.METHODS[1:]
    }
    analysis = {
        "status": "PILOT_PASS",
        "configuration_grid": [
            combat._configuration(config) for config in combat.CONFIG_GRID
        ],
        "primary_candidate_evaluations": [
            {
                "configuration": combat._configuration(combat.CONFIG_GRID[0]),
                "status": "EVALUATED",
                "mean_pilot_deviance_per_cell": float(primary.mean()),
            }
        ],
        "classical_candidate_evaluations": [
            {
                "family": combat.CLASSICAL_GRID[0][0],
                "centered": combat.CLASSICAL_GRID[0][1],
                "status": "EVALUATED",
                "mean_pilot_deviance_per_cell": float((primary + 1.0).mean()),
            }
        ],
        "unstructured_candidate_evaluations": [
            {
                "ridge_penalty": 0.0,
                "status": "EVALUATED",
                "mean_pilot_deviance_per_cell": float((primary + 1.0).mean()),
            }
        ],
        "selection": {
            "selected_primary_configuration": combat._configuration(
                combat.CONFIG_GRID[0]
            ),
            "selected_classical_residual": {
                "family": combat.CLASSICAL_GRID[0][0],
                "centered": combat.CLASSICAL_GRID[0][1],
            },
            "selected_unstructured_ridge_penalty": 0.0,
            "fit_samples": list(calibration_samples),
            "selection_samples": list(pilot_samples),
            "refit_samples_after_gate": list(calibration_samples + pilot_samples),
            "retuned_after_gate": False,
        },
        "pilot_losses": {
            name: {
                sample: float(value)
                for sample, value in zip(pilot_samples, method_losses)
            }
            for name, method_losses in losses.items()
        },
        "pilot_prediction_flags": {name: [] for name in combat.METHODS},
        "pilot_comparisons": comparisons,
        "promotion_comparators": list(combat.PROMOTION_COMPARATORS),
        "passes_pilot_gate": True,
        "frozen_source_models": models,
        "all_development_graph": {"graph": "expected"},
    }
    monkeypatch.setattr(
        combat,
        "_pilot_analysis",
        lambda records, calibration, pilot: json.loads(json.dumps(analysis)),
    )
    payload = {
        "schema": "combat-citeseq-pilot-fit/1.0",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "source_manifest_sha256": combat._sha256(source_path),
        "reduced_development_sha256": combat._sha256(reduced_path),
        "development_authorization_sha256": "a" * 64,
        "protocol_sha256": combat._sha256(protocol_path),
        "designation_sha256": combat._sha256(designation_path),
        "runner_sha256": combat._sha256(runner_path),
        **analysis,
        "reconstruction": {
            "primary": "exact fixed-margin moment-calibrated exponential tilt of centered Haldane statistic",
            "classical": "exact fixed-margin moment-calibrated exponential tilt after target sqrt(n) restoration",
            "out_of_range": "refuse; never clip",
            "direct_h_inverse": "diagnostic only and not used for prediction",
        },
    }
    pilot_path = tmp_path / "results/development/combat_citeseq_development.json"
    combat._write_json(pilot_path, payload, exclusive=True)
    serialized = combat._read_json(pilot_path)
    assert list(serialized["pilot_losses"]["primary"]) != list(pilot_samples)

    validated = combat._validated_pilot(
        pilot_path, source_path, reduced_path, require_pass=True
    )
    assert validated == serialized

    for label in ("coordinate", "configuration", "model", "coordinated"):
        tampered = json.loads(json.dumps(payload))
        if label == "coordinate":
            tampered["frozen_source_models"]["primary"]["source_coordinate"][0] += 0.1
        elif label == "configuration":
            tampered["selection"]["selected_primary_configuration"] = (
                combat._configuration(combat.CONFIG_GRID[1])
            )
        elif label == "model":
            tampered["frozen_source_models"]["primary"]["kind"] = "tampered"
        else:
            replacement = combat.CONFIG_GRID[1]
            tampered["selection"]["selected_primary_configuration"] = (
                combat._configuration(replacement)
            )
            tampered["frozen_source_models"] = frozen_models(
                {},
                calibration_samples + pilot_samples,
                replacement,
                combat.CLASSICAL_GRID[0],
                0.0,
            )
            fabricated = {sample: 0.1 for sample in pilot_samples}
            tampered["pilot_losses"]["primary"] = fabricated
        path = tmp_path / f"results/development/tampered-{label}.json"
        combat._write_json(path, tampered, exclusive=True)
        with pytest.raises(PermissionError):
            combat._validated_pilot(path, source_path, reduced_path, require_pass=True)


def test_fit_pilot_refuses_singular_zero_ridge_and_continues_positive_ridges(
    tmp_path, monkeypatch
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    defaults["DEFAULT_REDUCED"].parent.mkdir(parents=True)
    defaults["DEFAULT_REDUCED"].write_text("reduced\n")
    calibration = tuple(f"cal-{index:02d}" for index in range(12))
    pilot = tuple(f"pilot-{index:02d}" for index in range(24))
    monkeypatch.setattr(
        combat,
        "_validated_reduced",
        lambda *args: {
            "by_sample": {},
            "calibration_samples": calibration,
            "pilot_samples": pilot,
            "source": {"source_manifest_sha256": "a" * 64},
            "payload": {
                "development_authorization": {
                    "authorization_sha256": "b" * 64,
                    "binding_sha256": {
                        "protocol": "c" * 64,
                        "designation": "d" * 64,
                    },
                }
            },
        },
    )
    monkeypatch.setattr(
        combat,
        "_tables",
        lambda records, samples, key: np.zeros(
            (len(samples), len(combat.MARKERS), len(combat.MARKERS), 2, 2)
        ),
    )
    incidence = np.eye(len(combat.MARKERS))
    monkeypatch.setattr(
        combat, "_graphs", lambda *args: (incidence, incidence, {"graph": "test"})
    )
    monkeypatch.setattr(
        combat, "_field_model", lambda *args, **kwargs: {"kind": "field"}
    )
    monkeypatch.setattr(combat, "_classical_model", lambda *args: {"kind": "classical"})

    def unstructured(tables, ridge):
        if ridge == 0.0:
            raise ValueError("singular unregularized solve")
        return {"kind": "unstructured", "ridge": ridge}

    monkeypatch.setattr(combat, "_unstructured_model", unstructured)
    monkeypatch.setattr(
        combat,
        "_model_losses",
        lambda model, values, flags: np.ones(len(values), dtype=float),
    )
    monkeypatch.setattr(
        combat,
        "_fit_method_panel",
        lambda *args: {name: {"kind": name} for name in combat.METHODS},
    )
    monkeypatch.setattr(combat, "BOOTSTRAPS", 100)

    payload = combat.fit_pilot(
        defaults["DEFAULT_SOURCE_MANIFEST"],
        defaults["DEFAULT_REDUCED"],
        defaults["DEFAULT_PILOT"],
    )
    rows = payload["unstructured_candidate_evaluations"]
    assert len(payload["primary_candidate_evaluations"]) == 8
    assert len(payload["classical_candidate_evaluations"]) == 4
    assert [(row["ridge_penalty"], row["status"]) for row in rows] == [
        (0.0, "REFUSED"),
        (0.01, "EVALUATED"),
        (0.1, "EVALUATED"),
    ]


def test_exact_sample_pair_selection_never_pools_an_extra_timepoint(tmp_path):
    exact_sample = "S00024-Ca001E-PBCa"
    extra_sample = "S00024-Ca001E-PBCb"
    path = tmp_path / "synthetic.h5ad"
    _write_obs_h5ad(
        path,
        combat_ids=["S00024"] * 520,
        samples=[exact_sample] * 513 + [extra_sample] * 7,
        sources=["COVID_CRIT"] * 520,
        institutes=["Oxford"] * 520,
        cell_types=["B"] * 520,
    )
    record = {
        "combat_id": "S00024",
        "sample": exact_sample,
        "source": "COVID_CRIT",
        "institute": "Oxford",
    }

    selected = combat._selected_sample_rows(path, [record])[exact_sample]

    assert len(selected["rows"]) == combat.CELL_BUDGET
    assert selected["eligible_pool_cells"] == 513
    assert np.all(selected["rows"] < 513)


def test_legacy_h5ad_categorical_columns_decode_missing_codes(tmp_path):
    path = tmp_path / "legacy.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("Source", data=np.asarray([0, 1, -1], dtype=np.int8))
        categories = obs.create_group("__categories")
        _strings(categories, "Source", ["Oxford", "Flu"])
    with h5py.File(path, "r") as handle:
        decoded = combat._encoded_column(handle["obs"], "Source")
    assert decoded.tolist() == ["Oxford", "Flu", ""]


def test_locked_feature_lookup_uses_version_stripped_ensembl_and_cd44_alias(tmp_path):
    path = tmp_path / "features.h5ad"
    names = list(combat.MARKERS) + [
        combat.ADT_FEATURE[value] for value in combat.MARKERS
    ]
    gene_ids = [f"{combat.RNA_ENSEMBL[value]}.12" for value in combat.MARKERS]
    gene_ids += [""] * len(combat.MARKERS)
    feature_types = ["Gene Expression"] * len(combat.MARKERS)
    feature_types += ["Antibody Capture"] * len(combat.MARKERS)
    with h5py.File(path, "w") as handle:
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        _strings(var, "_index", names)
        _strings(var, "gene_ids", gene_ids)
        _strings(var, "feature_types", feature_types)
    with h5py.File(path, "r") as handle:
        columns = combat._feature_columns(handle)
    assert columns == {"rna": list(range(9)), "adt": list(range(9, 18))}
    assert names[15] == "AB_humanCD44"


def test_csr_reader_fetches_only_requested_rows_and_columns(tmp_path):
    dense = np.arange(1, 121, dtype=np.float32).reshape(4, 30)
    dense[dense % 5 == 0] = 0.0
    path = tmp_path / "matrix.h5"
    with h5py.File(path, "w") as handle:
        _write_csr(handle.create_group("raw"), dense)
    with h5py.File(path, "r") as handle:
        observed = combat._read_csr_feature_subset(
            handle["raw"], np.asarray([1, 3]), np.asarray([2, 20, 27])
        )
    np.testing.assert_array_equal(observed, dense[[1, 3]][:, [2, 20, 27]])


def test_csr_reader_scans_structure_but_decodes_only_frozen_numeric_columns():
    matrix = _FakeCsr(
        [0, 5, 10, 15, 20],
        [0, 1, 2, 3, 4] * 4,
        np.arange(20, dtype=float) + 1,
        (4, 5),
    )
    observed = combat._read_csr_feature_subset(
        matrix, np.asarray([1, 3]), np.asarray([1, 4])
    )
    np.testing.assert_array_equal(observed, [[7.0, 10.0], [17.0, 20.0]])
    assert len(matrix.datasets["indptr"].requests) == 1
    np.testing.assert_array_equal(
        matrix.datasets["indptr"].requests[0], np.asarray([1, 2, 3, 4])
    )
    assert matrix.datasets["indices"].requests == [slice(5, 10), slice(15, 20)]
    numeric_positions = np.concatenate(
        [np.asarray(request) for request in matrix.datasets["data"].requests]
    )
    np.testing.assert_array_equal(numeric_positions, [6, 9, 16, 19])


def test_csr_reader_refuses_duplicate_column_indices():
    matrix = _FakeCsr([0, 3], [0, 1, 1], [1.0, 2.0, 3.0], (1, 3))
    with pytest.raises(ValueError, match="duplicate column index"):
        combat._read_csr_feature_subset(matrix, np.asarray([0]), np.asarray([1]))


def test_adt_midrank_is_exact_and_destroyed_link_preserves_whole_profiles():
    counts = np.tile(np.arange(combat.CELL_BUDGET) % 7, (len(combat.MARKERS), 1))
    counts += np.arange(len(combat.MARKERS))[:, None]
    barcodes = np.asarray([f"cell-{index:04d}" for index in range(combat.CELL_BUDGET)])
    states = combat._adt_states(counts, barcodes, "S00024", "sample")
    repeated = combat._adt_states(counts, barcodes, "S00024", "sample")
    destroyed = combat._destroyed_adt(states, barcodes, "sample")

    np.testing.assert_array_equal(states, repeated)
    np.testing.assert_array_equal(states.sum(axis=1), 256)
    assert Counter(map(tuple, states.T)) == Counter(map(tuple, destroyed.T))
    np.testing.assert_array_equal(states.sum(axis=1), destroyed.sum(axis=1))


def test_graph_construction_is_deterministic_and_has_unweighted_two_endpoint_edges():
    profile = np.asarray(
        [
            [(row + 1) * (column + 2) + column**2 for column in range(9)]
            for row in range(13)
        ],
        dtype=float,
    )
    first = combat._knn_incidence(profile, 2)
    second = combat._knn_incidence(profile, 2)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), 2.0)
    assert set(np.unique(first)) <= {0.0, 1.0}
    assert np.all(first.sum(axis=1) > 0.0)


def test_label_permutation_assigns_hash_order_to_lexical_graph_positions():
    incidence = np.arange(9 * 4, dtype=float).reshape(9, 4)
    observed = combat._permuted_incidence(incidence, "rna")
    hashed = sorted(
        range(9),
        key=lambda index: combat.hashlib.sha256(
            "\0".join(
                (
                    combat.LABEL_PERMUTATION_SALT,
                    "rna",
                    combat.MARKERS[index],
                )
            ).encode()
        ).hexdigest(),
    )
    positions = sorted(range(9), key=lambda index: combat.MARKERS[index])
    for marker, position in zip(hashed, positions):
        np.testing.assert_array_equal(observed[marker], incidence[position])


def test_centered_haldane_zero_returns_the_independence_expectation():
    rows = np.asarray([3, 7])
    columns = np.asarray([4, 6])
    table, theta = combat._moment_calibrated_table(
        0.0, rows, columns, family="haldane", centered=True
    )

    assert theta == pytest.approx(0.0, abs=1e-12)
    assert table[0, 0] == pytest.approx(3 * 4 / 10)
    np.testing.assert_allclose(table.sum(axis=1), rows)
    np.testing.assert_allclose(table.sum(axis=0), columns)


def test_moment_calibration_has_exact_boundaries_and_refuses_out_of_range():
    rows = np.asarray([3, 7])
    columns = np.asarray([4, 6])
    support, statistic, _, null_mean = combat._moment_support("haldane", 3, 7, 4, 6)
    lower_coordinate = float(statistic[0] - null_mean)
    upper_coordinate = float(statistic[-1] - null_mean)

    lower, lower_theta = combat._moment_calibrated_table(
        lower_coordinate, rows, columns, family="haldane", centered=True
    )
    upper, upper_theta = combat._moment_calibrated_table(
        upper_coordinate, rows, columns, family="haldane", centered=True
    )
    assert lower[0, 0] == support[0]
    assert upper[0, 0] == support[-1]
    assert lower_theta == -math.inf
    assert upper_theta == math.inf
    with pytest.raises(ValueError, match="outside attainable"):
        combat._moment_calibrated_table(
            lower_coordinate - 0.1,
            rows,
            columns,
            family="haldane",
            centered=True,
        )


@pytest.mark.parametrize("family", ["haldane", "pearson", "deviance"])
def test_moment_calibration_strictly_refuses_outside_and_solves_inside(family):
    rows = np.asarray([17, 23])
    columns = np.asarray([19, 21])
    _, statistic, _, null_mean = combat._moment_support(family, 17, 23, 19, 21)
    lower = float(statistic[0] - null_mean)
    upper = float(statistic[-1] - null_mean)

    for outside in (np.nextafter(lower, -math.inf), np.nextafter(upper, math.inf)):
        with pytest.raises(ValueError, match="outside attainable"):
            combat._moment_calibrated_table(
                outside, rows, columns, family=family, centered=True
            )
    for inside in (np.nextafter(lower, upper), np.nextafter(upper, lower)):
        table, theta = combat._moment_calibrated_table(
            inside, rows, columns, family=family, centered=True
        )
        assert np.isfinite(theta)
        np.testing.assert_allclose(table.sum(axis=1), rows)
        np.testing.assert_allclose(table.sum(axis=0), columns)


def test_moment_calibration_differs_from_direct_h_inverse_for_3_4_10():
    rows = np.asarray([3, 7])
    columns = np.asarray([4, 6])
    _, statistic, _, null_mean = combat._moment_support("haldane", 3, 7, 4, 6)
    coordinate = float(statistic[2] - null_mean)

    moment, theta = combat._moment_calibrated_table(
        coordinate, rows, columns, family="haldane", centered=True
    )
    direct = combat._direct_haldane_table(coordinate, rows, columns, centered=True)

    assert np.isfinite(theta)
    assert direct[0, 0] == pytest.approx(2.0)
    assert moment[0, 0] == pytest.approx(1.9295622518446378)
    assert moment[0, 0] != pytest.approx(direct[0, 0])


def test_moment_calibration_round_trips_an_exponential_tilt_moment():
    rows = np.asarray([17, 23])
    columns = np.asarray([19, 21])
    support, statistic, logbase, null_mean = combat._moment_support(
        "haldane", 17, 23, 19, 21
    )
    source_theta = 0.7
    probability = np.exp(
        logbase
        + source_theta * support
        - combat.logsumexp(logbase + source_theta * support)
    )
    coordinate = float(probability @ statistic - null_mean)
    expected_x = float(probability @ support)

    table, fitted_theta = combat._moment_calibrated_table(
        coordinate, rows, columns, family="haldane", centered=True
    )

    assert fitted_theta == pytest.approx(source_theta, abs=1e-11)
    assert table[0, 0] == pytest.approx(expected_x, abs=1e-11)


def test_degenerate_margin_returns_and_flags_the_unique_table_for_any_coordinate():
    table, theta = combat._moment_calibrated_table(
        100.0,
        np.asarray([0, 10]),
        np.asarray([4, 6]),
        family="haldane",
        centered=True,
    )
    np.testing.assert_array_equal(table, [[0.0, 0.0], [4.0, 6.0]])
    assert theta == 0.0

    truth = _repeat_entities(np.asarray([[12, 8], [5, 15]]))
    truth[0] = np.tile(np.asarray([[0, 0], [17, 23]]), (9, 1, 1))
    flags = []
    losses = combat._model_losses(
        {"kind": "independence", "estimator": "test"}, truth[None, ...], flags
    )
    assert np.isfinite(losses).all()
    assert len(flags) == 1
    assert len(flags[0]["flags"]) == 9
    assert {row["status"] for row in flags[0]["flags"]} == {"degenerate_unique_table"}


@pytest.mark.parametrize(
    "statistic", [signed_pearson_coordinate, signed_deviance_coordinate]
)
def test_classical_coordinate_normalization_is_sample_size_invariant(statistic):
    table = np.asarray([[12, 8], [5, 15]])
    baseline = statistic(table) / math.sqrt(table.sum())
    for multiplier in (4, 9):
        scaled = table * multiplier
        assert statistic(scaled) / math.sqrt(scaled.sum()) == pytest.approx(
            baseline, abs=1e-14
        )


def test_prediction_is_nonnegative_and_preserves_every_target_margin():
    rows = np.tile(np.asarray([[3, 7]]), (9, 1))
    columns = np.tile(np.asarray([[4, 6]]), (9, 1))
    prediction = combat._predict_method(
        {"kind": "independence", "estimator": "test"}, rows, columns
    )

    assert np.isfinite(prediction).all()
    assert np.all(prediction >= 0.0)
    np.testing.assert_allclose(
        prediction.sum(axis=-1), np.broadcast_to(rows[:, None, :], (9, 9, 2))
    )
    np.testing.assert_allclose(
        prediction.sum(axis=-2), np.broadcast_to(columns[None, :, :], (9, 9, 2))
    )


def test_deviance_loss_matches_declared_formula_and_endpoint_zero_is_infinite():
    truth_entity = np.asarray([[12.0, 8.0], [5.0, 15.0]])
    truth = _repeat_entities(truth_entity)
    rows = truth_entity.sum(axis=1)
    columns = truth_entity.sum(axis=0)
    prediction_entity = np.outer(rows, columns) / truth_entity.sum()
    prediction = _repeat_entities(prediction_entity)
    positive = truth_entity > 0.0
    expected = (
        2.0
        / truth_entity.sum()
        * np.sum(
            truth_entity[positive]
            * np.log(truth_entity[positive] / prediction_entity[positive])
        )
    )
    support = np.ones(81, dtype=bool)

    assert combat._donor_loss(truth, prediction, support) == pytest.approx(expected)

    boundary = np.asarray([[0.0, rows[0]], [columns[0], rows[1] - columns[0]]])
    assert np.all(boundary >= 0.0)
    assert combat._donor_loss(truth, _repeat_entities(boundary), support) == math.inf


def test_invalid_development_authorization_refuses_before_source_access(
    tmp_path, monkeypatch
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    authorization = defaults["DEFAULT_DEVELOPMENT_AUTHORIZATION"]
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}")
    monkeypatch.setattr(
        combat,
        "_validated_source",
        lambda *args, **kwargs: pytest.fail("source was accessed before authorization"),
    )

    with pytest.raises(PermissionError, match="outcome access is disabled"):
        combat.reduce_development(
            defaults["DEFAULT_SOURCE_MANIFEST"],
            authorization,
            "a" * 40,
            defaults["DEFAULT_REDUCED"],
        )


def test_development_authorization_requires_identical_public_commit_bytes(
    tmp_path, monkeypatch
):
    (
        authorization,
        source,
        authorization_commit,
        freeze_commit,
        verification_commit,
    ) = _development_authorization_fixture(tmp_path, monkeypatch)
    requested = []

    def public_copy(request, timeout):
        requested.append((request.full_url, timeout))
        suffix = request.full_url.split("coupling-fields-benchmark/", 1)[1]
        commit, relative = suffix.split("/", 1)
        if commit == authorization_commit:
            payload = authorization.read_bytes()
        elif commit == verification_commit:
            payload = (tmp_path / relative).read_bytes()
        elif commit == freeze_commit:
            payload = (tmp_path / relative).read_bytes()
        else:
            raise AssertionError("unexpected public commit")
        return _PublicBytes(payload)

    monkeypatch.setattr(combat.urllib.request, "urlopen", public_copy)
    permit = combat._validated_development_authorization(
        authorization, source, authorization_commit
    )
    assert permit["public_authorization_commit"] == authorization_commit
    assert permit["public_verification_commit"] == verification_commit
    assert len(requested) == len(combat.DEVELOPMENT_BINDING_PATHS) + 1
    assert all(timeout == 120 for _, timeout in requested)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("verification", "fresh-clone verification differs"),
        ("freeze_wrong", "frozen runner differs"),
        ("freeze_unavailable", "frozen runner fetch failed"),
        ("authorization", "development authorization differs"),
    ],
)
def test_development_authorization_rejects_nonpublic_chain_bytes(
    tmp_path, monkeypatch, failure, message
):
    (
        authorization,
        source,
        authorization_commit,
        freeze_commit,
        verification_commit,
    ) = _development_authorization_fixture(tmp_path, monkeypatch)

    def public_copy(request, timeout):
        suffix = request.full_url.split("coupling-fields-benchmark/", 1)[1]
        commit, relative = suffix.split("/", 1)
        if failure == "verification" and commit == verification_commit:
            return _PublicBytes(b"wrong verification bytes")
        if (
            failure in {"freeze_wrong", "freeze_unavailable"}
            and commit == freeze_commit
            and relative == combat.DEVELOPMENT_BINDING_PATHS["runner"]
        ):
            if failure == "freeze_unavailable":
                raise OSError("missing frozen object")
            return _PublicBytes(b"wrong frozen runner")
        if failure == "authorization" and commit == authorization_commit:
            return _PublicBytes(b"wrong authorization bytes")
        return _PublicBytes((tmp_path / relative).read_bytes())

    monkeypatch.setattr(combat.urllib.request, "urlopen", public_copy)
    with pytest.raises(PermissionError, match=message):
        combat._validated_development_authorization(
            authorization, source, authorization_commit
        )


def test_invalid_margin_authorization_refuses_before_held_extraction(
    tmp_path, monkeypatch
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    authorization = defaults["DEFAULT_MARGIN_AUTHORIZATION"]
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}")
    monkeypatch.setattr(combat, "_validated_pilot", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        combat,
        "_extract_held_rna_margins",
        lambda *args, **kwargs: pytest.fail(
            "held RNA was accessed before authorization"
        ),
    )

    with pytest.raises(PermissionError, match="RNA margin access is disabled"):
        combat.predict_held_margins(
            defaults["DEFAULT_SOURCE_MANIFEST"],
            defaults["DEFAULT_PILOT"],
            authorization,
            "a" * 40,
            defaults["DEFAULT_PREDICTION_ATTEMPT"],
            defaults["DEFAULT_PREDICTION"],
        )


def test_terminal_phase_refuses_a_non_designated_output_path(tmp_path, monkeypatch):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="output path differs"):
        combat.predict_held_margins(
            defaults["DEFAULT_SOURCE_MANIFEST"],
            defaults["DEFAULT_PILOT"],
            defaults["DEFAULT_MARGIN_AUTHORIZATION"],
            "a" * 40,
            defaults["DEFAULT_PREDICTION_ATTEMPT"],
            tmp_path / "alternate-prediction.json",
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "prediction",
        "boundary",
        "permit_attempt",
        "top_truth_extra",
        "row_adt_extra",
        "audit_mutation",
        "audit_omission",
        "barcode_digest",
        "eligible_pool",
    ],
)
def test_prediction_validation_recomputes_models_boundaries_and_permit_chain(
    tmp_path, monkeypatch, tamper
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    runner = tmp_path / "experiments/confirm_combat_citeseq.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner\n")
    monkeypatch.setattr(combat, "__file__", str(runner))
    source_path = defaults["DEFAULT_SOURCE_MANIFEST"]
    pilot_path = defaults["DEFAULT_PILOT"]
    for path, text in ((source_path, "source\n"), (pilot_path, "pilot\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sample_names = [f"held-{index:02d}" for index in range(61)]
    roles = {
        sample: "held_donor" if index < 51 else "held_site"
        for index, sample in enumerate(sample_names)
    }
    source_record = {
        "source_manifest_sha256": combat._sha256(source_path),
        "h5ad_sha256": "d" * 64,
        "roles": roles,
        "h5ad": tmp_path / "COMBAT-CITESeq-DATA.h5ad",
        "records": [{"sample": sample} for sample in sample_names],
    }
    monkeypatch.setattr(
        combat, "_validated_source", lambda *args, **kwargs: source_record
    )
    selections = {
        sample: {
            "selected_barcode_sha256": f"{index:064x}",
            "eligible_pool_cells": 512,
        }
        for index, sample in enumerate(sample_names, start=1)
    }
    monkeypatch.setattr(combat, "_selected_sample_rows", lambda *args: selections)
    models = {
        name: {
            "kind": "independence",
            "estimator": "fixed-margin conditional independence",
        }
        for name in combat.METHODS
    }
    monkeypatch.setattr(
        combat,
        "_validated_pilot",
        lambda *args, **kwargs: {"frozen_source_models": models},
    )
    permit = {
        "authorization_sha256": "a" * 64,
        "public_authorization_commit": "b" * 40,
        "public_pilot_commit": "c" * 40,
    }
    monkeypatch.setattr(
        combat, "_validated_margin_authorization", lambda *args, **kwargs: permit
    )
    margins = np.tile(np.asarray([[256, 256]], dtype=np.int64), (9, 1))
    predictions = {}
    boundaries = {}
    for name in combat.METHODS:
        flags = []
        table = combat._predict_method(
            models[name], margins, margins, boundary_flags=flags
        )
        predictions[name] = table.reshape(81, 4).tolist()
        boundaries[name] = flags
    rows = [
        {
            "sample": sample,
            "role": roles[sample],
            "rna_margins": margins.tolist(),
            "adt_margins": margins.tolist(),
            "rna_margin_sha256": combat._array_sha256(margins),
            "selected_barcode_sha256": f"{index:064x}",
            "eligible_pool_cells": 512,
            "predictions": json.loads(json.dumps(predictions)),
            "boundary_tilts": json.loads(json.dumps(boundaries)),
        }
        for index, sample in enumerate(sample_names, start=1)
    ]
    attempt = {
        "schema": "combat-citeseq-held-prediction-attempt/1.0",
        "status": "TERMINAL_ATTEMPT_STARTED",
        "created_at_utc": "2026-08-28T00:00:00Z",
        "source_manifest_sha256": source_record["source_manifest_sha256"],
        "pilot_result_sha256": combat._sha256(pilot_path),
        "margin_authorization_sha256": permit["authorization_sha256"],
        "public_margin_authorization_commit": permit["public_authorization_commit"],
        "runner_sha256": combat._sha256(runner),
        "held_margin_request_begins_after_this_record": True,
        "selected_row_csr_structural_scan_authorized": True,
        "held_adt_numeric_data_access_authorized": False,
    }
    attempt_path = defaults["DEFAULT_PREDICTION_ATTEMPT"]
    combat._write_json(attempt_path, attempt)
    prediction = {
        "schema": "combat-citeseq-held-predictions/1.0",
        "status": "FROZEN_HELD_PREDICTIONS",
        "created_at_utc": "2026-08-28T00:00:01Z",
        "source_manifest_sha256": source_record["source_manifest_sha256"],
        "h5ad_sha256": source_record["h5ad_sha256"],
        "pilot_result_sha256": combat._sha256(pilot_path),
        "runner_sha256": combat._sha256(runner),
        "prediction_attempt": {
            "path": "data/confirmation/combat_citeseq/prediction_attempt_v1.json",
            "sha256": combat._sha256(attempt_path),
        },
        "held_rna_margin_authorization": permit,
        "markers": list(combat.MARKERS),
        "cells_per_sample": combat.CELL_BUDGET,
        "samples": rows,
        "access_audit": {
            "process_boundary": "spawned aggregate RNA-margin subprocess",
            "child_output": "aggregate 9x2 RNA margins and digests only",
            "selected_row_csr_structural_access": "indptr and full indices slices",
            "numeric_data_values_decoded": "nine frozen RNA columns only",
            "held_rna_samples_read": 61,
            "held_adt_numeric_data_values_read": 0,
            "held_adt_states_or_margins_formed": 0,
            "held_rna_adt_pairings_formed": 0,
            "held_truth_tables_formed": 0,
            "cell_vectors_serialized": False,
            "adt_margins": "fixed by frozen within-sample 256/256 midrank rule",
        },
    }
    prediction_path = defaults["DEFAULT_PREDICTION"]
    combat._write_json(prediction_path, prediction)
    assert (
        combat._validated_prediction(
            prediction_path, source_path, pilot_path, defaults["DEFAULT_REDUCED"]
        )["status"]
        == "FROZEN_HELD_PREDICTIONS"
    )

    if tamper == "prediction":
        values = prediction["samples"][0]["predictions"]["primary"][0]
        values[0] += 1.0
        values[1] -= 1.0
        values[2] -= 1.0
        values[3] += 1.0
    elif tamper == "boundary":
        prediction["samples"][0]["boundary_tilts"]["primary"] = [{"status": "tampered"}]
    elif tamper == "permit_attempt":
        attempt["margin_authorization_sha256"] = "f" * 64
        combat._write_json(attempt_path, attempt)
        prediction["prediction_attempt"]["sha256"] = combat._sha256(attempt_path)
    elif tamper == "top_truth_extra":
        prediction["truth_tables"] = [[[1, 2], [3, 4]]]
    elif tamper == "row_adt_extra":
        prediction["samples"][0]["actual_adt_margins"] = margins.tolist()
    elif tamper == "audit_mutation":
        prediction["access_audit"]["held_adt_numeric_data_values_read"] = 1
    elif tamper == "audit_omission":
        del prediction["access_audit"]
    elif tamper == "barcode_digest":
        prediction["samples"][0]["selected_barcode_sha256"] = "f" * 64
    else:
        prediction["samples"][0]["eligible_pool_cells"] = 513
    combat._write_json(prediction_path, prediction)
    with pytest.raises(PermissionError):
        combat._validated_prediction(
            prediction_path, source_path, pilot_path, defaults["DEFAULT_REDUCED"]
        )


@pytest.mark.parametrize("failure", ["different", "unavailable"])
def test_public_margin_authorization_failure_precedes_all_held_access(
    tmp_path, monkeypatch, failure
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    runner_path = tmp_path / "experiments/confirm_combat_citeseq.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner\n")
    monkeypatch.setattr(combat, "__file__", str(runner_path))
    protocol = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["protocol"]
    designation = tmp_path / combat.DEVELOPMENT_BINDING_PATHS["designation"]
    for path in (protocol, designation):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n")
    source = defaults["DEFAULT_SOURCE_MANIFEST"]
    pilot = defaults["DEFAULT_PILOT"]
    authorization = defaults["DEFAULT_MARGIN_AUTHORIZATION"]
    for path in (source, pilot, authorization):
        path.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source\n")
    pilot_payload = {
        "reduced_development_sha256": "d" * 64,
        "development_authorization_sha256": "e" * 64,
        "protocol_sha256": combat._sha256(protocol),
        "designation_sha256": combat._sha256(designation),
    }
    pilot.write_text(json.dumps(pilot_payload))
    authorization.write_text(
        json.dumps(
            {
                "schema": "combat-citeseq-held-rna-margin-authorization/1.0",
                "status": "RNA_MARGIN_ACCESS_AUTHORIZED",
                "public_pilot_commit": "b" * 40,
                "runner_sha256": combat._sha256(runner_path),
                "source_manifest_sha256": combat._sha256(source),
                "pilot_result_sha256": combat._sha256(pilot),
                "protocol_sha256": combat._sha256(protocol),
                "designation_sha256": combat._sha256(designation),
                "reduced_development_sha256": "d" * 64,
                "development_authorization_sha256": "e" * 64,
            }
        )
    )
    calls = 0

    def public_bytes(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _PublicBytes(pilot.read_bytes())
        if failure == "unavailable":
            raise OSError("offline")
        return _PublicBytes(b"different authorization")

    monkeypatch.setattr(combat.urllib.request, "urlopen", public_bytes)
    monkeypatch.setattr(combat, "_validated_pilot", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        combat,
        "_validated_source",
        lambda *args, **kwargs: pytest.fail("source opened after failed public auth"),
    )
    monkeypatch.setattr(
        combat,
        "_extract_held_rna_margins",
        lambda *args, **kwargs: pytest.fail("held RNA opened after failed public auth"),
    )

    with pytest.raises(PermissionError):
        combat.predict_held_margins(
            source,
            pilot,
            authorization,
            "a" * 40,
            defaults["DEFAULT_PREDICTION_ATTEMPT"],
            defaults["DEFAULT_PREDICTION"],
        )
    assert calls == 2


def test_invalid_score_authorization_refuses_before_attempt_or_matrix_access(
    tmp_path, monkeypatch
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        combat,
        "_validated_score_authorization",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("disabled")),
    )
    monkeypatch.setattr(
        combat,
        "_read_modality",
        lambda *args, **kwargs: pytest.fail("matrix was accessed before authorization"),
    )
    attempt = defaults["DEFAULT_SCORE_ATTEMPT"]

    with pytest.raises(PermissionError, match="disabled"):
        combat.score_held(
            defaults["DEFAULT_SOURCE_MANIFEST"],
            defaults["DEFAULT_PILOT"],
            defaults["DEFAULT_PREDICTION"],
            defaults["DEFAULT_AUTHORIZATION"],
            "a" * 40,
            attempt,
            defaults["DEFAULT_SCORE"],
        )
    assert not attempt.exists()


@pytest.mark.parametrize("failure", ["different", "unavailable", "wrong_owner"])
def test_public_score_authorization_failure_precedes_held_pairing(
    tmp_path, monkeypatch, failure
):
    defaults = _patch_designated_paths(tmp_path, monkeypatch)
    runner_path = tmp_path / "experiments/confirm_combat_citeseq.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner\n")
    monkeypatch.setattr(combat, "__file__", str(runner_path))
    source = defaults["DEFAULT_SOURCE_MANIFEST"]
    pilot = defaults["DEFAULT_PILOT"]
    prediction = defaults["DEFAULT_PREDICTION"]
    authorization = defaults["DEFAULT_AUTHORIZATION"]
    for path in (source, pilot, prediction, authorization):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n")
    prediction_commit = "b" * 40
    authorization.write_text(
        json.dumps(
            {
                "schema": "combat-citeseq-score-authorization/1.0",
                "status": "OUTCOME_ACCESS_AUTHORIZED",
                "prediction_path": "results/combat_citeseq_predictions.json",
                "prediction_sha256": combat._sha256(prediction),
                "prediction_bytes": prediction.stat().st_size,
                "runner_sha256": combat._sha256(runner_path),
                "source_manifest_sha256": combat._sha256(source),
                "pilot_result_sha256": combat._sha256(pilot),
                "public_prediction_commit": prediction_commit,
                "public_prediction_url": (
                    "https://github.com/"
                    + (
                        "another-owner/coupling-fields-benchmark/blob/"
                        if failure == "wrong_owner"
                        else "sushaan-k/coupling-fields-benchmark/blob/"
                    )
                    + f"{prediction_commit}/results/combat_citeseq_predictions.json"
                ),
            }
        )
    )
    calls = 0

    def public_bytes(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _PublicBytes(prediction.read_bytes())
        if failure == "unavailable":
            raise OSError("offline")
        return _PublicBytes(b"different authorization")

    monkeypatch.setattr(combat.urllib.request, "urlopen", public_bytes)
    monkeypatch.setattr(combat, "_validated_prediction", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        combat,
        "_read_modality",
        lambda *args, **kwargs: pytest.fail(
            "held pairing opened after failed public auth"
        ),
    )
    attempt = defaults["DEFAULT_SCORE_ATTEMPT"]
    with pytest.raises(PermissionError):
        combat.score_held(
            source,
            pilot,
            prediction,
            authorization,
            "a" * 40,
            attempt,
            defaults["DEFAULT_SCORE"],
        )
    assert calls == (0 if failure == "wrong_owner" else 2)
    assert not attempt.exists()


@pytest.mark.parametrize(("sample_count", "favorable"), [(10, 9), (51, 41)])
def test_exact_one_sided_sign_test_uses_fixed_n_and_nonfavorable_zeros(
    sample_count, favorable
):
    values = np.concatenate(
        (
            -np.ones(favorable),
            np.zeros(1),
            np.ones(sample_count - favorable - 1),
        )
    )
    result = combat._exact_sign_test(values)
    expected = sum(
        math.comb(sample_count, count) for count in range(favorable, sample_count + 1)
    ) / (2**sample_count)
    assert result["favorable_samples"] == favorable
    assert result["sample_count"] == sample_count
    assert result["one_sided_p"] == pytest.approx(expected)
    assert result["zeros_are_nonfavorable"] is True
    assert result["one_sided_p"] <= 0.025


def test_st_georges_eight_of_ten_does_not_pass_the_exact_sign_test():
    result = combat._exact_sign_test(np.r_[-np.ones(8), np.ones(2)])
    assert result["one_sided_p"] == pytest.approx(56 / 1024)
    assert result["one_sided_p"] > 0.025


def test_confirmation_requires_field_transfer_and_graph_specific_superiority():
    field_only = {
        "passes_field_transfer": True,
        "supports_graph_specific_superiority": False,
    }
    decision = combat._confirmation_decision(field_only, field_only)
    assert decision["field_transfer_status"] == "FIELD_TRANSFER_PASS"
    assert decision["primary_method_status"] == "PRIMARY_METHOD_FAIL"
    assert decision["status"] == "CONFIRMATION_FAIL"

    complete = {**field_only, "supports_graph_specific_superiority": True}
    decision = combat._confirmation_decision(complete, complete)
    assert decision["primary_method_status"] == "PRIMARY_METHOD_PASS"
    assert decision["status"] == "CONFIRMATION_PASS"


def test_all_bootstrap_comparisons_restart_the_exact_shared_seed(monkeypatch):
    original = np.random.default_rng
    seeds = []

    def recorded(seed):
        seeds.append(seed)
        return original(seed)

    monkeypatch.setattr(combat.np.random, "default_rng", recorded)
    monkeypatch.setattr(combat, "BOOTSTRAPS", 100)
    samples = ("a", "b", "c")
    combat._comparison(
        samples,
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 3.0, 4.0]),
        favorable_required=None,
    )
    combat._comparison(
        samples,
        np.asarray([2.0, 1.0, 3.0]),
        np.asarray([4.0, 2.0, 5.0]),
        favorable_required=None,
    )
    assert seeds == [combat.BOOTSTRAP_SEED, combat.BOOTSTRAP_SEED]
