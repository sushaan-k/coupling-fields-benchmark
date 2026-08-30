from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import evaluate_gse239452_standard_poisson_posthoc as subject
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)


LEGACY_SHA256 = "bc6efbb2ffe3404a294eae26b51e214054718113a8deeaf6b9f4e73ebf05f305"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pooled_poisson_fit_is_the_saturated_loglinear_interaction() -> None:
    first = np.asarray([[4, 6], [3, 7]], dtype=np.int64)
    second = np.asarray([[5, 5], [2, 8]], dtype=np.int64)
    tables = np.broadcast_to(first, (2, 9, 9, 2, 2)).copy()
    tables[1] = second

    interaction, certificate = subject._pooled_poisson_interaction(tables)
    pooled = first + second
    expected = np.log(pooled[0, 0] * pooled[1, 1]) - np.log(pooled[0, 1] * pooled[1, 0])

    np.testing.assert_allclose(interaction, expected, atol=1e-15)
    assert certificate["donors"] == 2
    assert certificate["structural_zeros"] == 0
    assert "Poisson" in certificate["estimator"]


def test_profiled_poisson_reconstruction_matches_margins_and_interaction_not_nch() -> (
    None
):
    rows = np.asarray([10, 14], dtype=np.int64)
    columns = np.asarray([12, 12], dtype=np.int64)
    interaction = 1.2

    poisson = subject._profile_poisson_table(interaction, rows, columns)
    nch = expected_binary_table_from_log_odds(interaction, rows, columns)

    np.testing.assert_allclose(poisson.sum(axis=1), rows, atol=1e-12)
    np.testing.assert_allclose(poisson.sum(axis=0), columns, atol=1e-12)
    reconstructed = np.log(poisson[0, 0] * poisson[1, 1]) - np.log(
        poisson[0, 1] * poisson[1, 0]
    )
    assert reconstructed == pytest.approx(interaction, abs=1e-12)
    assert np.max(np.abs(poisson - nch)) > 1e-3
    independence = subject._profile_poisson_table(0.0, rows, columns)
    np.testing.assert_allclose(independence, np.outer(rows, columns) / rows.sum())


def test_real_development_selection_uses_only_the_frozen_7_plus_8_units() -> None:
    inputs = subject._load_frozen_inputs()
    first = subject._development_analysis(inputs["records"])
    second = subject._development_analysis(inputs["records"])

    assert first == second
    assert first["calibration_donors"] == list(subject.confirmation.CALIBRATION)
    assert first["pilot_donors"] == list(subject.confirmation.PILOT)
    assert first["held_donors_used_for_selection"] == []
    assert first["selected_alpha"] == 1.0
    assert [row["alpha"] for row in first["alpha_candidates"]] == [
        0.5,
        0.75,
        1.0,
        1.25,
    ]
    assert first["refit_certificate"]["donors"] == 15
    assert first["refit_certificate"]["minimum_pooled_cell_count"] == 12
    interaction = np.asarray(first["refit_interaction"], dtype=float)
    for donor in subject.confirmation.HELD:
        frozen = inputs["prediction_by_donor"][donor]
        _, certificate = subject._predict_poisson(
            interaction,
            np.asarray(frozen["row_margins"], dtype=np.int64),
            np.asarray(frozen["column_margins"], dtype=np.int64),
            first["selected_alpha"],
        )
        assert certificate["maximum_absolute_interaction_error"] < 1e-10


def test_real_pooled_saturated_poisson_fit_reconstructs_every_table() -> None:
    inputs = subject._load_frozen_inputs()
    development = np.asarray(
        [
            inputs["records"][donor]["tables"]
            for donor in (
                *subject.confirmation.CALIBRATION,
                *subject.confirmation.PILOT,
            )
        ],
        dtype=np.int64,
    )

    _, certificate = subject._pooled_poisson_interaction(development)

    assert certificate["saturated_tables_reconstructed"] == 81
    assert certificate["maximum_normalized_saturated_cell_error"] <= 1e-10
    assert certificate["normalized_saturated_cell_error_tolerance"] == 1e-10


def test_official_plan_is_nine_individual_pairs_and_fits_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subject,
        "_download",
        lambda *_args, **_kwargs: pytest.fail("plan attempted an assay download"),
    )
    monkeypatch.setattr(
        subject.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=2_000_000_000),
    )
    plan = subject.download_plan(tmp_path / "unused-cache")

    assert plan["download_execution_started"] is False
    assert plan["official_individual_file_pairs"] == 9
    assert [row["donor"] for row in plan["donors"]] == list(subject.confirmation.HELD)
    assert all(len(row["files"]) == 2 for row in plan["donors"])
    assert all(
        file["url"].startswith("https://ftp.ncbi.nlm.nih.gov/geo/samples/")
        for row in plan["donors"]
        for file in row["files"]
    )
    assert all(
        len(file["archive_sha256"]) == 64
        and len(file["h5ad_sha256"]) == 64
        and file["archive_bytes"] > 0
        and file["h5ad_bytes"] > 0
        for row in plan["donors"]
        for file in row["files"]
    )
    assert plan["sequential_maximum_peak_bytes"] == 841_682_025
    assert plan["fits_current_disk"] is True


def test_materializer_verifies_archive_and_h5ad_then_removes_archive(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-h5ad-bytes"
    source_dir = tmp_path / "source"
    destination = tmp_path / "destination"
    source_dir.mkdir()
    destination.mkdir()
    h5ad_name = "donor_GEX.h5ad"
    source_h5ad = source_dir / h5ad_name
    source_h5ad.write_bytes(payload)
    source_archive = source_dir / "donor_GEX.h5ad.tar.gz"
    with tarfile.open(source_archive, "w:gz") as bundle:
        bundle.add(source_h5ad, arcname=h5ad_name)

    spec = {
        "url": source_archive.as_uri(),
        "archive_name": source_archive.name,
        "archive_bytes": source_archive.stat().st_size,
        "archive_sha256": _sha256(source_archive),
        "h5ad_name": h5ad_name,
        "h5ad_bytes": len(payload),
        "h5ad_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output = subject._materialize_modality(spec, destination)

    assert output.read_bytes() == payload
    assert not (destination / source_archive.name).exists()


def test_truth_verification_requires_the_published_hash_and_frozen_axes() -> None:
    inputs = subject._load_frozen_inputs()
    donor = subject.confirmation.HELD[0]
    frozen = inputs["prediction_by_donor"][donor]
    rows = np.asarray(frozen["row_margins"], dtype=np.int64)
    columns = np.asarray(frozen["column_margins"], dtype=np.int64)
    truth = np.empty(rows.shape[:-1] + (2, 2), dtype=np.int64)
    for entity in np.ndindex(rows.shape[:-1]):
        r0 = int(rows[entity][0])
        c0 = int(columns[entity][0])
        total = int(rows[entity].sum())
        lower = max(0, r0 + c0 - total)
        upper = min(r0, c0)
        upper_left = (lower + upper) // 2
        truth[entity] = [
            [upper_left, r0 - upper_left],
            [c0 - upper_left, total - r0 - c0 + upper_left],
        ]
    table_hash = subject.confirmation._array_sha256(truth)
    record = {
        "tables": truth.tolist(),
        "table_sha256": table_hash,
        "selected_barcode_axis_sha256": frozen["selected_barcode_axis_sha256"],
    }
    scored = {"truth_table_sha256": table_hash}

    observed = subject._verify_truth_record(donor, record, frozen, scored)
    np.testing.assert_array_equal(observed, truth)

    scored["truth_table_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="reproduced held truth differs"):
        subject._verify_truth_record(donor, record, frozen, scored)


def test_paired_inference_is_deterministic_and_donor_equal() -> None:
    donors = ["a", "b", "c", "d"]
    primary = np.asarray([1.0, 1.1, 1.2, 1.3])
    poisson = np.asarray([1.5, 1.6, 1.7, 1.8])

    first = subject._comparison(donors, primary, poisson)
    second = subject._comparison(donors, primary, poisson)

    assert first == second
    assert first["bootstrap_draws"] == 20_000
    assert first["favorable_donors_primary_lower"] == 4
    assert first["exact_one_sided_sign_test_p"] == 1 / 16
    assert first["primary_mean_loss"] == pytest.approx(primary.mean())


def test_run_refuses_existing_output_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve me\n")
    monkeypatch.setattr(
        subject,
        "download_plan",
        lambda *_args, **_kwargs: pytest.fail("existing output was not refused first"),
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        subject.run(tmp_path / "cache", output)
    assert output.read_text() == "preserve me\n"


def test_legacy_mislabeled_artifact_remains_byte_identical() -> None:
    assert subject._sha256(subject.LEGACY_AUDIT) == LEGACY_SHA256


def test_published_posthoc_result_is_complete_and_nonconfirmatory() -> None:
    payload = json.loads(subject.DEFAULT_OUTPUT.read_text())
    held = payload["held"]
    comparison = held["comparison"]
    samples = held["samples"]

    assert payload["status"] == "POST_HOC_NONCONFIRMATORY_HEAD_TO_HEAD"
    assert payload["confirmatory"] is False
    assert payload["method"]["noncentral_hypergeometric_reconstruction_used"] is False
    assert payload["bindings"]["runner_sha256"] == _sha256(Path(subject.__file__))
    assert payload["bindings"]["legacy_mislabeled_audit_sha256"] == LEGACY_SHA256
    assert payload["development"]["selected_alpha"] == 1.0
    assert (
        payload["development"]["refit_certificate"][
            "maximum_normalized_saturated_cell_error"
        ]
        <= 1e-10
    )
    assert [row["donor"] for row in samples] == list(subject.confirmation.HELD)
    assert len({row["truth_table_sha256"] for row in samples}) == len(samples) == 9

    primary = np.asarray([row["primary_loss"] for row in samples])
    poisson = np.asarray([row["standard_poisson_loss"] for row in samples])
    assert held["primary_donor_equal_mean_loss"] == pytest.approx(primary.mean())
    assert held["standard_poisson_donor_equal_mean_loss"] == pytest.approx(
        poisson.mean()
    )
    assert np.all(primary < poisson)
    assert comparison["favorable_donors_primary_lower"] == 9
    assert comparison["exact_one_sided_sign_test_p"] == 1 / 512
    assert comparison["paired_difference_95_percentile_ci"][1] < 0.0
    assert comparison["relative_loss_reduction_primary_vs_poisson"] == pytest.approx(
        0.14786492732600487
    )
