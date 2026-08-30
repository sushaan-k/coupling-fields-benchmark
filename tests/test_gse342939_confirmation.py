from __future__ import annotations

import gzip
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from experiments import confirm_gse342939_ra_bcell as confirmation
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)
from mapreg.longitudinal_conditional_coupling import (
    fit_visit_agnostic_conditional_log_odds,
)


def _write_mtx(
    path: Path,
    shape: tuple[int, int],
    entries: list[tuple[int, int, int]],
    *,
    declared: int | None = None,
) -> None:
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        "% synthetic contract fixture",
        f"{shape[0]} {shape[1]} {len(entries) if declared is None else declared}",
        *(f"{row} {column} {value}" for row, column, value in entries),
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def _longitudinal_tables(donors: int = 6) -> np.ndarray:
    values = np.empty(
        (donors, 2, confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2, 2),
        dtype=np.int64,
    )
    values[:, 0] = np.asarray([[32, 16], [16, 32]], dtype=np.int64)
    values[:, 1] = np.asarray([[28, 20], [20, 28]], dtype=np.int64)
    return values


def test_runner_binds_exact_nine_stage_firewall_and_frozen_grids() -> None:
    amendment = confirmation._amendment()
    clarification = confirmation._streaming_clarification()
    assert amendment["access_firewall"]["future_cli_stages"] == [
        "claim-source",
        "run-source",
        "claim-held-rna",
        "run-held-rna",
        "claim-held-adt",
        "run-held-adt",
        "predict-held",
        "authorize-score",
        "score-held",
    ]
    assert confirmation.HETEROGENEITY_GRID == (0.1, 1.0, 10.0)
    assert confirmation.RIDGE_GRID == (0.01, 0.1)
    assert confirmation.GRAPH_GRID == (0.0, 0.03, 0.3)
    assert confirmation.TRANSPORT_GRID == (0.0, 0.5, 1.0, 1.5)
    assert confirmation.GRAPH_NEIGHBORS == 2
    assert clarification["scientific_contract_changed"] is False
    assert (
        clarification["streaming_reduction_contract"][
            "maximum_packed_detection_bitset_bytes"
        ]
        == confirmation.MAXIMUM_PACKED_DETECTION_BITSET_BYTES
    )
    runner_source = inspect.getsource(confirmation)
    assert "import sqlite3" not in runner_source
    assert "temporary_sqlite_without_rowid" not in runner_source


def test_matrix_market_stream_validates_unsorted_duplicates_without_densifying(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.mtx.gz"
    _write_mtx(path, (3, 4), [(2, 1, 3), (1, 1, 2), (3, 4, 9), (1, 1, 5)])
    with confirmation._matrix_entries(path, (3, 4)) as (entries, audit):
        observed = list(entries)
    assert observed == [(1, 0, 3), (0, 0, 2), (2, 3, 9), (0, 0, 5)]
    assert audit["raw_entries_seen"] == 4
    assert audit["raw_entries_yielded"] == 4
    assert audit["maximum_raw_count"] == 9
    assert audit["parse_completed"] is True
    assert audit["entry_iteration_completed"] is True


def test_matrix_market_stream_refuses_incomplete_bodies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.mtx.gz"
    _write_mtx(path, (3, 3), [(1, 1, 1)], declared=2)
    audit: dict[str, object] = {}
    with pytest.raises(confirmation.ProtocolRefusal, match="DECLARED_ENTRY_COUNT"):
        with confirmation._matrix_entries(path, (3, 3), audit) as (stream, _):
            list(stream)
    assert audit["parser_started"] is True
    assert audit["expected_shape"] == [3, 3]
    assert audit["raw_entries_seen"] == 1
    assert audit["parse_completed"] is False
    assert audit["entry_iteration_completed"] is False


def test_matrix_market_axis_shape_is_exact_and_transpose_is_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transposed.mtx.gz"
    _write_mtx(path, (4, 3), [(1, 1, 1)])
    with pytest.raises(confirmation.ProtocolRefusal, match="AXIS_DIMENSION"):
        with confirmation._matrix_entries(path, (3, 4)) as (stream, _):
            list(stream)


def test_matrix_market_parser_has_a_constant_memory_certificate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scaled-unsorted.mtx.gz"
    unique = [
        (row + 1, column + 1, 1)
        for row in range(250)
        for column in range(100)
    ]
    entries = [*unique, *[(row, column, 2) for row, column, _ in unique[:5_000]]]
    np.random.default_rng(17).shuffle(entries)
    _write_mtx(path, (250, 100), entries)
    with confirmation._matrix_entries(path, (250, 100)) as (stream, audit):
        observed = list(stream)
    assert len(observed) == audit["raw_entries_yielded"] == 30_000
    assert audit["peak_in_memory_entry_batch"] == 1
    assert audit["parser_backend"] == "validated_single_pass_gzip_stream"
    assert audit["temporary_storage_bytes"] == 0
    assert not list(tmp_path.glob(".gse342939-mtx-*.sqlite"))


def test_matrix_market_total_certificate_covers_unselected_duplicate_overflow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overflow.mtx.gz"
    _write_mtx(
        path,
        (3, 3),
        [(3, 3, int(np.iinfo(np.int64).max)), (3, 3, 1)],
    )
    audit: dict[str, object] = {}
    with pytest.raises(
        confirmation.ProtocolRefusal, match="TOTAL_EXCEEDS_INT64_CERTIFICATE"
    ):
        with confirmation._matrix_entries(path, (3, 3), audit) as (stream, _):
            list(stream)
    assert audit["raw_entries_seen"] == 2
    assert audit["matrix_total_count"] == int(np.iinfo(np.int64).max) + 1
    assert audit["matrix_total_within_int64"] is False
    assert audit["all_body_entries_validated"] is False


def test_matrix_market_refuses_success_without_exhausting_gzip_body(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-exhausted.mtx.gz"
    _write_mtx(path, (3, 3), [(1, 1, 1), (2, 2, 1)])
    audit: dict[str, object] = {}
    with pytest.raises(confirmation.ProtocolRefusal, match="ITERATION_INCOMPLETE"):
        with confirmation._matrix_entries(path, (3, 3), audit) as (stream, _):
            next(stream)
    assert audit["entry_iteration_completed"] is False
    assert audit["gzip_eof_verified"] is False


def test_truncated_gzip_never_receives_a_completion_certificate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truncated.mtx.gz"
    _write_mtx(path, (3, 3), [(1, 1, 1), (2, 2, 1)])
    path.write_bytes(path.read_bytes()[:-8])
    audit: dict[str, object] = {}
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        with confirmation._matrix_entries(path, (3, 3), audit) as (stream, _):
            list(stream)
    assert audit["parse_completed"] is False
    assert audit["gzip_eof_verified"] is False


def test_gex_reduction_counts_unique_detected_genes_and_sums_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(confirmation, "MINIMUM_CELLS", 2)
    monkeypatch.setattr(confirmation, "MAXIMUM_CELLS", 2)
    monkeypatch.setattr(confirmation, "MINIMUM_DETECTED_GENES", 2)
    features = [f"ENSG{row}\tG{row}\tGene Expression" for row in range(45)]
    barcodes = [f"cell{column}-1" for column in range(4)]
    axes = {
        "gex_features": features,
        "gex_barcodes": barcodes,
        "gex_normalized_barcodes": [f"cell{column}" for column in range(4)],
        "intersection": [f"cell{column}" for column in range(4)],
        "rna_rows": np.arange(45, dtype=np.int64),
        "gene_expression_rows": np.ones(45, dtype=bool),
    }
    entries = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 2, 2),
        (1, 3, 1),
        (2, 3, 1),
        (1, 4, 1),
        (2, 4, 1),
    ]
    np.random.default_rng(4).shuffle(entries)
    path = tmp_path / "gex.mtx.gz"
    _write_mtx(path, (45, 4), entries)
    audit: dict[str, object] = {}
    reduced = confirmation._reduce_gex_matrix(
        path, "D", "pre", axes, audit
    )
    assert reduced["eligible_barcodes"] == 2
    assert reduced["barcodes"] == ["cell2", "cell3"]
    assert reduced["states"][:, :2].tolist() == [[1, 1], [1, 1]]
    assert audit["reduction_duplicate_coordinates"] == 2
    assert audit["temporary_storage_bytes"] == 0
    assert confirmation._complete_matrix_audit(audit, (45, 4), "gex")
    assert not confirmation._complete_matrix_audit(audit, (45, 4), "cite")


def test_cite_reduction_sums_unsorted_selected_coordinate_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(confirmation, "MINIMUM_CELLS", 2)
    axes = {
        "cite_features": [f"ADT{row}" for row in range(45)],
        "cite_barcodes": ["cell0", "cell1", "unused"],
        "adt_rows": np.arange(45, dtype=np.int64),
    }
    entries = [(1, 1, 1), (1, 2, 2), (1, 1, 2), (2, 2, 4)]
    np.random.default_rng(8).shuffle(entries)
    path = tmp_path / "cite.mtx.gz"
    _write_mtx(path, (45, 3), entries)
    audit: dict[str, object] = {}
    reduced = confirmation._reduce_cite_matrix(
        path, "D", "pre", axes, ["cell0", "cell1"], audit
    )
    expected_counts = np.zeros((2, confirmation.MARKER_COUNT), dtype=np.int64)
    expected_counts[0, 0] = 3
    expected_counts[1, 0] = 2
    expected_counts[1, 1] = 4
    assert reduced["count_panel_sha256"] == confirmation._array_sha256(
        expected_counts
    )
    assert audit["reduction_duplicate_coordinates"] == 1
    assert audit["temporary_storage_bytes"] == 0
    assert confirmation._complete_matrix_audit(audit, (45, 3), "cite")
    assert not confirmation._complete_matrix_audit(audit, (45, 3), "gex")

    for key, bad_value in (
        ("gzip_eof_verified", False),
        ("matrix_total_within_int64", False),
        ("temporary_database_created", True),
        ("reduction_completed", False),
        ("selected_coordinate_seen_bytes", 0),
    ):
        corrupted = dict(audit)
        corrupted[key] = bad_value
        assert not confirmation._complete_matrix_audit(corrupted, (45, 3), "cite")


def test_scratch_capacity_is_checked_without_disclosing_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrices = [{"expected_bytes": 10}]
    certificate = confirmation._scratch_capacity_certificate(
        tmp_path / "scratch", matrices
    )
    assert certificate["scratch_path_disclosed"] is False
    assert certificate["temporary_coordinate_store_permitted"] is False
    assert confirmation._valid_scratch_capacity_certificate(certificate, matrices)
    confirmation._require_scratch_capacity_certificate(
        certificate, matrices, "synthetic"
    )
    with pytest.raises(PermissionError, match="scratch capacity"):
        confirmation._require_scratch_capacity_certificate(None, matrices, "synthetic")
    corrupted = dict(certificate)
    corrupted["temporary_coordinate_store_permitted"] = True
    with pytest.raises(PermissionError, match="scratch capacity"):
        confirmation._require_scratch_capacity_certificate(
            corrupted, matrices, "synthetic"
        )

    empty = tmp_path / "too-small"
    monkeypatch.setattr(
        confirmation.shutil,
        "disk_usage",
        lambda _path: type(
            "Usage",
            (),
            {"free": confirmation.MINIMUM_SCRATCH_FREE_BYTES - 1},
        )(),
    )
    with pytest.raises(confirmation.ProtocolRefusal, match="SCRATCH_CAPACITY"):
        confirmation._scratch_capacity_certificate(empty, matrices)


def test_failed_fetch_records_final_url_and_hashes_every_read_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_url = "https://ftp.ncbi.nlm.nih.gov/frozen.mtx.gz"

    class Response:
        def __init__(self, final_url: str, blocks: list[bytes]) -> None:
            self.final_url = final_url
            self.blocks = iter(blocks)

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return self.final_url

        def read(self, _size: int) -> bytes:
            return next(self.blocks, b"")

    class Opener:
        def __init__(self, response: Response) -> None:
            self.response = response

        def open(self, *_args: object, **_kwargs: object) -> Response:
            return self.response

    redirected: dict[str, object] = {}
    monkeypatch.setattr(
        confirmation.urllib.request,
        "build_opener",
        lambda *_: Opener(Response("https://example.org/redirected.mtx.gz", [])),
    )
    with pytest.raises(PermissionError):
        confirmation._fetch_matrix(
            {"url": expected_url, "expected_bytes": 3}, tmp_path, redirected
        )
    assert redirected["final_url"] == "https://example.org/redirected.mtx.gz"
    assert redirected["deleted"] is True

    oversized: dict[str, object] = {}
    block = b"four"
    monkeypatch.setattr(
        confirmation.urllib.request,
        "build_opener",
        lambda *_: Opener(Response(expected_url, [block])),
    )
    with pytest.raises(confirmation.ProtocolRefusal, match="BYTE_COUNT_EXCEEDED"):
        confirmation._fetch_matrix(
            {"url": expected_url, "expected_bytes": 3}, tmp_path, oversized
        )
    assert oversized["observed_bytes"] == len(block)
    assert oversized["hashed_bytes"] == len(block)
    assert oversized["partial_sha256"] == hashlib.sha256(block).hexdigest()
    assert oversized["deleted"] is True


def test_true_fixed_interaction_poisson_has_exact_margins_and_is_not_nch() -> None:
    source = np.asarray([[50.0, 20.0], [20.0, 50.0]])
    log_odds = confirmation._table_log_odds(source)
    rows = np.asarray([70.0, 30.0])
    columns = np.asarray([40.0, 60.0])
    direct, certificate = confirmation._fixed_interaction_table(
        log_odds, rows, columns
    )
    conditional = expected_binary_table_from_log_odds(log_odds, rows, columns)
    np.testing.assert_allclose(direct.sum(axis=1), rows, atol=1e-10)
    np.testing.assert_allclose(direct.sum(axis=0), columns, atol=1e-10)
    assert certificate["absolute_log_odds_error"] <= 1e-8
    assert not np.allclose(direct, conditional, rtol=0.0, atol=1e-4)


def test_poisson_zero_transport_is_independence_and_never_uses_nch() -> None:
    rows = np.broadcast_to(
        np.asarray([70.0, 30.0]),
        (2, confirmation.MARKER_COUNT, confirmation.MARKER_COUNT, 2),
    ).copy()
    columns = np.broadcast_to(
        np.asarray([40.0, 60.0]), rows.shape
    ).copy()
    mask = np.zeros(rows.shape[:-1], dtype=bool)
    mask[:, 0, 0] = True
    model = {
        "population_mean": np.ones((confirmation.MARKER_COUNT,) * 2),
        "population_change": np.ones((confirmation.MARKER_COUNT,) * 2),
    }
    prediction, certificate = confirmation._predict_poisson(
        model,
        rows,
        columns,
        confirmation.TransportConfig(0.0, 0.0),
        mask,
    )
    np.testing.assert_allclose(
        prediction, confirmation._independence(rows, columns), atol=1e-10
    )
    assert certificate["conditional_noncentral_hypergeometric_reconstruction"] is False


def test_fold_mask_uses_six_training_donors_and_validation_margins_only() -> None:
    training = _longitudinal_tables()
    first_truth = np.empty(training.shape[1:], dtype=np.int64)
    first_truth[0] = np.asarray([[32, 16], [16, 32]])
    first_truth[1] = np.asarray([[28, 20], [20, 28]])
    second_truth = np.flip(first_truth, axis=-1).copy()
    first_mask, first_scored, first_audit = confirmation._fold_mask(
        training, first_truth.sum(axis=-1), first_truth.sum(axis=-2)
    )
    second_mask, second_scored, second_audit = confirmation._fold_mask(
        training, second_truth.sum(axis=-1), second_truth.sum(axis=-2)
    )
    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_array_equal(first_scored, second_scored)
    assert first_audit["training_donors"] == 6
    assert first_audit["validation_association_used_for_mask"] is False
    assert first_audit["scored_coordinate_counts"] == [2025, 2025]
    assert first_audit["scored_mask_sha256"] == second_audit["scored_mask_sha256"]


def test_source_and_held_gate_requirements_are_donor_blocked_and_exact() -> None:
    source_primary = np.full(7, 0.80)
    source_control = np.full(7, 1.00)
    for label in ("residual", "pooled_poisson", "destroyed"):
        result = confirmation._comparison(
            source_primary, source_control, label, "source"
        )
        assert result["passes_frozen_requirement"]
        assert result["favorable_physical_donors"] == 7
    held = confirmation._held_comparison(
        np.full(6, 0.80), np.full(6, 1.00), "residual", True
    )
    assert held["passes_frozen_confirmation_requirement"]
    assert held["exact_one_sided_sign_test_p"] == 1 / 64
    failed = confirmation._held_comparison(
        np.asarray([0.8, 0.8, 0.8, 0.8, 0.8, 1.2]),
        np.ones(6),
        "residual",
        True,
    )
    assert not failed["passes_frozen_confirmation_requirement"]


def test_published_candidate_selection_recomputes_completeness_and_mean() -> None:
    valid = {
        "configuration": {"name": "valid"},
        "complete": True,
        "fold_losses": [0.8] * 7,
        "mean_physical_donor_loss": 0.8,
    }
    configuration, losses = confirmation._selected_published_candidate([valid])
    assert configuration == {"name": "valid"}
    np.testing.assert_array_equal(losses, np.full(7, 0.8))

    wrong_mean = {**valid, "mean_physical_donor_loss": 0.7}
    with pytest.raises(PermissionError, match="mean loss does not recompute"):
        confirmation._selected_published_candidate([wrong_mean])

    wrong_completeness = {**valid, "complete": False}
    with pytest.raises(PermissionError, match="completeness flag is inconsistent"):
        confirmation._selected_published_candidate([wrong_completeness])


def test_published_source_gate_replays_grids_losses_masks_and_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donors = [f"D{index}" for index in range(7)]
    candidate = {"source_donors": []}
    urls = []
    for donor in donors:
        visits = []
        for visit in confirmation.VISITS:
            assays = {}
            for assay in ("gex", "cite"):
                url = f"https://ftp.ncbi.nlm.nih.gov/{donor}-{visit}-{assay}.mtx.gz"
                urls.append(url)
                assays[assay] = {"matrix": {"url": url, "expected_bytes": 10}}
            visits.append(assays)
        candidate["source_donors"].append({"donor": donor, "visits": visits})
    monkeypatch.setattr(confirmation, "_candidate", lambda: candidate)
    monkeypatch.setattr(confirmation, "_binding_hashes", lambda: {"runner": "hash"})
    monkeypatch.setattr(
        confirmation,
        "_validate_mixed_download_provenance",
        lambda _observed, _designated, _cohort: None,
    )

    primary_configs = [
        confirmation.asdict(
            confirmation.PrimaryConfig(heterogeneity, ridge, graph, baseline, change)
        )
        for heterogeneity, ridge, graph, baseline, change in product(
            confirmation.HETEROGENEITY_GRID,
            confirmation.RIDGE_GRID,
            confirmation.GRAPH_GRID,
            confirmation.TRANSPORT_GRID,
            confirmation.TRANSPORT_GRID,
        )
    ]
    agnostic_configs = [
        confirmation.asdict(
            confirmation.PrimaryConfig(heterogeneity, ridge, graph, baseline, 0.0)
        )
        for heterogeneity, ridge, graph, baseline in product(
            confirmation.HETEROGENEITY_GRID,
            confirmation.RIDGE_GRID,
            confirmation.GRAPH_GRID,
            confirmation.TRANSPORT_GRID,
        )
    ]
    residual_configs = [
        confirmation.asdict(confirmation.ResidualConfig(family, baseline, change))
        for family, baseline, change in product(
            confirmation.RESIDUAL_FAMILIES,
            confirmation.TRANSPORT_GRID,
            confirmation.TRANSPORT_GRID,
        )
    ]
    transport_configs = [
        confirmation.asdict(confirmation.TransportConfig(baseline, change))
        for baseline, change in product(
            confirmation.TRANSPORT_GRID, confirmation.TRANSPORT_GRID
        )
    ]

    def evaluations(configurations: list[dict], selected_loss: float) -> list[dict]:
        return [
            {
                "configuration": configuration,
                "complete": True,
                "fold_losses": [selected_loss if index == 0 else 1.4] * 7,
                "mean_physical_donor_loss": selected_loss if index == 0 else 1.4,
            }
            for index, configuration in enumerate(configurations)
        ]

    primary = np.full(7, 0.8)
    control = np.ones(7)
    mask = np.ones(
        (confirmation.MARKER_COUNT, confirmation.MARKER_COUNT), dtype=np.uint8
    )
    zeros = [0.0] * (confirmation.MARKER_COUNT**2)

    def model(kind: str, configuration: dict) -> dict:
        return {
            "kind": kind,
            "configuration": configuration,
            "population_mean": zeros,
            "population_change": zeros,
            "fit_certificate": {"passes": True},
        }

    source_files = [
        {
            "requested_url": url,
            "expected_bytes": 10,
            "observed_bytes": 10,
            "sha256": "1" * 64,
            "completed": True,
            "deleted": True,
            "reduction_completed": True,
        }
        for url in urls
    ]
    source = {
        "schema": "gse342939-ra-bcell-source-result/1.0",
        "status": "SOURCE_PROMOTION_PASS",
        "candidate_sha256": confirmation.CANDIDATE_SHA256,
        "manifest_sha256": confirmation.MANIFEST_SHA256,
        "amendment_sha256": confirmation.AMENDMENT_SHA256,
        "implementation_bindings": {"runner": "hash"},
        "passes_source_promotion_gate": True,
        "held_numeric_access_authorized": True,
        "rerun_permitted": False,
        "source_files": source_files,
        "access_audit": {
            "source_files": source_files,
            "held_numeric_urls_requested": 0,
            "raw_tar_or_bcr_urls_requested": 0,
            "scratch_capacity_before_consumption": {
                "scratch_path_disclosed": False,
                "scratch_empty_at_check": True,
                "filesystem_free_bytes": confirmation.MINIMUM_SCRATCH_FREE_BYTES,
                "maximum_stage_matrix_bytes": 10,
                "maximum_campaign_matrix_bytes": (
                    confirmation.MAXIMUM_COMPRESSED_MATRIX_BYTES
                ),
                "maximum_packed_detection_bitset_bytes": (
                    confirmation.MAXIMUM_PACKED_DETECTION_BITSET_BYTES
                ),
                "minimum_free_bytes": confirmation.MINIMUM_SCRATCH_FREE_BYTES,
                "temporary_coordinate_store_permitted": False,
                "passes": True,
            },
        },
        "candidate_evaluations": {
            "primary": evaluations(primary_configs, 0.8),
            "visit_agnostic_primary": evaluations(agnostic_configs, 0.9),
            "residual": evaluations(residual_configs, 1.0),
            "common_effect": evaluations(transport_configs, 1.0),
            "pooled_poisson": evaluations(transport_configs, 1.0),
        },
        "selected_primary": primary_configs[0],
        "selected_visit_agnostic_primary": agnostic_configs[0],
        "selected_residual": residual_configs[0],
        "selected_common_effect": transport_configs[0],
        "selected_pooled_poisson": transport_configs[0],
        "losses": {
            "primary": primary.tolist(),
            "visit_agnostic_primary": np.full(7, 0.9).tolist(),
            "selected_residual": control.tolist(),
            "common_effect_cmle": control.tolist(),
            "pooled_saturated_poisson": control.tolist(),
            "destroyed_link": control.tolist(),
            "independence": control.tolist(),
        },
        "comparisons": {
            "selected_residual": confirmation._comparison(
                primary, control, "residual", "source"
            ),
            "pooled_saturated_poisson": confirmation._comparison(
                primary, control, "pooled_poisson", "source"
            ),
            "destroyed_link": confirmation._comparison(
                primary, control, "destroyed", "source"
            ),
            "common_effect_cmle": confirmation._comparison(
                primary, control, "common_effect_cmle", "source"
            ),
            "independence": confirmation._comparison(
                primary, control, "independence", "source"
            ),
        },
        "models": {
            "final_mask": mask.tolist(),
            "final_mask_sha256": confirmation._array_sha256(mask),
            "primary": model(
                "paired_longitudinal_exact_conditional_coupling_field",
                primary_configs[0],
            ),
            "selected_residual": model(
                "visit_aware_raw_signed_residual", residual_configs[0]
            ),
            "common_effect_cmle": model(
                "visit_specific_exact_conditional_common_log_odds",
                transport_configs[0],
            ),
            "pooled_saturated_poisson": model(
                "visit_specific_pooled_saturated_poisson_fixed_interaction",
                transport_configs[0],
            ),
            "destroyed_link": model(
                "destroyed_paired_longitudinal_exact_conditional_coupling_field",
                primary_configs[0],
            ),
            "visit_agnostic_primary": model(
                "visit_agnostic_exact_conditional_coupling_field",
                agnostic_configs[0],
            ),
            "independence": {"kind": "recipient_margin_independence"},
        },
        "fold_masks": {
            held: {
                "training_donors": [donor for donor in donors if donor != held],
                "validation_donor": held,
                "validation_association_used_for_mask": False,
                "scored_coordinate_counts": [2025, 2025],
            }
            for held in donors
        },
    }
    source["models"]["pooled_saturated_poisson"]["fit_certificate"].update(
        {"conditional_noncentral_hypergeometric_reconstruction": False}
    )
    confirmation._validate_source_pass_payload(source)
    capacity = source["access_audit"].pop("scratch_capacity_before_consumption")
    with pytest.raises(PermissionError, match="scratch capacity"):
        confirmation._validate_source_pass_payload(source)
    source["access_audit"]["scratch_capacity_before_consumption"] = capacity
    source["fold_masks"]["D0"]["training_donors"] = donors[1:-1]
    with pytest.raises(PermissionError, match="training-donor-only"):
        confirmation._validate_source_pass_payload(source)


def test_visit_agnostic_ablation_constrains_every_change_field_to_zero() -> None:
    tables = np.empty((3, 2, 1, 1, 2, 2), dtype=np.int64)
    tables[:, 0, 0, 0] = np.asarray([[26, 14], [14, 26]])
    tables[:, 1, 0, 0] = np.asarray([[16, 24], [24, 16]])
    fit = fit_visit_agnostic_conditional_log_odds(
        tables,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        heterogeneity_penalty=1.0,
        population_ridge=0.1,
    )
    np.testing.assert_array_equal(fit.population_change, np.zeros((1, 1)))
    np.testing.assert_array_equal(
        fit.donor_change_deviation, np.zeros((3, 1, 1))
    )
    np.testing.assert_allclose(fit.donor_log_odds[:, 0], fit.donor_log_odds[:, 1])


def test_source_failure_is_terminal_and_preserves_failing_file_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "source-attempt.json"
    consumption = tmp_path / "source-consumption.json"
    output = tmp_path / "source-result.json"
    attempt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(confirmation, "SOURCE_ATTEMPT", attempt)
    monkeypatch.setattr(confirmation, "SOURCE_CONSUMPTION", consumption)
    monkeypatch.setattr(confirmation, "SOURCE_RESULT", output)
    monkeypatch.setattr(
        confirmation,
        "_validate_source_attempt",
        lambda *_: ({}, {"runtime": {"synthetic": True}}, "a" * 40),
    )
    monkeypatch.setattr(confirmation, "_binding_hashes", lambda: {})

    def refuse(_candidate, _axis_cache, _scratch, audit):
        audit["source_files"].append(
            {
                "requested_url": "https://ftp.ncbi.nlm.nih.gov/failing.mtx.gz",
                "request_started": True,
                "observed_bytes": 17,
                "hashed_bytes": 17,
                "partial_sha256": "1" * 64,
                "deleted": True,
                "matrix_market": {
                    "parser_started": True,
                    "expected_shape": [3, 3],
                    "raw_entries_seen": 1,
                    "parse_completed": False,
                    "entry_iteration_completed": False,
                },
            }
        )
        raise confirmation.ProtocolRefusal("SYNTHETIC_SOURCE_REDUCTION_FAILURE")

    monkeypatch.setattr(confirmation, "_read_source_records", refuse)
    scratch = tmp_path / "scratch"
    result = confirmation.run_source(attempt, output, tmp_path / "axes", scratch)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result == persisted
    assert result["status"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert result["held_numeric_access_authorized"] is False
    assert result["access_audit"]["held_numeric_urls_requested"] == 0
    assert result["source_files"][0]["observed_bytes"] == 17
    assert result["source_files"][0]["matrix_market"]["parse_completed"] is False
    assert consumption.exists()
    with pytest.raises(FileExistsError):
        confirmation.run_source(attempt, output, tmp_path / "axes", scratch)


def test_source_attempt_revalidates_the_complete_public_tag_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_path = tmp_path / "source-attempt.json"
    matrices = [
        {"url": f"https://ftp.ncbi.nlm.nih.gov/{index}.mtx.gz", "expected_bytes": 10}
        for index in range(28)
    ]
    iterator = iter(matrices)
    candidate = {
        "source_donors": [
            {
                "visits": [
                    {
                        "gex": {"matrix": next(iterator)},
                        "cite": {"matrix": next(iterator)},
                    }
                    for _visit in confirmation.VISITS
                ]
            }
            for _donor in range(7)
        ]
    }
    public_tags = {
        "candidate_commit": "1" * 40,
        "amendment_commit": "2" * 40,
        "implementation_commit": "3" * 40,
    }
    payload = {
        "schema": "gse342939-ra-bcell-source-attempt/1.0",
        "status": "CLAIMED_BEFORE_FIRST_NUMERIC_MATRIX_GET",
        "candidate_sha256": confirmation.CANDIDATE_SHA256,
        "manifest_sha256": confirmation.MANIFEST_SHA256,
        "amendment_sha256": confirmation.AMENDMENT_SHA256,
        "implementation_bindings": {"runner": "hash"},
        "public_tags": public_tags,
        "runtime": {"runtime": "frozen"},
        "axis_cache_certificate": {"axes": "frozen"},
        "scratch_capacity_certificate": {
            "scratch_path_disclosed": False,
            "scratch_empty_at_check": True,
            "filesystem_free_bytes": confirmation.MINIMUM_SCRATCH_FREE_BYTES,
            "maximum_stage_matrix_bytes": 10,
            "maximum_campaign_matrix_bytes": (
                confirmation.MAXIMUM_COMPRESSED_MATRIX_BYTES
            ),
            "maximum_packed_detection_bitset_bytes": (
                confirmation.MAXIMUM_PACKED_DETECTION_BITSET_BYTES
            ),
            "minimum_free_bytes": confirmation.MINIMUM_SCRATCH_FREE_BYTES,
            "temporary_coordinate_store_permitted": False,
            "passes": True,
        },
        "source_numeric_file_count": 28,
        "source_numeric_urls": [record["url"] for record in matrices],
        "source_numeric_expected_bytes": 280,
        "held_numeric_access_authorized": False,
        "rerun_permitted": False,
    }
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(confirmation, "SOURCE_ATTEMPT", attempt_path)
    monkeypatch.setattr(confirmation, "_candidate", lambda: candidate)
    monkeypatch.setattr(confirmation, "_amendment", lambda: {})
    monkeypatch.setattr(confirmation, "_verify_public_freezes", lambda: public_tags)
    monkeypatch.setattr(confirmation, "_binding_hashes", lambda: {"runner": "hash"})
    monkeypatch.setattr(
        confirmation, "_runtime_record", lambda: {"runtime": "frozen"}
    )
    monkeypatch.setattr(
        confirmation, "_validate_axis_cache", lambda _path: {"axes": "frozen"}
    )
    monkeypatch.setattr(confirmation, "_require_public_tag", lambda *_: "4" * 40)
    monkeypatch.setattr(confirmation, "_require_ancestor", lambda *_: None)
    monkeypatch.setattr(confirmation, "_relative", lambda _path: "attempt.json")
    _, observed, commit = confirmation._validate_source_attempt(
        attempt_path, tmp_path
    )
    assert observed == payload
    assert commit == "4" * 40

    payload["public_tags"] = {**public_tags, "candidate_commit": "9" * 40}
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="differs from the frozen"):
        confirmation._validate_source_attempt(attempt_path, tmp_path)


def test_stage_functions_reject_noncanonical_artifact_paths(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="canonical"):
        confirmation.claim_source(tmp_path / "attempt.json", tmp_path)
    with pytest.raises(PermissionError, match="canonical"):
        confirmation.run_source(
            confirmation.SOURCE_ATTEMPT,
            tmp_path / "result.json",
            tmp_path,
            tmp_path,
        )
    with pytest.raises(PermissionError, match="canonical"):
        confirmation.claim_held_adt(tmp_path / "held-adt.json", tmp_path)
    with pytest.raises(PermissionError, match="canonical"):
        confirmation.run_prediction(tmp_path / "prediction.json")
    with pytest.raises(PermissionError, match="canonical"):
        confirmation.score_held(
            tmp_path / "score-attempt.json",
            tmp_path / "score-result.json",
        )


def test_held_reducers_and_prediction_preserve_modality_firewalls() -> None:
    rna_source = inspect.getsource(confirmation.run_held_rna)
    adt_source = inspect.getsource(confirmation.run_held_adt)
    prediction_source = inspect.getsource(confirmation.run_prediction)
    assert "_joint_tables" not in rna_source
    assert "_joint_tables" not in adt_source
    assert "private_rna" not in inspect.signature(confirmation.run_held_adt).parameters
    assert "_sha256(PRIVATE_RNA)" not in adt_source
    assert "np.load(PRIVATE_RNA" not in adt_source
    assert "PRIVATE_RNA" not in prediction_source
    assert "PRIVATE_ADT" not in prediction_source
    assert "_joint_tables" not in prediction_source


def test_held_download_provenance_uses_emitted_stage_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        confirmation,
        "_validate_mixed_download_provenance",
        lambda _records, _designated, cohort: observed.append(cohort),
    )
    confirmation._validate_download_provenance([], [], "gex")
    confirmation._validate_download_provenance([], [], "cite")
    assert observed == ["held_rna", "held_adt"]


def test_network_disabled_context_blocks_and_then_restores_openers() -> None:
    original = confirmation.urllib.request.build_opener
    with confirmation._network_disabled():
        with pytest.raises(PermissionError, match="disabled"):
            confirmation.urllib.request.build_opener()
    assert confirmation.urllib.request.build_opener is original


def test_private_artifacts_are_exclusive_atomic_and_owner_only(tmp_path: Path) -> None:
    json_path = tmp_path / "private.json"
    npz_path = tmp_path / "private.npz"
    confirmation._exclusive_private_json(json_path, {"value": 1})
    confirmation._exclusive_private_npz(npz_path, {"x": np.arange(4)})
    assert stat.S_IMODE(json_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(npz_path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        confirmation._exclusive_private_json(json_path, {"value": 2})
    assert json.loads(json_path.read_text()) == {"value": 1}


def test_score_attempt_is_published_before_any_private_state_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "score-attempt.json"
    result = tmp_path / "score-result.json"
    rna = tmp_path / "rna.npz"
    adt = tmp_path / "adt.npz"
    rna.write_bytes(b"sealed-rna")
    adt.write_bytes(b"sealed-adt")
    authorization = {
        "private_rna_states_sha256": "1" * 64,
        "private_adt_states_sha256": "2" * 64,
    }
    monkeypatch.setattr(confirmation, "SCORE_ATTEMPT", attempt)
    monkeypatch.setattr(confirmation, "SCORE_RESULT", result)
    monkeypatch.setattr(confirmation, "PRIVATE_RNA", rna)
    monkeypatch.setattr(confirmation, "PRIVATE_ADT", adt)
    monkeypatch.setattr(confirmation, "SCORE_AUTHORIZATION", tmp_path / "auth.json")
    monkeypatch.setattr(confirmation, "HELD_PREDICTIONS", tmp_path / "prediction.json")
    monkeypatch.setattr(confirmation, "HELD_MARGINS", tmp_path / "margins.json")
    for path in (
        confirmation.SCORE_AUTHORIZATION,
        confirmation.HELD_PREDICTIONS,
        confirmation.HELD_MARGINS,
    ):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        confirmation,
        "_validated_score_authorization",
        lambda: (authorization, "a" * 40, {"held_records": []}, {"donors": {}}),
    )

    def private_read(*_args, **_kwargs):
        assert attempt.exists()
        raise confirmation.ProtocolRefusal("SYNTHETIC_PRIVATE_READ_FAILURE")

    monkeypatch.setattr(confirmation, "_load_private_states", private_read)
    output = confirmation.score_held(attempt, result, rna, adt)
    assert attempt.exists()
    assert output["status"] == "TERMINAL_HELD_SCORE_EXECUTION_REFUSAL"
    assert result.exists()
