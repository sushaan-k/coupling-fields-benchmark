"""Post-access correction of the GSE239452 residual inversion defect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments import confirm_gse239452_citeseq as confirmation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/gse239452_citeseq_post_access_correction.json"
ORIGINAL_ARTIFACTS = {
    "reduced_development": confirmation.DEFAULT_REDUCED,
    "pilot_result": confirmation.DEFAULT_PILOT,
    "held_predictions": confirmation.DEFAULT_PREDICTION,
    "held_score": confirmation.DEFAULT_SCORE,
}


def _original_artifact_bindings() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    runner_hashes = set()
    for label, path in ORIGINAL_ARTIFACTS.items():
        payload = confirmation._read_json(path)
        runner_hash = str(payload.get("runner_sha256"))
        runner_hashes.add(runner_hash)
        bindings[label] = {
            "path": confirmation._relative(path),
            "sha256": confirmation._sha256(path),
            "schema": str(payload.get("schema")),
            "original_runner_sha256": runner_hash,
        }
    if len(runner_hashes) != 1:
        raise PermissionError("original GSE239452 artifacts do not share one runner")
    return bindings


def _development_records(
    source_root: Path,
) -> dict[str, dict[str, Any]]:
    manifest = confirmation._sample_manifest(confirmation.DEFAULT_SOURCE)
    preflight = confirmation._preflight_records(confirmation.DEFAULT_PREFLIGHT)
    records = [
        confirmation._reduce_one(
            donor,
            source_root,
            manifest,
            preflight,
            read_adt_numeric=True,
        )
        for donor in (*confirmation.CALIBRATION, *confirmation.PILOT)
    ]
    return confirmation._validated_reduced_records(records, manifest, preflight)


def _held_reanalysis(
    models: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    original_prediction = confirmation._read_json(confirmation.DEFAULT_PREDICTION)
    frozen_by_donor = {
        sample["donor"]: sample for sample in original_prediction["samples"]
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for donor in confirmation.HELD:
        frozen = frozen_by_donor[donor]
        predictions[donor] = confirmation._predict_panel(
            models,
            np.asarray(frozen["row_margins"], dtype=np.int64),
            np.asarray(frozen["column_margins"], dtype=np.int64),
        )

    manifest = confirmation._sample_manifest(confirmation.DEFAULT_SOURCE)
    preflight = confirmation._preflight_records(confirmation.DEFAULT_PREFLIGHT)
    methods = (
        "primary",
        "best_residual",
        "destroyed_link",
        "graph_zero_diagnostic",
    )
    losses = {
        method: np.empty(len(confirmation.HELD), dtype=float) for method in methods
    }
    samples: list[dict[str, Any]] = []
    residual_model = models["best_residual"]
    residual_config = confirmation._residual_from_dict(
        residual_model["configuration"]
    )
    target_coordinates = (
        residual_config.transport_multiplier
        * np.asarray(residual_model["pooled_coordinate"])
        * np.sqrt(confirmation.CELL_BUDGET)
    )
    residual_statistic = (
        confirmation._fractional_pearson
        if residual_config.family == "pearson"
        else confirmation._fractional_deviance
    )
    coordinate_tolerance = 1e-8
    original_mismatches = 0
    corrected_mismatches = 0
    original_maximum_error = 0.0
    corrected_maximum_error = 0.0
    for index, donor in enumerate(confirmation.HELD):
        truth_record = confirmation._reduce_one(
            donor,
            source_root,
            manifest,
            preflight,
            read_adt_numeric=True,
        )
        truth = np.asarray(truth_record["tables"], dtype=np.int64)
        rows, columns = confirmation._margins(truth)
        frozen = frozen_by_donor[donor]
        if (
            rows.tolist() != frozen["row_margins"]
            or columns.tolist() != frozen["column_margins"]
        ):
            raise PermissionError("held truth margins differ from frozen predictions")

        donor_losses: dict[str, float] = {}
        prediction_hashes: dict[str, str] = {}
        for method in methods:
            predicted = predictions[donor][method]
            loss = confirmation._donor_loss(truth, predicted)
            losses[method][index] = loss
            donor_losses[method] = float(loss)
            prediction_hashes[method] = confirmation._array_sha256(predicted)

        old_residual = np.asarray(
            frozen["predicted_tables"]["best_residual"], dtype=float
        )
        for entity in np.ndindex(target_coordinates.shape):
            target = float(target_coordinates[entity])
            original_error = abs(residual_statistic(old_residual[entity]) - target)
            corrected_error = abs(
                residual_statistic(predictions[donor]["best_residual"][entity])
                - target
            )
            original_mismatches += int(original_error > coordinate_tolerance)
            corrected_mismatches += int(corrected_error > coordinate_tolerance)
            original_maximum_error = max(original_maximum_error, original_error)
            corrected_maximum_error = max(corrected_maximum_error, corrected_error)
        samples.append(
            {
                "donor": donor,
                "losses": donor_losses,
                "prediction_sha256": prediction_hashes,
                "truth_table_sha256": truth_record["table_sha256"],
                "primary_prediction_matches_original": bool(
                    np.array_equal(
                        predictions[donor]["primary"],
                        np.asarray(frozen["predicted_tables"]["primary"]),
                    )
                ),
            }
        )

    gate = confirmation._gate(
        confirmation.HELD, losses, required_favorable=8
    )
    held = {
        "samples": samples,
        "losses": {
            method: {
                donor: float(value)
                for donor, value in zip(confirmation.HELD, values)
            }
            for method, values in losses.items()
        },
        "gate": gate,
        "residual_reconstruction_audit": {
            "coordinate_tolerance": coordinate_tolerance,
            "tables_checked": len(confirmation.HELD) * len(confirmation.MARKERS) ** 2,
            "original_coordinate_mismatches": original_mismatches,
            "corrected_coordinate_mismatches": corrected_mismatches,
            "original_maximum_absolute_coordinate_error": original_maximum_error,
            "corrected_maximum_absolute_coordinate_error": corrected_maximum_error,
        },
    }
    return held


def run(source_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("post-access correction output already exists")
    original_bindings = _original_artifact_bindings()
    records = _development_records(source_root)
    development = confirmation._pilot_analysis(records)
    if not development["pilot_gate"]["passes"]:
        raise RuntimeError("corrected development analysis did not pass the frozen gate")
    held = _held_reanalysis(development["all_development_models"], source_root)

    payload = {
        "schema": "gse239452-post-access-correction/1.0",
        "status": "POST_ACCESS_CORRECTION_COMPLETE",
        "created_at_utc": confirmation._timestamp(),
        "outcome_blind": False,
        "original_sealed_artifacts_overwritten": False,
        "correction": {
            "defect": "signed-root-deviance lower endpoint underflowed and projected some negative coordinates",
            "repair": "replace nextafter endpoints with an interior epsilon of min(1e-10, one quarter of the feasible interval)",
            "scope": "residual inversion only; splits, grids, states, losses, and gates unchanged",
        },
        "corrected_runner": {
            "path": confirmation._relative(Path(confirmation.__file__)),
            "sha256": confirmation._sha256(Path(confirmation.__file__)),
        },
        "correction_runner": {
            "path": confirmation._relative(Path(__file__)),
            "sha256": confirmation._sha256(Path(__file__)),
        },
        "original_sealed_artifacts": original_bindings,
        "splits": {
            "calibration": list(confirmation.CALIBRATION),
            "pilot": list(confirmation.PILOT),
            "held": list(confirmation.HELD),
        },
        "grids": {
            "neighbors": list(confirmation.NEIGHBOR_GRID),
            "heterogeneity": list(confirmation.HETEROGENEITY_GRID),
            "ridge": list(confirmation.RIDGE_GRID),
            "graph": list(confirmation.GRAPH_GRID),
            "transport": list(confirmation.ALPHA_GRID),
            "residual_families": list(confirmation.RESIDUAL_FAMILIES),
        },
        "development": development,
        "held": held,
        "interpretation": "Post-access numerical correction; it does not replace the chronology of the original sealed attempt.",
    }
    confirmation._write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=confirmation.SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.source_root, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
