"""Post-hoc standard-Poisson interaction baseline for GSE239452.

The frozen development tables determine a donor-pooled saturated 2x2 Poisson
interaction.  Its transport multiplier is selected on the original pilot
donors, after fitting on the original calibration donors, and the interaction
is then refit on all 15 development donors.  At each recipient margin, row and
column nuisance terms are profiled with the transported interaction fixed.

Held assay files are streamed one donor at a time from their frozen official
GEO URLs.  Every reproduced table must match its published hash before it can
be scored, and each donor's archives and H5AD files are deleted before the next
donor is downloaded.  Running the CLI without ``--execute-downloads`` only
prints the validated disk/download plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Iterable
import urllib.request

import numpy as np
from scipy.optimize import brentq

from experiments import confirm_gse239452_citeseq as confirmation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "results/development/gse239452_standard_poisson_interaction_posthoc.json"
)
DEFAULT_CACHE = ROOT / "data/confirmation/gse239452_citeseq/posthoc_stream_cache"
LEGACY_AUDIT = ROOT / "results/development/classical_interaction_baselines_posthoc.json"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260829
ROOT_TOLERANCE = 1e-12
CERTIFICATE_TOLERANCE = 1e-9
DISK_RESERVE_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def _load_frozen_inputs() -> dict[str, Any]:
    manifest_payload = _read_json(confirmation.DEFAULT_SOURCE)
    manifest = confirmation._sample_manifest(confirmation.DEFAULT_SOURCE)
    preflight = confirmation._preflight_records(confirmation.DEFAULT_PREFLIGHT)

    reduced_payload = _read_json(confirmation.DEFAULT_REDUCED)
    if (
        reduced_payload.get("schema") != "gse239452-reduced-development/1.0"
        or reduced_payload.get("status") != "DEVELOPMENT_REDUCTION_COMPLETE"
        or reduced_payload.get("source_manifest_sha256")
        != _sha256(confirmation.DEFAULT_SOURCE)
        or reduced_payload.get("metadata_preflight_sha256")
        != _sha256(confirmation.DEFAULT_PREFLIGHT)
    ):
        raise PermissionError("reduced development artifact differs from its freeze")
    records = confirmation._validated_reduced_records(
        reduced_payload.get("samples"), manifest, preflight
    )

    prediction = _read_json(confirmation.DEFAULT_PREDICTION)
    score = _read_json(confirmation.DEFAULT_SCORE)
    legacy = _read_json(LEGACY_AUDIT)
    if (
        prediction.get("schema") != "gse239452-held-predictions/1.0"
        or prediction.get("status") != "PREDICTIONS_FROZEN"
        or prediction.get("held_donors") != list(confirmation.HELD)
        or score.get("schema") != "gse239452-held-confirmation/1.0"
        or score.get("held_donors") != list(confirmation.HELD)
        or legacy.get("schema") != "classical-interaction-baseline-audit/1.0"
    ):
        raise PermissionError("held or legacy artifact differs from the frozen schema")

    prediction_by_donor = {row.get("donor"): row for row in prediction["samples"]}
    score_by_donor = {row.get("donor"): row for row in score["samples"]}
    if set(prediction_by_donor) != set(confirmation.HELD) or set(score_by_donor) != set(
        confirmation.HELD
    ):
        raise PermissionError("held donor axes differ across frozen artifacts")
    for donor in confirmation.HELD:
        frozen = prediction_by_donor[donor]
        scored = score_by_donor[donor]
        rows = np.asarray(frozen.get("row_margins"), dtype=np.int64)
        columns = np.asarray(frozen.get("column_margins"), dtype=np.int64)
        primary = np.asarray(
            frozen.get("predicted_tables", {}).get("primary"), dtype=float
        )
        if (
            rows.shape != (len(confirmation.MARKERS),) * 2 + (2,)
            or columns.shape != rows.shape
            or primary.shape != rows.shape[:-1] + (2, 2)
            or not np.isfinite(primary).all()
            or np.any(primary <= 0.0)
            or not np.allclose(primary.sum(axis=-1), rows)
            or not np.allclose(primary.sum(axis=-2), columns)
            or frozen.get("selected_barcode_axis_sha256")
            != scored.get("selected_barcode_axis_sha256")
            or not isinstance(scored.get("truth_table_sha256"), str)
            or len(scored["truth_table_sha256"]) != 64
        ):
            raise PermissionError(f"frozen held record differs for donor {donor}")

    old_study = legacy.get("studies", {}).get(
        "gse239452_held_cohort_post_access_correction", {}
    )
    if old_study.get("held_losses", {}).get("primary") != score.get(
        "held_losses", {}
    ).get("primary"):
        raise PermissionError("legacy audit and frozen primary losses differ")

    return {
        "manifest_payload": manifest_payload,
        "manifest": manifest,
        "preflight": preflight,
        "records": records,
        "prediction": prediction,
        "prediction_by_donor": prediction_by_donor,
        "score": score,
        "score_by_donor": score_by_donor,
        "legacy": legacy,
    }


def _pooled_poisson_interaction(
    tables: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(tables)
    expected_shape = (
        values.shape[0],
        len(confirmation.MARKERS),
        len(confirmation.MARKERS),
        2,
        2,
    )
    if (
        values.shape != expected_shape
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
    ):
        raise ValueError("development tables have an invalid donor/entity/2x2 axis")
    pooled = values.sum(axis=0, dtype=np.int64)
    if np.any(pooled <= 0):
        raise ValueError("a pooled interaction has a structural zero")
    interaction = (
        np.log(pooled[..., 0, 0])
        + np.log(pooled[..., 1, 1])
        - np.log(pooled[..., 0, 1])
        - np.log(pooled[..., 1, 0])
    )
    maximum_cell_error = 0.0
    maximum_normalized_cell_error = 0.0
    for entity in np.ndindex(interaction.shape):
        table = pooled[entity]
        reconstructed = _profile_poisson_table(
            float(interaction[entity]), table.sum(axis=1), table.sum(axis=0)
        )
        cell_error = float(np.max(np.abs(reconstructed - table)))
        maximum_cell_error = max(maximum_cell_error, cell_error)
        maximum_normalized_cell_error = max(
            maximum_normalized_cell_error, cell_error / float(table.sum())
        )
    if maximum_normalized_cell_error > 1e-10:
        raise FloatingPointError("saturated Poisson fit missed its table certificate")
    return interaction, {
        "estimator": "donor-pooled saturated 2x2 Poisson log-linear interaction",
        "model": ("log(mu_ij)=intercept+row_i+column_j+theta*I(i=1,j=1)"),
        "closed_form": "theta=log(N00*N11/(N01*N10))",
        "donors": int(values.shape[0]),
        "minimum_pooled_cell_count": int(pooled.min()),
        "maximum_pooled_cell_count": int(pooled.max()),
        "structural_zeros": 0,
        "saturated_tables_reconstructed": int(interaction.size),
        "maximum_absolute_saturated_cell_error": maximum_cell_error,
        "maximum_normalized_saturated_cell_error": maximum_normalized_cell_error,
        "normalized_saturated_cell_error_tolerance": 1e-10,
    }


def _profile_poisson_table(
    interaction: float, row_margin: np.ndarray, column_margin: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_margin, dtype=float)
    columns = np.asarray(column_margin, dtype=float)
    if (
        rows.shape != (2,)
        or columns.shape != (2,)
        or np.any(rows <= 0.0)
        or np.any(columns <= 0.0)
        or not math.isclose(float(rows.sum()), float(columns.sum()))
        or not math.isfinite(float(interaction))
    ):
        raise ValueError(
            "finite Poisson profiling requires positive compatible margins"
        )
    total = float(rows.sum())
    lower = max(0.0, float(rows[0] + columns[0] - total))
    upper = min(float(rows[0]), float(columns[0]))
    if not lower < upper:
        raise ValueError("recipient margins have no interior 2x2 table")

    def residual(upper_left: float) -> float:
        cells = (
            upper_left,
            float(rows[0] - upper_left),
            float(columns[0] - upper_left),
            float(total - rows[0] - columns[0] + upper_left),
        )
        if any(value <= 0.0 for value in cells):
            return -math.inf if cells[0] <= 0.0 or cells[3] <= 0.0 else math.inf
        return (
            math.log(cells[0])
            + math.log(cells[3])
            - math.log(cells[1])
            - math.log(cells[2])
            - float(interaction)
        )

    left = np.nextafter(lower, upper)
    right = np.nextafter(upper, lower)
    upper_left = brentq(
        residual,
        left,
        right,
        xtol=ROOT_TOLERANCE,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=100,
    )
    return np.asarray(
        [
            [upper_left, rows[0] - upper_left],
            [columns[0] - upper_left, total - rows[0] - columns[0] + upper_left],
        ],
        dtype=float,
    )


def _predict_poisson(
    interaction: np.ndarray,
    row_margins: np.ndarray,
    column_margins: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    field = np.asarray(interaction, dtype=float)
    rows = np.asarray(row_margins, dtype=float)
    columns = np.asarray(column_margins, dtype=float)
    if (
        field.shape != (len(confirmation.MARKERS),) * 2
        or rows.shape != field.shape + (2,)
        or columns.shape != rows.shape
        or not math.isfinite(float(alpha))
        or alpha <= 0.0
    ):
        raise ValueError("Poisson field, margins, or transport multiplier are invalid")
    transported = float(alpha) * field
    prediction = np.empty(field.shape + (2, 2), dtype=float)
    for entity in np.ndindex(field.shape):
        prediction[entity] = _profile_poisson_table(
            float(transported[entity]), rows[entity], columns[entity]
        )

    reconstructed_interaction = (
        np.log(prediction[..., 0, 0])
        + np.log(prediction[..., 1, 1])
        - np.log(prediction[..., 0, 1])
        - np.log(prediction[..., 1, 0])
    )
    certificate = {
        "maximum_absolute_row_margin_error": float(
            np.max(np.abs(prediction.sum(axis=-1) - rows))
        ),
        "maximum_absolute_column_margin_error": float(
            np.max(np.abs(prediction.sum(axis=-2) - columns))
        ),
        "maximum_absolute_interaction_error": float(
            np.max(np.abs(reconstructed_interaction - transported))
        ),
        "minimum_fitted_mean": float(prediction.min()),
    }
    if (
        max(
            certificate["maximum_absolute_row_margin_error"],
            certificate["maximum_absolute_column_margin_error"],
            certificate["maximum_absolute_interaction_error"],
        )
        > CERTIFICATE_TOLERANCE
        or certificate["minimum_fitted_mean"] <= 0.0
    ):
        raise FloatingPointError(
            "profiled Poisson reconstruction missed its certificate"
        )
    return prediction, certificate


def _development_analysis(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calibration = np.asarray(
        [records[donor]["tables"] for donor in confirmation.CALIBRATION],
        dtype=np.int64,
    )
    pilot = np.asarray(
        [records[donor]["tables"] for donor in confirmation.PILOT], dtype=np.int64
    )
    calibration_interaction, calibration_certificate = _pooled_poisson_interaction(
        calibration
    )
    candidates = []
    for alpha in confirmation.ALPHA_GRID:
        unit_losses = {}
        for donor, truth in zip(confirmation.PILOT, pilot):
            rows, columns = confirmation._margins(truth)
            prediction, _ = _predict_poisson(
                calibration_interaction, rows, columns, float(alpha)
            )
            unit_losses[donor] = float(confirmation._donor_loss(truth, prediction))
        candidates.append(
            {
                "alpha": float(alpha),
                "donor_losses": unit_losses,
                "donor_equal_mean_loss": float(np.mean(list(unit_losses.values()))),
            }
        )
    selected = min(
        candidates, key=lambda row: (row["donor_equal_mean_loss"], row["alpha"])
    )
    development = np.concatenate((calibration, pilot), axis=0)
    interaction, fit_certificate = _pooled_poisson_interaction(development)
    return {
        "selection_design": "fit_7_calibration_select_8_pilot_refit_15_development",
        "calibration_donors": list(confirmation.CALIBRATION),
        "pilot_donors": list(confirmation.PILOT),
        "held_donors_used_for_selection": [],
        "alpha_grid": [float(value) for value in confirmation.ALPHA_GRID],
        "alpha_candidates": candidates,
        "selected_alpha": selected["alpha"],
        "calibration_fit_certificate": calibration_certificate,
        "refit_certificate": fit_certificate,
        "refit_interaction": interaction.tolist(),
    }


def _official_specs(inputs: dict[str, Any], donor: str) -> list[dict[str, Any]]:
    sample = inputs["manifest"][donor]
    frozen = inputs["preflight"][donor]
    template = str(inputs["manifest_payload"]["geo_sample_url_template"])
    specs = []
    for modality in ("gex", "adt"):
        source = sample[modality]
        observed = frozen[modality]
        expected_url = template.format(
            accession=source["accession"], filename=source["filename"]
        )
        if (
            observed.get("accession") != source["accession"]
            or observed.get("filename") != source["filename"]
            or observed.get("h5ad") != source["h5ad"]
            or observed.get("archive_bytes") != source["bytes"]
            or observed.get("url") != expected_url
        ):
            raise PermissionError(f"official {modality} record changed for {donor}")
        specs.append(
            {
                "modality": modality,
                "url": expected_url,
                "archive_name": source["filename"],
                "archive_bytes": int(observed["archive_bytes"]),
                "archive_sha256": str(observed["archive_sha256"]),
                "h5ad_name": source["h5ad"],
                "h5ad_bytes": int(observed["h5ad_bytes"]),
                "h5ad_sha256": str(observed["h5ad_sha256"]),
            }
        )
    return specs


def _peak_bytes(specs: list[dict[str, Any]]) -> int:
    by_modality = {row["modality"]: row for row in specs}
    gex = by_modality["gex"]
    adt = by_modality["adt"]
    return max(
        gex["archive_bytes"] + gex["h5ad_bytes"],
        gex["h5ad_bytes"] + adt["archive_bytes"] + adt["h5ad_bytes"],
    )


def download_plan(cache_root: Path = DEFAULT_CACHE) -> dict[str, Any]:
    inputs = _load_frozen_inputs()
    donors = []
    for donor in confirmation.HELD:
        specs = _official_specs(inputs, donor)
        donors.append(
            {
                "donor": donor,
                "files": specs,
                "peak_bytes": _peak_bytes(specs),
            }
        )
    probe = cache_root.resolve()
    while not probe.exists():
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    maximum_peak = max(row["peak_bytes"] for row in donors)
    required = maximum_peak + DISK_RESERVE_BYTES
    return {
        "schema": "gse239452-standard-poisson-download-plan/1.0",
        "download_execution_started": False,
        "official_individual_file_pairs": len(donors),
        "donors": donors,
        "sequential_maximum_peak_bytes": maximum_peak,
        "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "required_free_bytes": required,
        "available_free_bytes": available,
        "fits_current_disk": available >= required,
        "retention": "archives and H5AD files deleted after each donor reduction",
    }


def _download(spec: dict[str, Any], destination: Path) -> None:
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "coupling-fields/standard-poisson-posthoc"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("xb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=8 << 20)
        if (
            destination.stat().st_size != spec["archive_bytes"]
            or _sha256(destination) != spec["archive_sha256"]
        ):
            raise PermissionError(
                "downloaded official archive failed its frozen digest"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _extract_h5ad(spec: dict[str, Any], archive: Path, directory: Path) -> Path:
    output = directory / spec["h5ad_name"]
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = [member for member in bundle.getmembers() if member.isfile()]
            matching = [
                member
                for member in members
                if Path(member.name).name == spec["h5ad_name"]
            ]
            if len(members) != 1 or len(matching) != 1:
                raise PermissionError("official archive is not one frozen H5AD member")
            member = matching[0]
            if member.size != spec["h5ad_bytes"]:
                raise PermissionError(
                    "official H5AD member size differs from preflight"
                )
            source = bundle.extractfile(member)
            if source is None:
                raise PermissionError("official H5AD member cannot be read")
            with source, output.open("xb") as stream:
                shutil.copyfileobj(source, stream, length=8 << 20)
        if (
            output.stat().st_size != spec["h5ad_bytes"]
            or _sha256(output) != spec["h5ad_sha256"]
        ):
            raise PermissionError("extracted official H5AD failed its frozen digest")
        return output
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _materialize_modality(spec: dict[str, Any], directory: Path) -> Path:
    archive = directory / spec["archive_name"]
    try:
        _download(spec, archive)
        return _extract_h5ad(spec, archive, directory)
    finally:
        archive.unlink(missing_ok=True)


def _verify_truth_record(
    donor: str,
    record: dict[str, Any],
    frozen: dict[str, Any],
    scored: dict[str, Any],
) -> np.ndarray:
    truth = np.asarray(record.get("tables"), dtype=np.int64)
    rows, columns = confirmation._margins(truth)
    if (
        truth.shape != (len(confirmation.MARKERS),) * 2 + (2, 2)
        or record.get("table_sha256") != confirmation._array_sha256(truth)
        or record.get("table_sha256") != scored.get("truth_table_sha256")
        or record.get("selected_barcode_axis_sha256")
        != frozen.get("selected_barcode_axis_sha256")
        or rows.tolist() != frozen.get("row_margins")
        or columns.tolist() != frozen.get("column_margins")
    ):
        raise PermissionError(f"reproduced held truth differs for donor {donor}")
    return truth


def _comparison(
    donors: Iterable[str], primary: np.ndarray, poisson: np.ndarray
) -> dict[str, Any]:
    units = list(donors)
    primary_values = np.asarray(primary, dtype=float)
    poisson_values = np.asarray(poisson, dtype=float)
    if (
        primary_values.shape != (len(units),)
        or poisson_values.shape != primary_values.shape
        or np.any(primary_values <= 0.0)
        or np.any(poisson_values <= 0.0)
    ):
        raise ValueError("paired comparison requires positive donor losses")
    difference = primary_values - poisson_values
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(units), size=(BOOTSTRAP_DRAWS, len(units)), endpoint=False
    )
    boot_difference = difference[indices].mean(axis=1)
    boot_relative = 1.0 - (
        primary_values[indices].mean(axis=1) / poisson_values[indices].mean(axis=1)
    )
    favorable = int(np.count_nonzero(difference < 0.0))
    sign_p = sum(
        math.comb(len(units), value) for value in range(favorable, len(units) + 1)
    ) / float(2 ** len(units))
    return {
        "unit": "physical donor",
        "units": len(units),
        "primary_mean_loss": float(primary_values.mean()),
        "standard_poisson_mean_loss": float(poisson_values.mean()),
        "paired_difference_primary_minus_poisson": {
            donor: float(value) for donor, value in zip(units, difference)
        },
        "paired_difference_95_percentile_ci": np.quantile(
            boot_difference, [0.025, 0.975], method="linear"
        ).tolist(),
        "relative_loss_reduction_primary_vs_poisson": float(
            1.0 - primary_values.mean() / poisson_values.mean()
        ),
        "relative_loss_reduction_95_percentile_ci": np.quantile(
            boot_relative, [0.025, 0.975], method="linear"
        ).tolist(),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "favorable_donors_primary_lower": favorable,
        "exact_one_sided_sign_test_p": sign_p,
        "ties_counted_as_nonfavorable": True,
        "post_hoc_nonconfirmatory": True,
    }


def run(cache_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("post-hoc output already exists")
    plan = download_plan(cache_root)
    if not plan["fits_current_disk"]:
        raise OSError(
            "sequential held-donor materialization does not fit current free disk"
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    if any(cache_root.iterdir()):
        raise FileExistsError("post-hoc streaming cache must be empty")

    inputs = _load_frozen_inputs()
    development = _development_analysis(inputs["records"])
    interaction = np.asarray(development["refit_interaction"], dtype=float)
    alpha = float(development["selected_alpha"])
    old_field = np.asarray(
        inputs["legacy"]["studies"]["gse239452_held_cohort_post_access_correction"][
            "fitted_fields"
        ]["pooled_poisson_loglinear_interaction"]["log_odds"],
        dtype=float,
    )
    fit_replay_error = float(np.max(np.abs(interaction - old_field)))
    if fit_replay_error > 1e-12:
        raise PermissionError("pooled interaction does not replay the legacy fit")

    primary_losses = np.empty(len(confirmation.HELD), dtype=float)
    poisson_losses = np.empty(len(confirmation.HELD), dtype=float)
    samples = []
    for index, donor in enumerate(confirmation.HELD):
        specs = _official_specs(inputs, donor)
        free = shutil.disk_usage(cache_root).free
        required = _peak_bytes(specs) + DISK_RESERVE_BYTES
        if free < required:
            raise OSError(f"donor {donor} cannot fit the sequential disk budget")
        with tempfile.TemporaryDirectory(
            prefix=f".{donor}-", dir=cache_root
        ) as temporary:
            donor_root = Path(temporary)
            for spec in specs:
                _materialize_modality(spec, donor_root)
            record = confirmation._reduce_one(
                donor,
                donor_root,
                inputs["manifest"],
                inputs["preflight"],
                read_adt_numeric=True,
            )
            frozen = inputs["prediction_by_donor"][donor]
            scored = inputs["score_by_donor"][donor]
            truth = _verify_truth_record(donor, record, frozen, scored)
            rows, columns = confirmation._margins(truth)
            poisson_prediction, certificate = _predict_poisson(
                interaction, rows, columns, alpha
            )
            primary_prediction = np.asarray(
                frozen["predicted_tables"]["primary"], dtype=float
            )
            primary_loss = float(confirmation._donor_loss(truth, primary_prediction))
            published_primary = float(scored["losses"]["primary"])
            if not math.isclose(
                primary_loss, published_primary, rel_tol=0.0, abs_tol=1e-15
            ):
                raise PermissionError(
                    f"frozen primary loss does not replay for donor {donor}"
                )
            poisson_loss = float(confirmation._donor_loss(truth, poisson_prediction))
            primary_losses[index] = primary_loss
            poisson_losses[index] = poisson_loss
            samples.append(
                {
                    "donor": donor,
                    "truth_table_sha256": record["table_sha256"],
                    "selected_barcode_axis_sha256": record[
                        "selected_barcode_axis_sha256"
                    ],
                    "primary_prediction_sha256": confirmation._array_sha256(
                        primary_prediction
                    ),
                    "standard_poisson_prediction_sha256": (
                        confirmation._array_sha256(poisson_prediction)
                    ),
                    "primary_loss": primary_loss,
                    "standard_poisson_loss": poisson_loss,
                    "reconstruction_certificate": certificate,
                }
            )
        if any(cache_root.iterdir()):
            raise RuntimeError(f"raw donor cache was not deleted after {donor}")

    comparison = _comparison(confirmation.HELD, primary_losses, poisson_losses)
    payload = {
        "schema": "gse239452-standard-poisson-interaction-posthoc/1.0",
        "status": "POST_HOC_NONCONFIRMATORY_HEAD_TO_HEAD",
        "confirmatory": False,
        "reason_post_hoc": (
            "The standard-Poisson reconstruction and comparison were defined "
            "after held outcomes had been accessed."
        ),
        "bindings": {
            "runner_sha256": _sha256(Path(__file__)),
            "source_manifest_sha256": _sha256(confirmation.DEFAULT_SOURCE),
            "metadata_preflight_sha256": _sha256(confirmation.DEFAULT_PREFLIGHT),
            "reduced_development_sha256": _sha256(confirmation.DEFAULT_REDUCED),
            "frozen_prediction_sha256": _sha256(confirmation.DEFAULT_PREDICTION),
            "frozen_score_sha256": _sha256(confirmation.DEFAULT_SCORE),
            "legacy_mislabeled_audit_sha256": _sha256(LEGACY_AUDIT),
        },
        "method": {
            "source_fit": "donor-pooled saturated 2x2 Poisson interaction",
            "recipient_reconstruction": (
                "profile row and column nuisance terms at recipient margins "
                "with transported interaction fixed"
            ),
            "noncentral_hypergeometric_reconstruction_used": False,
            "loss": "multinomial deviance per cell",
            "held_aggregation": (
                "equal weight per physical donor after averaging informative "
                "RNA--ADT pairs within donor"
            ),
            "fixed_total_equivalence": (
                "With identical observed and fitted donor totals, Poisson "
                "nuisance terms cancel and the reported expression is the "
                "multinomial deviance."
            ),
        },
        "streaming_access": {
            "official_individual_file_pairs": len(confirmation.HELD),
            "sequential_maximum_peak_bytes": plan["sequential_maximum_peak_bytes"],
            "official_archive_and_h5ad_provenance": plan["donors"],
            "raw_files_retained_after_each_donor": False,
            "held_truth_tables_serialized": False,
            "held_truth_hashes_reproduced": len(samples),
        },
        "development": {
            **development,
            "maximum_absolute_refit_difference_vs_legacy_coefficient": (
                fit_replay_error
            ),
        },
        "held": {
            "samples": samples,
            "primary_donor_equal_mean_loss": float(primary_losses.mean()),
            "standard_poisson_donor_equal_mean_loss": float(poisson_losses.mean()),
            "comparison": comparison,
        },
    }
    _write_json_exclusive(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-downloads", action="store_true")
    args = parser.parse_args()
    if args.execute_downloads:
        payload = run(args.cache_root, args.output)
    else:
        payload = download_plan(args.cache_root)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
