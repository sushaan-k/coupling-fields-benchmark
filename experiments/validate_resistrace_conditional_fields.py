"""Validate conditional association fields in the ReSisTrace release.

The deposited barcode groups link related pre- and post-treatment cells; they
do not track one physical cell twice.  Each barcode group is therefore reduced
to one lineage-level pre coordinate and one lineage-level post coordinate.
The state encoder is fitted only to pooled pre-treatment Control lineages, then
applied unchanged to every endpoint in all eight condition-by-replicate arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from experiments.development.resistrace_paired_route_gate import (
    TARGET_CONDITIONS,
    SampleKey,
    discover_samples,
    fit_preonly_encoder,
    lineage_state_pairs,
    read_cell_info,
    read_selected_log_expression,
    select_preonly_features,
    summarize_linkage,
)
from mapreg.coupling_fields import (
    association_coordinates,
    association_field,
    conditional_association_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/resistrace_conditional_fields.json"
DATA_ROOT = ROOT / "data/external/resistrace_gse223003"
STATE_COUNT = 3
PSEUDOCOUNT = 0.5
PERMUTATIONS = 64
MINIMUM_STATE_SUPPORT = 10
SEED = 223_003


@dataclass(frozen=True)
class ControlPreStateEncoder:
    """A fixed nearest-centroid encoder calibrated on control-pre lineages."""

    center: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray
    calibration_lineages: int
    inertia: float

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        values = (np.asarray(coordinates, dtype=float) - self.center) / self.scale
        distance = np.square(values[:, None, :] - self.centroids[None, :, :]).sum(
            axis=2
        )
        return np.argmin(distance, axis=1).astype(int)


def fit_control_pre_state_encoder(
    coordinates: Sequence[np.ndarray],
    *,
    state_count: int,
    seed: int,
) -> ControlPreStateEncoder:
    """Fit standardized k-means using pooled control-pre lineages only."""

    pooled = np.vstack([np.asarray(values, dtype=float) for values in coordinates])
    if pooled.ndim != 2 or pooled.shape[0] < 10 * state_count:
        raise ValueError("too few control-pre lineages for state calibration")
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (pooled - center) / scale
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cluster = KMeans(n_clusters=state_count, n_init=100, random_state=seed).fit(
            standardized
        )
    if not np.isfinite(cluster.cluster_centers_).all():
        raise FloatingPointError("state calibration produced a non-finite centroid")
    return ControlPreStateEncoder(
        center=center,
        scale=scale,
        centroids=cluster.cluster_centers_,
        calibration_lineages=len(pooled),
        inertia=float(cluster.inertia_),
    )


def _stable_seed(label: str, seed: int) -> int:
    return int(seed + zlib.crc32(label.encode("utf-8")))


def _association_coordinates_from_states(
    first: np.ndarray,
    second: np.ndarray,
    *,
    state_count: int,
    pseudocount: float,
) -> np.ndarray:
    table = np.bincount(first * state_count + second, minlength=state_count**2).reshape(
        state_count, state_count
    )
    return association_coordinates(association_field(table, pseudocount=pseudocount))


def conditional_field_with_null(
    first: np.ndarray,
    second: np.ndarray,
    *,
    state_count: int,
    pseudocount: float,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the production field and disjoint fixed-margin test fields.

    The production estimate is centered on ``permutations`` reference draws and
    is left unchanged.  The randomization test uses a separate set of
    ``permutations`` draws, each centered on that same reference mean.  The
    observed field and test fields are therefore computed by the same rule and
    are exchangeable under the fixed-margin link null.
    """

    left = np.asarray(first, dtype=int)
    right = np.asarray(second, dtype=int)
    estimate = conditional_association_coordinates(
        left,
        right,
        first_levels=state_count,
        second_levels=state_count,
        pseudocount=pseudocount,
        permutations=permutations,
        seed=seed,
    )

    # Reproduce the production estimator's randomization stream exactly, then
    # continue it to obtain a test set disjoint from the reference permutations.
    # Draw 0 remains the production estimator's reported destruction control;
    # draws 1..B are its reference set; draws B+1..2B-1 complete the B-draw test
    # set without altering the fitted field.
    rng = np.random.default_rng(seed)
    null_raw = np.asarray(
        [
            _association_coordinates_from_states(
                left,
                right[rng.permutation(len(left))],
                state_count=state_count,
                pseudocount=pseudocount,
            )
            for _ in range(2 * permutations)
        ]
    )
    null_mean = null_raw[1 : permutations + 1].mean(axis=0)
    test_raw = np.concatenate((null_raw[:1], null_raw[permutations + 1 :]), axis=0)
    null_centered = test_raw - null_mean
    expected_destroyed = null_raw[0] - null_mean
    if not np.allclose(estimate.coordinates, estimate.observed_coordinates - null_mean):
        raise RuntimeError("conditional-field reconstruction disagrees with production")
    if not np.allclose(estimate.destroyed_coordinates, expected_destroyed):
        raise RuntimeError("destroyed-link reconstruction disagrees with production")
    return estimate.coordinates, null_centered, estimate.destroyed_coordinates


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def _support(states: np.ndarray, state_count: int) -> list[int]:
    return np.bincount(states, minlength=state_count).astype(int).tolist()


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _benjamini_hochberg(values: Sequence[float]) -> list[float]:
    p_values = np.asarray(values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p-values must be a finite vector")
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_contrast_cosines(
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    condition: str,
    *,
    bootstraps: int,
    state_count: int,
    pseudocount: float,
    permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstraps, dtype=float)
    for draw in range(bootstraps):
        replicate_contrasts = []
        for replicate in (1, 2):
            fields = []
            for arm_condition in (condition, "Control"):
                namespace = SampleKey(arm_condition, replicate, "pre").namespace
                first, second = arms[namespace]
                index = rng.integers(0, len(first), size=len(first))
                field = conditional_association_coordinates(
                    first[index],
                    second[index],
                    first_levels=state_count,
                    second_levels=state_count,
                    pseudocount=pseudocount,
                    permutations=permutations,
                    seed=_stable_seed(
                        f"{condition}|{arm_condition}|{replicate}|{draw}", seed
                    ),
                ).coordinates
                fields.append(field)
            replicate_contrasts.append(fields[0] - fields[1])
        samples[draw] = _cosine(*replicate_contrasts)
    return samples


def run_validation(
    data_root: Path,
    *,
    n_features: int,
    n_components: int,
    min_detection_fraction: float,
    chunk_size: int,
    permutations: int,
    bootstraps: int,
    minimum_state_support: int,
    seed: int,
) -> dict[str, object]:
    files = discover_samples(data_root)
    metadata = {
        key: read_cell_info(sample.path)
        for (key, kind), sample in files.items()
        if kind == "cell_info"
    }
    pre_inputs = [
        (files[(key, "UMIcounts")].path, metadata[key])
        for key in sorted(metadata)
        if key.timepoint == "pre"
    ]
    genes, moments = select_preonly_features(
        pre_inputs,
        n_features=n_features,
        min_detection_fraction=min_detection_fraction,
        chunk_size=chunk_size,
    )

    cells: dict[SampleKey, list[str]] = {}
    expression: dict[SampleKey, np.ndarray] = {}
    for key in sorted(metadata):
        names, values = read_selected_log_expression(
            files[(key, "UMIcounts")].path,
            metadata[key],
            genes,
            chunk_size=chunk_size,
        )
        cells[key] = names
        expression[key] = values

    # Apple Accelerate can leave benign floating-point status flags set during
    # randomized SVD. Explicit finiteness checks below remain the hard gate.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pca = fit_preonly_encoder(
            [values for key, values in expression.items() if key.timepoint == "pre"],
            n_components=n_components,
            seed=seed,
        )
        embedded = {key: pca.transform(values) for key, values in expression.items()}
    if not all(np.isfinite(values).all() for values in embedded.values()):
        raise FloatingPointError("PCA embedding contains a non-finite coordinate")

    lineage_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    linkage = {}
    for condition in (*TARGET_CONDITIONS, "Control"):
        for replicate in (1, 2):
            pre_key = SampleKey(condition, replicate, "pre")
            post_key = SampleKey(condition, replicate, "post")
            linkage[pre_key.namespace] = asdict(
                summarize_linkage(pre_key, metadata[pre_key], metadata[post_key])
            )
            _, pre_values, post_values = lineage_state_pairs(
                embedded[pre_key],
                cells[pre_key],
                metadata[pre_key],
                embedded[post_key],
                cells[post_key],
                metadata[post_key],
            )
            lineage_coordinates[pre_key.namespace] = (pre_values, post_values)

    state_encoder = fit_control_pre_state_encoder(
        [
            lineage_coordinates[SampleKey("Control", replicate, "pre").namespace][0]
            for replicate in (1, 2)
        ],
        state_count=STATE_COUNT,
        seed=seed + 1,
    )
    lineage_states = {
        namespace: (
            state_encoder.transform(pre_values),
            state_encoder.transform(post_values),
        )
        for namespace, (pre_values, post_values) in lineage_coordinates.items()
    }

    support = {
        namespace: {
            "pre": _support(first, STATE_COUNT),
            "post": _support(second, STATE_COUNT),
        }
        for namespace, (first, second) in lineage_states.items()
    }
    support_pass = all(
        min(endpoint) >= minimum_state_support
        for arm in support.values()
        for endpoint in arm.values()
    )

    arm_fields: dict[str, np.ndarray] = {}
    arm_nulls: dict[str, np.ndarray] = {}
    arm_results = {}
    for namespace in sorted(lineage_states):
        first, second = lineage_states[namespace]
        arm_seed = _stable_seed(namespace, seed)
        field, null, destroyed = conditional_field_with_null(
            first,
            second,
            state_count=STATE_COUNT,
            pseudocount=PSEUDOCOUNT,
            permutations=permutations,
            seed=arm_seed,
        )
        arm_fields[namespace] = field
        arm_nulls[namespace] = null
        observed_norm = float(np.linalg.norm(field))
        null_norm = np.linalg.norm(null.reshape(len(null), -1), axis=1)
        arm_results[namespace] = {
            "paired_lineages": len(first),
            "field_coordinates": field.ravel().tolist(),
            "field_frobenius": observed_norm,
            "held_destroyed_field_coordinates": destroyed.ravel().tolist(),
            "held_destroyed_field_frobenius": float(np.linalg.norm(destroyed)),
            "destroyed_null_median": float(np.median(null_norm)),
            "destroyed_null_p95": float(np.quantile(null_norm, 0.95)),
            "destroyed_link_p_value": float(
                (1 + np.count_nonzero(null_norm >= observed_norm)) / (permutations + 1)
            ),
        }
    arm_q_values = _benjamini_hochberg(
        [
            arm_results[namespace]["destroyed_link_p_value"]
            for namespace in sorted(arm_results)
        ]
    )
    for namespace, q_value in zip(sorted(arm_results), arm_q_values):
        arm_results[namespace]["destroyed_link_bh_q_value"] = q_value

    contrast_results = {}
    reproducible = []
    for condition in TARGET_CONDITIONS:
        contrasts = []
        destroyed_contrasts = []
        for replicate in (1, 2):
            target = SampleKey(condition, replicate, "pre").namespace
            control = SampleKey("Control", replicate, "pre").namespace
            contrasts.append(arm_fields[target] - arm_fields[control])
            destroyed_contrasts.append(arm_nulls[target] - arm_nulls[control])
        observed_cosine = _cosine(*contrasts)
        null_cosine = np.asarray(
            [
                _cosine(destroyed_contrasts[0][draw], destroyed_contrasts[1][draw])
                for draw in range(permutations)
            ]
        )
        bootstrap = _bootstrap_contrast_cosines(
            lineage_states,
            condition,
            bootstraps=bootstraps,
            state_count=STATE_COUNT,
            pseudocount=PSEUDOCOUNT,
            permutations=permutations,
            seed=_stable_seed(f"bootstrap|{condition}", seed),
        )
        interval = _percentile_interval(bootstrap)
        null_p = float(
            (1 + np.count_nonzero(null_cosine >= observed_cosine)) / (permutations + 1)
        )
        if interval[0] > 0.0 and null_p <= 0.05:
            reproducible.append(condition)
        contrast_results[condition] = {
            "replicate_1_coordinates": contrasts[0].ravel().tolist(),
            "replicate_2_coordinates": contrasts[1].ravel().tolist(),
            "replicate_contrast_cosine": observed_cosine,
            "lineage_bootstrap_cosine_ci95": interval,
            "lineage_bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
            "destroyed_link_cosine_null_median": float(np.median(null_cosine)),
            "destroyed_link_cosine_null_ci95": _percentile_interval(null_cosine),
            "destroyed_link_reproducibility_p_value": null_p,
        }
    contrast_q_values = _benjamini_hochberg(
        [
            contrast_results[condition]["destroyed_link_reproducibility_p_value"]
            for condition in TARGET_CONDITIONS
        ]
    )
    for condition, q_value in zip(TARGET_CONDITIONS, contrast_q_values):
        contrast_results[condition]["destroyed_link_reproducibility_bh_q_value"] = (
            q_value
        )

    if support_pass and reproducible:
        status = "CONDITIONAL_FIELD_CONTRAST_REPRODUCED"
    elif support_pass:
        status = "REFUSE_TREATMENT_CONTRAST_NOT_REPRODUCED"
    else:
        status = "REFUSE_INSUFFICIENT_THREE_STATE_SUPPORT"

    provenance = [
        {
            "gsm": sample.gsm,
            "condition": sample.key.condition,
            "replicate": sample.key.replicate,
            "timepoint": sample.key.timepoint,
            "kind": sample.kind,
            "path": str(sample.path.relative_to(data_root)),
            "bytes": sample.path.stat().st_size,
            "sha256": _sha256(sample.path),
        }
        for sample in sorted(files.values(), key=lambda value: str(value.path))
    ]
    selected = moments.loc[genes]
    return {
        "schema": "resistrace-conditional-fields-v1",
        "status": status,
        "claim_boundary": {
            "paired_unit": "barcode-defined lineage of related cells",
            "not_observed": "the same physical cell before and after treatment",
            "biological_replicates": 2,
            "lineage_ci_scope": (
                "conditional on the two deposited replicates; lineage resampling "
                "does not create additional biological replicates"
            ),
            "contrast": (
                "condition-minus-handling-control association field within each "
                "deposited replicate"
            ),
        },
        "estimator": {
            "field": "3x3 marginal-invariant log association in Helmert coordinates",
            "pseudocount": PSEUDOCOUNT,
            "conditional_centering": (
                "mean of exact within-arm post-state permutations preserving both "
                "empirical state margins"
            ),
            "fixed_margin_reference_permutations": permutations,
            "fixed_margin_test_permutations": permutations,
            "link_test": (
                "observed and disjoint test permutations centered against the "
                "same independent reference-permutation mean"
            ),
            "state_encoder": (
                "eight lineage-level PCA coordinates standardized on pooled Control "
                "pre lineages, then 3-state k-means (100 starts) fitted on those "
                "same Control pre lineages and applied unchanged to all endpoints"
            ),
            "minimum_lineages_per_endpoint_state": minimum_state_support,
        },
        "feature_freeze": {
            "fit_data": "all eight pre-treatment cell libraries only",
            "normalization": "log1p(10000 * UMI / deposited nCount_RNA)",
            "selection": "highest pooled pre-treatment log-expression variance",
            "n_features": n_features,
            "selected_genes": genes,
            "selected_gene_detection_fraction": {
                gene: float(selected.loc[gene, "detection_fraction"]) for gene in genes
            },
            "pca_components": n_components,
            "pca_explained_variance_ratio": pca.explained_variance_ratio.tolist(),
            "control_pre_calibration_lineages": state_encoder.calibration_lineages,
            "control_pre_kmeans_inertia": state_encoder.inertia,
        },
        "linkage": linkage,
        "state_support": support,
        "arm_fields": arm_results,
        "treatment_minus_control_reproducibility": contrast_results,
        "gate": {
            "support_pass_all_arms": support_pass,
            "reproducible_treatment_contrasts": reproducible,
            "criterion": (
                "positive lineage-bootstrap 95% CI and one-sided exact "
                "link-destruction p <= 0.05"
            ),
        },
        "lineage_bootstraps": bootstraps,
        "random_seed": seed,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--min-detection-fraction", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=2_000)
    parser.add_argument(
        "--minimum-state-support", type=int, default=MINIMUM_STATE_SUPPORT
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_validation(
        args.data_root,
        n_features=args.n_features,
        n_components=args.n_components,
        min_detection_fraction=args.min_detection_fraction,
        chunk_size=args.chunk_size,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        minimum_state_support=args.minimum_state_support,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
