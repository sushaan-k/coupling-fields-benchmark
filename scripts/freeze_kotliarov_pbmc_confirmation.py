"""Create the disabled-outcome Kotliarov confirmation designation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/confirmation/kotliarov_pbmc/candidate_designation_v1.json"
ARTIFACTS = {
    "protocol": "docs/KOTLIAROV_PBMC_HELD_BATCH_CONFIRMATION_PROTOCOL_2026-08-28.md",
    "source_manifest": "data/development/kotliarov_pbmc/source_manifest_v1.json",
    "metadata_support_artifact": "data/development/kotliarov_pbmc/metadata_support_v1.json",
    "alias_table": "data/development/kotliarov_pbmc/adt_gene_aliases_v1.tsv",
    "lineage_markers": "data/development/kotliarov_pbmc/lineage_markers_v1.tsv",
    "runner": "experiments/confirm_kotliarov_pbmc.py",
    "reducer": "experiments/reduce_kotliarov_pbmc.py",
    "test": "tests/test_kotliarov_pbmc_confirmation.py",
    "reducer_test": "tests/test_kotliarov_pbmc_reducer.py",
    "authorization_template": "data/confirmation/kotliarov_pbmc/score_authorization_template_v1.json",
    "authorization_script": "scripts/authorize_kotliarov_pbmc_outcomes.py",
    "verification_script": "scripts/verify_kotliarov_public_freeze.py",
    "embedding": "data/scgpt_gene_embeddings.npz",
    "embedding_manifest": "data/scgpt_gene_embeddings_manifest.json",
}
IMPLEMENTATION = (
    "mapreg/coupling_fields.py",
    "mapreg/classical_residuals.py",
    "mapreg/table_prediction.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [relative for relative in (*ARTIFACTS.values(), *IMPLEMENTATION) if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen artifacts: {missing}")
    if OUTPUT.exists():
        raise FileExistsError(f"designation already exists: {OUTPUT}")

    paths = {key: value for key, value in ARTIFACTS.items()}
    hash_keys = {
        "metadata_support_artifact": "metadata_support_sha256",
        "alias_table": "alias_sha256",
        **{
            key: f"{key}_sha256"
            for key in ARTIFACTS
            if key not in {"metadata_support_artifact", "alias_table"}
        },
    }
    hashes = {
        hash_keys[key]: sha256(ROOT / value) for key, value in ARTIFACTS.items()
    }
    source = json.loads((ROOT / ARTIFACTS["source_manifest"]).read_text())
    support = json.loads((ROOT / ARTIFACTS["metadata_support_artifact"]).read_text())
    record = {
        "schema": "kotliarov-pbmc-coupling-candidate-designation/1.0",
        "designation": str(OUTPUT.relative_to(ROOT)),
        "status": "OUTCOME_ACCESS_DISABLED",
        "designated_at_utc": "2026-08-28T04:30:00Z",
        "dataset": "KotliarovPBMCData",
        "study_doi": "10.1038/s41591-020-0769-8",
        "design": "held_experimental_batch_and_disjoint_donors",
        "biological_replication_unit": "donor",
        "confirmatory_family": {
            "name": "kotliarov-held-batch-v1",
            "scoreable_candidates": ["KotliarovPBMCData"],
            "execute_all_candidates": True,
            "stopping_rule": "none",
            "family_alpha": 0.05,
            "directional_alpha": 0.025,
            "prior_family": "Lawlor/Hao family closed after two pre-outcome procedural refusals; no held joint score was formed",
        },
        **paths,
        **hashes,
        "source_total_bytes": sum(item["bytes"] for item in source["files"]),
        "source_files": source["files"],
        "preseal_source_access": source["preseal_access"],
        "development_donors": support["development"]["donors"],
        "held_donors": support["held"]["donors"],
        "held_high_responders": support["held"]["high_responders"],
        "held_low_responders": support["held"]["low_responders"],
        "excluded_donors": [support["excluded_donor"]],
        "expected_cells_before_rna_qc": support["retained_cells_before_rna_qc"],
        "lineages": ["B", "CD4 T", "CD8 T", "NK", "Monocyte"],
        "minimum_lineages": 4,
        "minimum_lineage_cells": 50,
        "minimum_state_cells": 5,
        "minimum_state_fraction": 0.02,
        "minimum_markers": 16,
        "minimum_entities": 32,
        "minimum_embedding_markers": 12,
        "state_levels": 3,
        "pseudocount": 0.5,
        "permutations": 64,
        "destroyed_permutation_index": 65,
        "primary_nuclear_grid": [0.03, 0.1, 0.3, 1.0],
        "primary_graph_grid": [0.5, 2.0, 5.0, 10.0],
        "nuclear_only_grid": [0.03, 0.1, 0.3, 1.0],
        "hypergraph_only_grid": [0.5, 2.0, 5.0, 10.0],
        "external_neighbors": 6,
        "development_folds": 10,
        "bootstrap_draws": 10000,
        "sign_flip_assignments": 512,
        "seed": 20260828,
        "primary_endpoint": "donor-equal multinomial deviance per held cell on identical observed 3x3 margins",
        "primary_gate": [
            "field correlation bootstrap lower endpoint > 0, exact one-sided sign-flip p <= 0.025, and at least 8/9 donors positive",
            "primary minus unstructured field deviance bootstrap upper endpoint < 0, exact p <= 0.025, relative reduction >= 5%, and at least 8/9 donors favor primary",
            "same conditions versus bootstrap-reselected best full-matrix classical residual comparator",
            "primary minus pairing-destroyed deviance bootstrap upper endpoint < 0, exact p <= 0.025, and at least 8/9 donors favor primary",
            "primary minus bootstrap-reselected best matched non-primary field method bootstrap upper endpoint < 0, exact p <= 0.025, and at least 8/9 donors favor primary",
            "all source, donor-exclusion, support, pairing-seal, optimization, reconstruction, and hash checks pass",
        ],
        "prediction_bundle": "data/development/kotliarov_pbmc/prediction_bundle_v1",
        "score_bundle": "data/confirmation/kotliarov_pbmc/score_bundle_v1",
        "prepare_record": "data/development/kotliarov_pbmc/prepared_v1.json",
        "prediction_path": "results/kotliarov_pbmc_predictions.json",
        "score_authorization": "data/confirmation/kotliarov_pbmc/score_authorization_v1.json",
        "score_release": "data/confirmation/kotliarov_pbmc/score_release_v1.json",
        "score_output": "results/kotliarov_pbmc_confirmation.json",
        "score_arrays": "results/kotliarov_pbmc_confirmation_arrays.npz",
        "prepare_refusal_path": "results/kotliarov_pbmc_prepare_refusal.json",
        "prediction_refusal_path": "results/kotliarov_pbmc_prediction_refusal.json",
        "score_refusal_path": "results/kotliarov_pbmc_score_refusal.json",
        "implementation_sha256": {relative: sha256(ROOT / relative) for relative in IMPLEMENTATION},
        "public_freeze_commit": None,
        "public_freeze_url": None,
        "sealed_at_utc": None,
        "outcome_access_authorized": False,
        "public_verification_requirement": "Fresh-clone and byte-verify the disabled designation and every bound artifact before count-matrix acquisition; publish and independently verify predictions and score authorization before opening the score-only pairing bundle.",
        "seal_blocker": "outcome access remains disabled until the public freeze is independently verified",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
