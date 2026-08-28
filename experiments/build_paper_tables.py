"""Build exact LaTeX comparator tables from frozen result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def _interval(value: list[float] | None) -> str:
    return "--" if value is None else f"{value[0]:.3f}, {value[1]:.3f}"


def _escape(value: str) -> str:
    return value.replace("_", r"\_").replace("--", r"{}--{}")


def _method_label(value: str) -> str:
    labels = {
        "variance_weighted_nuclear_scgpt_hypergraph": "weighted nuclear + scGPT hypergraph",
        "nuclear_scgpt_hypergraph": "nuclear + scGPT hypergraph",
        "nuclear_gene2vec_hypergraph": "nuclear + gene2vec hypergraph",
        "conditional_nuclear_scgpt_hypergraph": "conditional nuclear + scGPT hypergraph",
        "conditional_nuclear_gene2vec_hypergraph": "conditional nuclear + gene2vec hypergraph",
        "scgpt_hypergraph": "scGPT hypergraph",
        "shuffled_hypergraph": "shuffled hypergraph",
        "graph_only": "hypergraph only",
        "gene2vec_prior": "gene2vec hypergraph",
        "shuffled_scgpt_prior": "shuffled scGPT hypergraph",
        "endpoint_plus_structured_residual": "endpoint + structured residual",
        "permutation_reliability_shrinkage": "permutation-reliability shrinkage",
        "linear_cross_covariance": "linear cross-covariance",
        "marginal_residual_atlas": "marginal + structured residual",
        "direct_mean_remaining_dates": "direct mean",
        "endpoint_margin_ridge": "endpoint ridge",
        "fixed_structured": "weighted nuclear + scGPT hypergraph",
    }
    return labels.get(value, value.replace("_", " "))


def _method_rows(panel: str, methods: dict) -> list[str]:
    rows = []
    for method, record in methods.items():
        metrics = record["metrics"]
        intervals = record.get("target_bootstrap_95_ci", {})
        rows.append(
            " & ".join(
                (
                    _escape(panel),
                    _escape(_method_label(method)),
                    _fmt(metrics.get("pooled_pearson")),
                    _interval(intervals.get("pooled_pearson")),
                    _fmt(metrics.get("macro_target_cosine")),
                    _fmt(metrics.get("standardized_rmse")),
                    _interval(intervals.get("standardized_rmse")),
                )
            )
            + r" \\"
        )
    return rows


def build(output: Path) -> None:
    rows: list[str] = []
    atlas = _load(
        ROOT / "results/public_coupling_atlas_benchmark_v4_final_estimator.json"
    )
    for panel in atlas["panels"]:
        rows.extend(_method_rows(panel["panel"], panel["methods"]))
    multiperturb = _load(ROOT / "results/multiperturb_conditional_fields.json")
    rows.extend(_method_rows("MultiPerturb", multiperturb["methods"]))
    perturbfate = _load(
        ROOT / "results/development/perturbfate_conditional_fields.json"
    )
    rows.extend(_method_rows("PerturbFate", perturbfate["methods"]))
    arce = _load(ROOT / "results/arce_gse278572_postlock_controls.json")
    rows.extend(_method_rows("Arce held donor", arce["methods"]))

    text = "\n".join(
        (
            r"\begin{longtable}{@{}p{2.5cm}p{5.0cm}rrrrr@{}}",
            r"\caption{Complete held-unit comparator results. Intervals are 95\% target-resampling intervals.}\label{tab:s-comparators}\\",
            r"\toprule",
            r"Panel & Method & $r$ & $r$ interval & Cosine & RMSE & RMSE interval \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Panel & Method & $r$ & $r$ interval & Cosine & RMSE & RMSE interval \\",
            r"\midrule",
            r"\endhead",
            *rows,
            r"\bottomrule",
            r"\end{longtable}",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "paper/generated_comparator_tables.tex"
    )
    args = parser.parse_args()
    build(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
