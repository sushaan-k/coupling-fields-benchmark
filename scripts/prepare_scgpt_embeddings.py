#!/usr/bin/env python3
"""Extract Ensembl-aligned gene-token embeddings from an official scGPT checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hgnc_symbols(path: Path, vocabulary):
    """Choose one scGPT vocabulary symbol for each Ensembl gene ID."""
    chosen = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            gene_id = row["ensembl_gene_id"]
            if not gene_id:
                continue
            approved = row["symbol"]
            aliases = []
            for field in ("prev_symbol", "alias_symbol"):
                aliases.extend(value for value in row[field].split("|") if value)
            candidates = [approved] + sorted(set(aliases))
            symbol = next((value for value in candidates if value in vocabulary), None)
            if symbol is not None:
                chosen[gene_id] = (approved, symbol)
    return chosen


def extract(checkpoint_path: Path, vocab_path: Path, hgnc_path: Path, output: Path):
    import torch

    checkpoint_sha256 = checksum(checkpoint_path)
    vocab_sha256 = checksum(vocab_path)
    hgnc_sha256 = checksum(hgnc_path)
    vocabulary = json.loads(vocab_path.read_text())
    symbols = load_hgnc_symbols(hgnc_path, vocabulary)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight = state["encoder.embedding.weight"].detach().cpu().numpy()
    norm_weight = state["encoder.enc_norm.weight"].detach().cpu().numpy()
    norm_bias = state["encoder.enc_norm.bias"].detach().cpu().numpy()
    if weight.shape[0] != len(vocabulary):
        raise ValueError("scGPT vocabulary and embedding table have different sizes")

    gene_ids = sorted(symbols)
    approved_symbols = np.asarray([symbols[gene_id][0] for gene_id in gene_ids])
    vocab_symbols = np.asarray([symbols[gene_id][1] for gene_id in gene_ids])
    rows = np.asarray([vocabulary[symbol] for symbol in vocab_symbols])
    embedding = weight[rows].astype(np.float64)
    variance = embedding.var(axis=1, keepdims=True)
    embedding = (
        (embedding - embedding.mean(axis=1, keepdims=True))
        / np.sqrt(variance + 1e-5)
        * norm_weight
        + norm_bias
    ).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite scGPT gene-token embedding")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        label=np.asarray("scGPT whole-human gene-token embeddings"),
        gene_names=approved_symbols,
        vocabulary_symbols=vocab_symbols,
        gene_ids=np.asarray(gene_ids),
        embedding=embedding,
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        vocab_sha256=np.asarray(vocab_sha256),
        hgnc_sha256=np.asarray(hgnc_sha256),
    )
    return {
        "n_genes": len(gene_ids),
        "dimension": int(embedding.shape[1]),
        "checkpoint_sha256": checkpoint_sha256,
        "vocab_sha256": vocab_sha256,
        "hgnc_sha256": hgnc_sha256,
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("hgnc", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            extract(args.checkpoint, args.vocabulary, args.hgnc, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
