#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepare and score a deterministic SciFact retrieval subset for Qwen3 NPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from compare_qwen3_embedding_reference import encode


DEFAULT_REVISION = "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_subset(
    query_rows: list[dict[str, object]],
    corpus_rows: list[dict[str, object]],
    qrel_rows: list[dict[str, object]],
    query_count: int,
    corpus_count: int,
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, set[str]]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in qrel_rows:
        if int(str(row["score"])) > 0:
            qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    query_ids = sorted(qrels, key=lambda value: (int(value), value))[:query_count]
    query_by_id = {str(row["_id"]): row for row in query_rows}
    corpus_by_id = {str(row["_id"]): row for row in corpus_rows}
    missing_queries = [query_id for query_id in query_ids if query_id not in query_by_id]
    if missing_queries:
        raise ValueError(f"SciFact queries missing IDs {missing_queries[:8]}")
    relevant = set().union(*(qrels[query_id] for query_id in query_ids))
    missing_corpus = sorted(relevant - corpus_by_id.keys())
    if missing_corpus:
        raise ValueError(f"SciFact corpus missing relevant IDs {missing_corpus[:8]}")
    if len(relevant) > corpus_count:
        raise ValueError(
            f"{query_count} queries require {len(relevant)} relevant documents, exceeding corpus_count={corpus_count}"
        )
    distractors = sorted(
        (corpus_id for corpus_id in corpus_by_id if corpus_id not in relevant),
        key=lambda corpus_id: hashlib.sha256(f"{seed}:{corpus_id}".encode()).digest(),
    )[: corpus_count - len(relevant)]
    corpus_ids = sorted(relevant) + distractors
    return (
        [query_by_id[query_id] for query_id in query_ids],
        [corpus_by_id[corpus_id] for corpus_id in corpus_ids],
        {query_id: qrels[query_id] for query_id in query_ids},
    )


def tokenize_fixture(
    source: Path,
    queries: list[dict[str, object]],
    corpus: list[dict[str, object]],
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(source)
    texts = [QUERY_PROMPT + str(row["text"]) for row in queries]
    texts.extend("\n".join(part for part in (str(row.get("title", "")), str(row["text"])) if part) for row in corpus)
    token_ids = [tokenizer.encode(text, add_special_tokens=True, truncation=False) for text in texts]
    oversized = [index for index, tokens in enumerate(token_ids) if len(tokens) > 2048]
    if oversized:
        raise ValueError(f"SciFact subset inputs exceed 2048 tokens at {oversized[:8]}")
    return token_ids


def dcg_at_10(ranked_ids: list[str], relevant: set[str]) -> float:
    return sum(1.0 / np.log2(rank + 2.0) for rank, corpus_id in enumerate(ranked_ids[:10]) if corpus_id in relevant)


def ndcg_at_10(scores: np.ndarray, corpus_ids: list[str], relevant: set[str]) -> float:
    ranked = sorted(
        zip(corpus_ids, scores.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    actual = dcg_at_10([corpus_id for corpus_id, _ in ranked], relevant)
    ideal = dcg_at_10(sorted(relevant), relevant)
    return actual / ideal if ideal else 0.0


def encode_bf16(source: Path, token_ids: list[list[int]], device: str) -> torch.Tensor:
    model = AutoModel.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    indexed = sorted(enumerate(token_ids), key=lambda item: (len(item[1]), item[0]))
    output: list[torch.Tensor | None] = [None] * len(token_ids)
    for offset in range(0, len(indexed), 16):
        chunk = indexed[offset : offset + 16]
        embeddings, _, _ = encode(model, [tokens for _, tokens in chunk], device)
        for embedding, (index, _) in zip(embeddings, chunk, strict=True):
            output[index] = embedding
    return torch.stack([value for value in output if value is not None])


def prepare(args: argparse.Namespace) -> int:
    queries_path = args.dataset_root / "queries.jsonl"
    corpus_path = args.dataset_root / "corpus.jsonl"
    qrels_path = args.dataset_root / "qrels/test.jsonl"
    queries, corpus, qrels = select_subset(
        read_jsonl(queries_path),
        read_jsonl(corpus_path),
        read_jsonl(qrels_path),
        args.query_count,
        args.corpus_count,
        args.seed,
    )
    token_ids = tokenize_fixture(args.source, queries, corpus)
    fixture = {
        "schema": "hipfire.qwen3_scifact_subset.v1",
        "dataset": "mteb/scifact",
        "revision": args.revision,
        "source_sha256": {
            "queries.jsonl": sha256(queries_path),
            "corpus.jsonl": sha256(corpus_path),
            "qrels/test.jsonl": sha256(qrels_path),
        },
        "seed": args.seed,
        "query_prompt": QUERY_PROMPT,
        "document_prompt": "",
        "query_ids": [str(row["_id"]) for row in queries],
        "corpus_ids": [str(row["_id"]) for row in corpus],
        "qrels": {query_id: sorted(values) for query_id, values in qrels.items()},
        "token_lengths": [len(tokens) for tokens in token_ids],
        "token_ids": token_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.token_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    args.token_output.write_text(json.dumps(token_ids) + "\n")
    print(json.dumps({key: fixture[key] for key in ("schema", "revision", "query_ids", "token_lengths")}, indent=2))
    return 0


def score(args: argparse.Namespace) -> int:
    fixture = json.loads(args.fixture.read_text())
    npu_record = json.loads(args.npu_output.read_text())
    token_ids = fixture["token_ids"]
    if npu_record["token_ids"] != token_ids:
        raise ValueError("NPU output token IDs do not match the pinned SciFact fixture")
    npu = torch.tensor(npu_record["embeddings"], dtype=torch.float32)
    bf16 = encode_bf16(args.source, token_ids, args.device)
    query_count = len(fixture["query_ids"])
    corpus_ids = fixture["corpus_ids"]
    npu_scores = (npu[:query_count] @ npu[query_count:].T).numpy()
    bf16_scores = (bf16[:query_count] @ bf16[query_count:].T).numpy()
    per_query = []
    for row, query_id in enumerate(fixture["query_ids"]):
        relevant = set(fixture["qrels"][query_id])
        per_query.append(
            {
                "query_id": query_id,
                "bf16_ndcg_at_10": ndcg_at_10(bf16_scores[row], corpus_ids, relevant),
                "npu_ndcg_at_10": ndcg_at_10(npu_scores[row], corpus_ids, relevant),
            }
        )
    bf16_ndcg = float(np.mean([row["bf16_ndcg_at_10"] for row in per_query]))
    npu_ndcg = float(np.mean([row["npu_ndcg_at_10"] for row in per_query]))
    degradation = (bf16_ndcg - npu_ndcg) / bf16_ndcg if bf16_ndcg else 0.0
    result = {
        "schema": "hipfire.qwen3_scifact_quality.v1",
        "fixture": str(args.fixture),
        "queries": query_count,
        "corpus": len(corpus_ids),
        "bf16_ndcg_at_10": bf16_ndcg,
        "npu_ndcg_at_10": npu_ndcg,
        "relative_degradation": degradation,
        "maximum_allowed_degradation": 0.01,
        "admitted": degradation <= 0.01,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["admitted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--token-output", type=Path, required=True)
    prepare_parser.add_argument("--revision", default=DEFAULT_REVISION)
    prepare_parser.add_argument("--query-count", type=int, default=16)
    prepare_parser.add_argument("--corpus-count", type=int, default=128)
    prepare_parser.add_argument("--seed", default="hipfire-qwen3-scifact-v1")
    prepare_parser.set_defaults(run=prepare)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--fixture", type=Path, required=True)
    score_parser.add_argument("--source", type=Path, required=True)
    score_parser.add_argument("--npu-output", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--device", default="cuda")
    score_parser.set_defaults(run=score)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
