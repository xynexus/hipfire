#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible /v1/embeddings endpoint with JSONL text.

The input remains natural text: documents are truncated to a maximum byte budget
and are never padded. Hipfire currently reports HTTP usage as ceil(UTF-8 bytes / 4),
so the default 1,024-byte cap corresponds to 256 estimated input tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def truncate_utf8(text: str, max_bytes: int) -> str:
    text = text.strip()
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    word_clipped = clipped.rsplit(None, 1)[0] if any(c.isspace() for c in clipped) else clipped
    return word_clipped.strip() or clipped.strip()


def load_documents(path: Path, max_bytes: int) -> list[str]:
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            document = "\n".join(part for part in (title, text) if part)
            if document:
                clipped = truncate_utf8(document, max_bytes)
                if clipped:
                    documents.append(clipped)
    return documents


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def take_batch(documents: list[str], cursor: int, batch_size: int) -> tuple[list[str], int]:
    batch = [documents[(cursor + offset) % len(documents)] for offset in range(batch_size)]
    return batch, cursor + batch_size


def request_embeddings(
    endpoint: str,
    model: str,
    texts: list[str],
    dimensions: int | None,
    timeout: float,
    api_key: str | None,
    max_rate_limit_retries: int,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(max_rate_limit_retries + 1):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            elapsed = time.perf_counter() - started
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code != 429 or attempt == max_rate_limit_retries:
                raise RuntimeError(f"HTTP {error.code} from {endpoint}: {body}") from error
            retry_after = max(float(error.headers.get("Retry-After", "1")), 0.05)
            print(
                f"rate limited; waiting {retry_after:.3f}s before retry "
                f"{attempt + 1}/{max_rate_limit_retries}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_after)
        except urllib.error.URLError as error:
            raise RuntimeError(f"request to {endpoint} failed: {error}") from error
    decoded = json.loads(body)
    data = decoded.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(
            f"expected {len(texts)} embeddings, received "
            f"{len(data) if isinstance(data, list) else 'no data array'}: {decoded}"
        )
    return decoded, elapsed


def summarize(batch_size: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(sample["latency_ms"]) for sample in samples]
    total_seconds = sum(latencies) / 1_000.0
    documents = sum(int(sample["documents"]) for sample in samples)
    estimated_tokens = sum(int(sample["estimated_tokens"]) for sample in samples)
    input_bytes = sum(int(sample["input_bytes"]) for sample in samples)
    return {
        "batch_size": batch_size,
        "requests": len(samples),
        "documents": documents,
        "input_bytes": input_bytes,
        "estimated_tokens": estimated_tokens,
        "mean_estimated_tokens_per_document": estimated_tokens / documents,
        "total_ms": total_seconds * 1_000.0,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "documents_per_second": documents / total_seconds,
        "estimated_tokens_per_second": estimated_tokens / total_seconds,
        "samples": samples,
    }


def parse_batch_sizes(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4,8,16,32,64"))
    parser.add_argument("--requests", type=int, default=3, help="timed requests per batch size")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--max-estimated-tokens", type=int, default=256)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-rate-limit-retries", type=int, default=120)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.requests <= 0
        or args.warmup_requests < 0
        or args.max_estimated_tokens <= 0
        or args.max_rate_limit_retries < 0
    ):
        raise SystemExit(
            "requests and max-estimated-tokens must be positive; "
            "warmup and max-rate-limit-retries may be zero"
        )
    endpoint = args.url.rstrip("/") + "/v1/embeddings"
    max_bytes = args.max_estimated_tokens * 4
    documents = load_documents(args.corpus, max_bytes=max_bytes)
    if not documents:
        raise SystemExit(f"no non-empty documents found in {args.corpus}")
    random.Random(args.seed).shuffle(documents)
    corpus_sha256 = file_sha256(args.corpus)
    api_key = os.environ.get("HIPFIRE_API_KEY")
    print(
        f"loaded {len(documents)} documents; endpoint={endpoint} model={args.model} "
        f"max_bytes={max_bytes}",
        file=sys.stderr,
        flush=True,
    )

    cursor = 0
    results: list[dict[str, Any]] = []
    embedding_dimensions: int | None = None
    for batch_size in args.batch_sizes:
        for warmup_index in range(args.warmup_requests):
            batch, cursor = take_batch(documents, cursor, batch_size)
            request_embeddings(
                endpoint,
                args.model,
                batch,
                args.dimensions,
                args.timeout,
                api_key,
                args.max_rate_limit_retries,
            )
            print(
                f"batch={batch_size} warmup={warmup_index + 1}/{args.warmup_requests} ok",
                file=sys.stderr,
                flush=True,
            )

        samples: list[dict[str, Any]] = []
        for request_index in range(args.requests):
            batch, cursor = take_batch(documents, cursor, batch_size)
            response, elapsed = request_embeddings(
                endpoint,
                args.model,
                batch,
                args.dimensions,
                args.timeout,
                api_key,
                args.max_rate_limit_retries,
            )
            first_embedding = response["data"][0].get("embedding", [])
            current_dimensions = len(first_embedding)
            if embedding_dimensions is None:
                embedding_dimensions = current_dimensions
            elif current_dimensions != embedding_dimensions:
                raise RuntimeError(
                    f"embedding width changed from {embedding_dimensions} to {current_dimensions}"
                )
            usage = response.get("usage") or {}
            estimated_tokens = int(usage.get("prompt_tokens", 0))
            sample = {
                "request": request_index + 1,
                "documents": batch_size,
                "input_bytes": sum(len(text.encode("utf-8")) for text in batch),
                "estimated_tokens": estimated_tokens,
                "latency_ms": elapsed * 1_000.0,
            }
            samples.append(sample)
            print(
                f"batch={batch_size} request={request_index + 1}/{args.requests} "
                f"latency_ms={sample['latency_ms']:.3f} docs={batch_size} "
                f"estimated_tokens={estimated_tokens}",
                file=sys.stderr,
                flush=True,
            )
        summary = summarize(batch_size, samples)
        results.append(summary)
        print(
            f"batch={batch_size} documents_per_second={summary['documents_per_second']:.3f} "
            f"estimated_tokens_per_second={summary['estimated_tokens_per_second']:.3f}",
            file=sys.stderr,
            flush=True,
        )

    report = {
        "schema_version": 1,
        "benchmark": "openai_http_embeddings",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "git_commit": git_commit(),
        "endpoint": endpoint,
        "model": args.model,
        "embedding_dimensions": embedding_dimensions,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": corpus_sha256,
        "corpus_documents_loaded": len(documents),
        "seed": args.seed,
        "max_estimated_tokens_per_document": args.max_estimated_tokens,
        "max_utf8_bytes_per_document": max_bytes,
        "token_accounting": "HTTP usage.prompt_tokens; Hipfire estimates ceil(UTF-8 bytes / 4)",
        "warmup_requests_per_batch_size": args.warmup_requests,
        "timed_requests_per_batch_size": args.requests,
        "http_concurrency": 1,
        "rate_limit_wait_excluded_from_request_latency": True,
        "results": results,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        print(f"wrote {args.output}", file=sys.stderr, flush=True)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
