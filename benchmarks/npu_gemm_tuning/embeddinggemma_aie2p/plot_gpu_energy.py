#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: plot_gpu_energy.py RAW.csv SUMMARY.csv CHART.svg")
    raw_path, summary_path, chart_path = map(Path, sys.argv[1:])
    with raw_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    summary = []
    for label, samples in grouped.items():
        field_values = lambda field: [float(sample[field]) for sample in samples]
        summary.append(
            {
                "label": label,
                "bytes": int(samples[0]["bytes"]),
                "runs": len(samples),
                "tok_s_median": statistics.median(field_values("tok_s")),
                "tok_s_min": min(field_values("tok_s")),
                "tok_s_max": max(field_values("tok_s")),
                "pkg_w_median": statistics.median(field_values("pkg_w")),
                "dyn_w_median": statistics.median(field_values("dyn_w")),
                "pkg_tok_j_median": statistics.median(field_values("pkg_tok_j")),
                "dyn_tok_j_median": statistics.median(field_values("dyn_tok_j")),
            }
        )
    summary.sort(key=lambda row: row["tok_s_median"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    references = {
        row["label"]: row
        for row in summary
        if "bf16" in row["label"].lower() or "w4a8sim" in row["label"].lower()
    }
    plotted = [row for row in summary if row["label"] not in references]
    labels = [short_label(row["label"]) for row in plotted]
    colors = ["#2563eb" for _ in plotted]
    figure, axes = plt.subplots(1, 2, figsize=(15, 8.5), constrained_layout=True)
    metrics = [
        ("tok_s_median", "Throughput (tokens/s)"),
        ("pkg_tok_j_median", "Package efficiency (tokens/joule)"),
    ]
    for axis, (field, title) in zip(axes, metrics):
        plot_values = [float(row[field]) for row in plotted]
        bars = axis.barh(labels, plot_values, color=colors)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        decimals = 1 if field == "pkg_tok_j_median" else 0
        axis.bar_label(
            bars,
            labels=[f"{value:,.{decimals}f}" for value in plot_values],
            padding=3,
            fontsize=8,
        )
        axis.set_xlim(0, max(plot_values) * 1.16)
    bf16 = references.get("EmbeddingGemma-300M.bf16")
    reference_note = ""
    if bf16:
        reference_note = (
            f" | BF16 reference: {bf16['tok_s_median']:,.0f} tokens/s, "
            f"{bf16['pkg_tok_j_median']:,.0f} package tokens/J"
        )
    figure.suptitle(
        "EmbeddingGemma 300M compressed Opus GPU efficiency — "
        f"256-token forward, median of 3{reference_note}",
        fontweight="bold",
    )
    figure.savefig(chart_path)


def short_label(label: str) -> str:
    return label.removeprefix("EmbeddingGemma-300M.")


if __name__ == "__main__":
    main()
