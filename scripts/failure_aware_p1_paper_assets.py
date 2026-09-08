#!/usr/bin/env python3
"""Render only final rolling-ImageNav assets from the formal evaluator output."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal/TrajectoryGrounding/P1"
TABLES = ROOT / "paper_assets/tables"
FIGURES = ROOT / "paper_assets/figures"


def main() -> None:
    summary = json.loads((OUT / "rolling_summary.json").read_text())["metrics"]
    if any(r["episodes"] < 100 for r in summary):
        raise RuntimeError("Refusing to render paper assets from a non-formal rolling run")
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    for split, name in (("val", "failure_aware_p1_main.csv"), ("unseen", "failure_aware_p1_unseen.csv")):
        rows = [r for r in summary if r["split"] == split]
        with (TABLES / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    methods = ["Planner Geometry", "Learned Execution Bridge", "Outcome Oracle"]
    splits = ["val", "unseen"]
    colors = ["#8c8c8c", "#0072B2", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, metric, title, ymax in zip(axes, ("SR", "SPL"), ("ImageNav Success Rate", "Success weighted by Path Length"), (1.0, 1.0)):
        x = np.arange(len(methods)); w = .34
        for si, split in enumerate(splits):
            values = [next(r[metric] for r in summary if r["method"] == method and r["split"] == split) for method in methods]
            bars = ax.bar(x + (si - .5) * w, values, w, label="Validation" if split == "val" else "Unseen", color=[colors[i] for i in range(len(methods))], alpha=1.0 if si == 0 else .55, edgecolor="white")
            for bar, value in zip(bars, values): ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{100*value:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, ["Geometry\nrule", "Learned\nBridge", "Oracle\nupper bound"]); ax.set_ylim(0, ymax + .13)
        ax.set_ylabel(metric); ax.set_title(title, fontsize=13, weight="bold"); ax.grid(axis="y", alpha=.25)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Failure-Aware Rolling Subgoal Selection: Formal Indoor ImageNav", fontsize=15, weight="bold")
    fig.savefig(FIGURES / "failure_aware_p1_main_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Paired episode matrix: exact same starts/goals for rules and learned bridge.
    records = [json.loads(x) for x in (OUT / "rolling_episode_results.jsonl").read_text().splitlines() if x.strip()]
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.2), sharex=True, constrained_layout=True)
    for ax, split in zip(axes, splits):
        ordered = sorted({r["episode_id"] for r in records if r["split"] == split})
        matrix = []
        for method in methods[:2]:
            lookup = {r["episode_id"]: int(r["success"]) for r in records if r["split"] == split and r["method"] == method}
            matrix.append([lookup[e] for e in ordered])
        ax.imshow(np.asarray(matrix), aspect="auto", cmap=plt.get_cmap("RdYlGn"), vmin=0, vmax=1)
        ax.set_yticks([0, 1], ["Geometry rule", "Learned Bridge"]); ax.set_title(f"Paired episode outcomes — {split}", loc="left", weight="bold")
    axes[-1].set_xlabel("Identical ImageNav episode index (green = success, red = failure)")
    fig.savefig(FIGURES / "failure_aware_p1_paired_episode_outcomes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
