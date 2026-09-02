#!/usr/bin/env python3
"""Review P0 counterfactual diversity without rerunning Habitat-GS."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "paper_assets/tables/b1_p0_revelation_candidates.csv"
OUT = ROOT / "outputs/formal/B1/p0"
TABLES = ROOT / "paper_assets/tables"
FIGURES = ROOT / "paper_assets/figures"
CONTEXTS = ("open_space", "turn_corner", "doorway_occlusion")
FAMILIES = ("straight", "left", "right", "mixed_turn")


def q(values: list[float], p: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = list(csv.DictReader(INPUT.open()))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["candidate_id"])].append(row)

    starts: list[dict[str, object]] = []
    for (scene, candidate), branch_rows in grouped.items():
        areas = [float(row["revelation_m2_3s"]) for row in branch_rows]
        context = branch_rows[0]["context"]
        mean = float(np.mean(areas))
        starts.append({
            "scene_id": scene,
            "candidate_id": int(candidate),
            "context": context,
            "R_max_m2": max(areas),
            "R_min_m2": min(areas),
            "R_mean_m2": mean,
            "delta_R_m2": max(areas) - min(areas),
            "coefficient_of_variation": float(np.std(areas) / mean) if mean else 0.0,
            "old_eligible": branch_rows[0]["eligible_after_threshold"].lower() == "true",
            "mask_data_available": False,
        })

    rmax_values = [float(s["R_max_m2"]) for s in starts if float(s["R_max_m2"]) > 0]
    delta_values = [float(s["delta_R_m2"]) for s in starts]
    rmax_threshold = q(rmax_values, 25) or 0.0
    delta_threshold = q(delta_values, 75) or 0.0
    for start in starts:
        start["magnitude_proxy_eligible"] = (
            float(start["R_max_m2"]) >= rmax_threshold
            and float(start["delta_R_m2"]) >= delta_threshold
        )

    context_summary: dict[str, dict[str, object]] = {}
    for context in CONTEXTS:
        subset = [s for s in starts if s["context"] == context]
        context_summary[context] = {
            "count": len(subset),
            "R_max_m2": {k: q([float(s["R_max_m2"]) for s in subset], k) for k in (25, 50, 75)},
            "delta_R_m2": {k: q([float(s["delta_R_m2"]) for s in subset], k) for k in (25, 50, 75)},
            "C_mask": {"p25": None, "median": None, "p75": None},
            "mask_status": "not_persisted_in_P0_outputs",
        }
    action_summary = {
        family: {
            "count": sum(row["action_family"] == family for row in rows),
            "revelation_m2": {k: q([float(row["revelation_m2_3s"]) for row in rows if row["action_family"] == family], k) for k in (25, 50, 75)},
        }
        for family in FAMILIES
    }

    omega = math.radians(3.0) / 0.2
    control_audit: dict[str, object] = {"all_rows_one_to_one": True, "families": {}}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["action_family"] == family]
        checks = []
        for row in family_rows:
            controls = json.loads(row["continuous_control"])
            values = [float(control["omega_radps"]) for control in controls]
            if family == "straight": expected = all(abs(value) < 1e-8 for value in values)
            elif family == "left": expected = all(abs(value - omega) < 1e-8 for value in values)
            elif family == "right": expected = all(abs(value + omega) < 1e-8 for value in values)
            else: expected = all(abs(value - omega) < 1e-8 for value in values[:7]) and all(abs(value + omega) < 1e-8 for value in values[7:])
            checks.append(len(controls) == 15 and all(abs(float(control["v_mps"]) - .35) < 1e-8 and abs(float(control["dt_s"]) - .2) < 1e-8 for control in controls) and expected)
        ratios = [float(row["translation_ratio"]) for row in family_rows]
        control_audit["families"][family] = {"rows": len(family_rows), "continuous_sequence_length": 15, "discrete_proxy": "turn_left/turn_right then move_forward per tick", "one_to_one": all(checks), "translation_ratio_p25_median_p75": [q(ratios, p) for p in (25, 50, 75)]}
        control_audit["all_rows_one_to_one"] = bool(control_audit["all_rows_one_to_one"] and all(checks))

    # Code-path audit: both initial and future observations are converted to
    # global X/Z cells before set subtraction. There is no tensor crop index.
    source = (ROOT / "scripts/b1_p0_revelation_probe.py").read_text()
    coordinate_audit = {
        "passed": "world = camera @ c2w[:3, :3].T + c2w[:3, 3]" in source and "observed - initial_cells" in source,
        "support": "global world-coordinate X/Z cell sets",
        "crop_index_comparison": False,
        "mask_artifacts_persisted": False,
        "limitation": "P0 persisted only scalar revelation areas, not per-branch cell masks or start poses; C_mask and mask examples cannot be reconstructed without rerunning Habitat-GS.",
    }

    # Required scalar figures are valid; the mask figure is an explicit audit
    # placeholder rather than fabricated spatial evidence.
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"open_space": "#377eb8", "turn_corner": "#ff7f00", "doorway_occlusion": "#4daf4a"}
    plt.figure(figsize=(7, 5))
    for context in CONTEXTS:
        subset = [s for s in starts if s["context"] == context]
        plt.scatter([s["R_max_m2"] for s in subset], [s["delta_R_m2"] for s in subset], s=20, alpha=.65, label=context, c=colors[context])
    plt.axvline(rmax_threshold, color="black", ls="--", lw=1, label=f"R_max P25={rmax_threshold:.3f}")
    plt.axhline(delta_threshold, color="crimson", ls="--", lw=1, label=f"delta_R P75={delta_threshold:.3f}")
    plt.xlabel("R_max (m²)"); plt.ylabel("delta_R = R_max - R_min (m²)"); plt.title("P0 magnitude vs counterfactual contrast"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES / "b1_p0_context_magnitude_vs_contrast.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 5)); plt.boxplot([[s["delta_R_m2"] for s in starts if s["context"] == c] for c in CONTEXTS], tick_labels=CONTEXTS, showfliers=True); plt.axhline(delta_threshold, color="crimson", ls="--", label="global delta_R P75"); plt.ylabel("delta_R (m²)"); plt.title("Counterfactual magnitude contrast by context"); plt.xticks(rotation=20); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / "b1_p0_counterfactual_contrast_boxplot.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 5))
    for context in CONTEXTS:
        values = sorted(float(s["delta_R_m2"]) for s in starts if s["context"] == context)
        plt.hist(values, bins=18, alpha=.45, label=context, color=colors[context])
    plt.axvline(delta_threshold, color="crimson", ls="--", label=f"delta_R P75={delta_threshold:.3f}"); plt.xlabel("delta_R (m²)"); plt.ylabel("start count"); plt.title("P0 delta revelation distribution"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / "b1_p0_delta_revelation_distribution.png", dpi=180); plt.close()

    plt.figure(figsize=(12, 5)); plt.axis("off"); plt.title("P0 revelation mask pair examples — audit limitation", pad=18)
    plt.text(.5, .62, "Per-branch newly-observed cell masks were not persisted by P0.", ha="center", va="center", fontsize=15, color="crimson")
    plt.text(.5, .46, "The scalar areas are reusable, but spatial pairwise IoU/Jaccard\nand example masks require a future mask-persistence run.", ha="center", va="center", fontsize=12)
    plt.text(.5, .25, "No Habitat-GS probe was rerun in this review.", ha="center", va="center", fontsize=11)
    plt.savefig(FIGURES / "b1_p0_revelation_mask_pair_examples.png", dpi=180, bbox_inches="tight"); plt.close()

    scene_context = []
    for scene in sorted({str(s["scene_id"]) for s in starts}):
        for context in CONTEXTS:
            subset = [s for s in starts if s["scene_id"] == scene and s["context"] == context]
            scene_context.append({"scene_id": scene, "context": context, "old_eligible": sum(bool(s["old_eligible"]) for s in subset), "magnitude_proxy_eligible": sum(bool(s["magnitude_proxy_eligible"]) for s in subset), "final_spatial_eligible": None})
    write_csv(TABLES / "b1_p0_counterfactual_start_metrics.csv", starts)
    write_csv(TABLES / "b1_p0_counterfactual_eligible_by_scene_context.csv", scene_context)
    review = {
        "source": "existing B1 P0 253 unique starts / 1012 branch probes",
        "coordinate_audit": coordinate_audit,
        "context_summary": context_summary,
        "action_family_revelation_magnitude": action_summary,
        "control_audit": control_audit,
        "eligibility": {
            "formula": "R_max >= positive-area R_max P25 AND delta_R >= delta_R P75 AND C_mask >= a distribution-derived C_mask threshold",
            "R_max_threshold_m2": rmax_threshold,
            "delta_R_threshold_m2": delta_threshold,
            "C_mask_threshold": None,
            "rationale": "P25 retains nontrivial revelation; P75 selects the upper-quartile magnitude contrast. Spatial criterion is not evaluable because masks were not persisted.",
            "magnitude_proxy_eligible_starts": sum(bool(s["magnitude_proxy_eligible"]) for s in starts),
            "final_spatial_eligible_starts": None,
        },
        "old_eligible_starts": sum(bool(s["old_eligible"]) for s in starts),
        "figures": [str((FIGURES / name).relative_to(ROOT)) for name in ("b1_p0_context_magnitude_vs_contrast.png", "b1_p0_counterfactual_contrast_boxplot.png", "b1_p0_delta_revelation_distribution.png", "b1_p0_revelation_mask_pair_examples.png")],
    }
    (OUT / "counterfactual_review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
