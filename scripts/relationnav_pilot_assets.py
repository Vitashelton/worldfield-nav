#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "outputs/formal/RelationNav/P2"
P1 = ROOT / "outputs/formal/RelationNav/P1/dataset"
ASSET = ROOT / "paper_assets"

def main():
    summary = json.loads((P2 / "summary.json").read_text())["metrics"]
    out_table = ASSET / "tables" / "relationnav_pilot_summary.csv"; out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    methods = ["Arrival-only", "Same-goal retry", "Relation-verified", "Relation-preserving recovery", "Oracle"]
    splits = ["train", "heldout"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    x = np.arange(len(methods)); width=.36
    for j, split in enumerate(splits):
        rows = {r["method"]: r for r in summary if r["split"] == split}
        ax[0].bar(x + (j-.5)*width, [100*rows[m]["Semantic_Task_Success"] for m in methods], width, label=split)
        ax[1].bar(x + (j-.5)*width, [100*rows[m]["False_Completion_Rate"] for m in methods], width, label=split)
    ax[0].set_title("Semantic task success (pilot)"); ax[1].set_title("False completion rate (pilot)")
    for a in ax:
        a.set_xticks(x, ["Arrival\nonly", "Same-goal\nretry", "Relation\nverified", "Preserving\nrecovery", "Oracle"], rotation=20)
        a.set_ylim(0, 105); a.set_ylabel("percent"); a.grid(axis="y", alpha=.25); a.legend()
    fig.suptitle("RelationNav: fixed executor vs relation-aware execution\n400 Habitat-GS episodes; pilot diagnostic, not final benchmark", fontsize=13)
    fig.tight_layout(); fig.savefig(ASSET / "figures" / "relationnav_pilot_results.png", bbox_inches="tight"); plt.close(fig)

    # Episode-level transitions: same episode, semantic outcome by method.
    rows = [json.loads(x) for x in (P2 / "episode_results.jsonl").read_text().splitlines()]
    by = {}
    for r in rows: by.setdefault(r["episode_id"], {})[r["method"]] = r
    vals = ["Arrival-only", "Relation-preserving recovery", "Oracle"]
    mat = np.array([[by[e][m]["semantic_task_complete"] for e in sorted(by)] for m in vals], float)
    fig, ax = plt.subplots(figsize=(14, 2.8), dpi=180)
    ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(vals)), vals); ax.set_xlabel("same deterministic episode"); ax.set_title("Episode-level semantic completion (pilot)")
    ax.set_xticks([]); fig.tight_layout(); fig.savefig(ASSET / "figures" / "relationnav_episode_completion_matrix.png", bbox_inches="tight"); plt.close(fig)

    # A readable case reel from existing Habitat-GS rendered candidate overlays.
    manifest = json.loads((P1 / "dataset_manifest.json").read_text())["episodes"]
    cases=[]
    for ep in manifest:
        if len(cases) >= 16: break
        if not ep["phases"]: continue
        p=ep["phases"][0]; image=P1 / p["candidate_overlay"]
        if image.exists(): cases.append((ep["scene_id"], ep["episode_id"], p["relation"], image))
    video_dir = ASSET / "videos"; video_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        writer=imageio.get_writer(video_dir / "relationnav_habitat_pilot.mp4", fps=2, codec="libx264", macro_block_size=1)
        for scene,eid,rel,image in cases:
            im=Image.open(image).convert("RGB").resize((768,768)); draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,768,58), fill=(15,15,15)); draw.text((18,16), f"Habitat-GS | {scene} | {eid} | stage={rel}", fill="white")
            writer.append_data(np.asarray(im))
        writer.close()
    except Exception as exc:
        (video_dir / "relationnav_video_error.txt").write_text(repr(exc)+"\n")

    report = ROOT / "docs/results/RELATIONNAV_PILOT_RESULTS.md"; report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("""# RelationNav Pilot Results\n\nThis is an internal pilot diagnostic from the existing four-scene, 400-episode manifest. It is not the final benchmark and does not claim a new planner. Habitat-GS supplies deterministic RGB-D, pose, NavMesh and privileged relation predicates.\n\n## Main observation\n\nArrival-only execution reports nominal completion even when the relation predicate is false. Relation-preserving recovery keeps the same entity-relation contract and selects a deeper admissible realization after a failed predicate.\n\n## Assets\n\n- `paper_assets/tables/relationnav_pilot_summary.csv`\n- `paper_assets/figures/relationnav_pilot_results.png`\n- `paper_assets/figures/relationnav_episode_completion_matrix.png`\n- `paper_assets/videos/relationnav_habitat_pilot.mp4`\n\nThe shallow boundary realization used by this pilot is a diagnostic stress protocol; it must not be presented as the final paper benchmark without scene-scale automatic generation and an executor protocol review.\n""")
    print(json.dumps({"table":str(out_table),"figures":["paper_assets/figures/relationnav_pilot_results.png","paper_assets/figures/relationnav_episode_completion_matrix.png"],"video":"paper_assets/videos/relationnav_habitat_pilot.mp4","cases":len(cases)}, indent=2))

if __name__ == "__main__": main()
