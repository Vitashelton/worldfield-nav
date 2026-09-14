#!/usr/bin/env python3
"""Render auditable anchor-view galleries for RelationNav spatial annotation.

This script deliberately makes no semantic or geometric labels.  It only
turns the already-recorded RGB-D anchor candidates into a human-reviewable
artifact with stable scene/view IDs.  Portal planes and relation entities are
entered separately through ``relationnav_annotation.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SCENES = (
    "interior_0405_840145",
    "interior_0135_840032",
    "interior_0121_840013",
    "interior_0093_839966",
)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_scene(root: Path, scene: str):
    src = root / "outputs/formal/GoalPose/P0/anchor_views" / f"anchor_view_candidates_{scene}.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No anchor rows in {src}")
    return rows


def render(root: Path, scene: str, outdir: Path, columns: int) -> Path:
    rows = load_scene(root, scene)
    thumb_w, thumb_h, label_h = 300, 225, 54
    rows_n = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_w, rows_n * (thumb_h + label_h)), "#10151e")
    draw = ImageDraw.Draw(canvas)
    for i, item in enumerate(rows):
        rgb = root / "outputs/formal/GoalPose" / item["rgb"]
        im = Image.open(rgb).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        tile = Image.new("RGB", (thumb_w, thumb_h), "black")
        tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        x, y = (i % columns) * thumb_w, (i // columns) * (thumb_h + label_h)
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="#202b3a")
        draw.text((x + 8, y + thumb_h + 5), item["anchor_view_id"], font=_font(14), fill="white")
        p = item["camera_xyz"]
        draw.text((x + 8, y + thumb_h + 26), f"cam=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})", font=_font(12), fill="#bcd5ee")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{scene}_anchor_gallery.png"
    canvas.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/formal/RelationNav/P1/annotation_review"))
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    args = ap.parse_args()
    for scene in args.scenes:
        print(render(args.root, scene, args.outdir, args.columns))


if __name__ == "__main__":
    main()
