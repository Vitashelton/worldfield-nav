#!/usr/bin/env python3
"""Trusted-PC-only DeepSeek Vision cache worker for Trajectory Grounding P1."""
from __future__ import annotations

import argparse, base64, io, json, os, time
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image

PROMPT = """You rank candidate local navigation trajectories by semantic relevance only.
The first image is GOAL and the second is CURRENT with trajectories T1..T8 overlaid.
Rank T1..T8 by how semantically likely each trajectory is to approach GOAL. Do not infer
coordinates, collision, obstacle geometry, actions, or hidden poses. Reply JSON only:
{"ranked_trajectory_ids":[1,2,3,4,5,6,7,8],"semantic_scores":[8 floats indexed T1..T8],"rationale":"short"}."""

def data_url(paths: list[Path]) -> str:
    canvas = Image.new("RGB", (512, 256), "white")
    for i, path in enumerate(paths): canvas.paste(Image.open(path).convert("RGB").resize((256, 256)), (i * 256, 0))
    buf = io.BytesIO(); canvas.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--p1-root", type=Path, required=True); ap.add_argument("--sleep-s", type=float, default=.2); args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise SystemExit("DEEPSEEK_API_KEY is required on the trusted PC; never copy it to cloud.")
    requests = json.loads((args.p1_root / "vlm_request_package.json").read_text()); out = args.p1_root / "vlm_cache.jsonl"; done = set()
    if out.exists(): done = {json.loads(line)["request_id"] for line in out.read_text().splitlines() if line.strip()}
    with out.open("a") as handle:
        for index, item in enumerate(requests, 1):
            if item["request_id"] in done: continue
            print(f"[{index}/{len(requests)}] {item['request_id']}", flush=True)
            content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": data_url([args.p1_root / item["goal_image"], args.p1_root / item["trajectory_overlay"]]), "detail": "original"}}]
            body = {"model": "deepseek-v4-flash-vision-exp", "temperature": 0, "messages": [{"role": "user", "content": content}]}
            request = Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json", "Authorization":f"Bearer {key}"})
            text = json.loads(urlopen(request, timeout=180).read())["choices"][0]["message"]["content"].strip().strip("`")
            if text.startswith("json"): text = text[4:].strip()
            result = json.loads(text); rank = [int(str(x).lstrip("Tt")) for x in result["ranked_trajectory_ids"]]; scores = result["semantic_scores"]
            if sorted(rank) != list(range(1, 9)) or len(scores) != 8: raise RuntimeError(f"bad schema: {item['request_id']}")
            handle.write(json.dumps({"request_id":item["request_id"], "model":"deepseek-v4-flash-vision-exp", "ranking":rank, "semantic_scores":scores, "rationale":result.get("rationale", "")}, ensure_ascii=False) + "\n"); handle.flush(); time.sleep(args.sleep_s)

if __name__ == "__main__": main()
