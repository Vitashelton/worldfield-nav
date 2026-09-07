#!/usr/bin/env python3
"""Local-only DeepSeek Vision worker for ExecField P0.

Run this on the trusted PC after copying the P0 directory there. It never opens
a port and reads DEEPSEEK_API_KEY only from the local environment.
"""
from __future__ import annotations
import argparse, base64, json, os, time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT = """You are a semantic navigation-ranking module. The first image is a goal image, the second is the robot's current view, and the remaining eight images are views from candidate local subgoals numbered 0 through 7. Rank candidate IDs by semantic likelihood of moving toward the place shown in the goal image. Do not infer coordinates, geometry, collision, or actions. Reply with JSON only: {\"ranked_candidate_ids\":[...],\"semantic_scores\":[8 floats indexed 0..7],\"rationale\":\"short\"}."""

def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--p0-root", type=Path, required=True); ap.add_argument("--output", type=Path); ap.add_argument("--sleep-s", type=float, default=.2); args=ap.parse_args()
    key=os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise SystemExit("DEEPSEEK_API_KEY is required locally; never place it in the cloud repo.")
    root=args.p0_root; out=args.output or root/"vlm_cache.jsonl"; reqs=json.loads((root/"requests/vlm_request_package.json").read_text())
    with out.open("w") as f:
      for item in reqs:
        blocks=[{"type":"text","text":PROMPT},{"type":"image_url","image_url":{"url":data_url(root/item["goal_image"]),"detail":"low"}},{"type":"image_url","image_url":{"url":data_url(root/item["current_rgb"]),"detail":"low"}}]
        blocks += [{"type":"image_url","image_url":{"url":data_url(root/c["image"]),"detail":"low"}} for c in item["candidates"]]
        body={"model":"deepseek-v4-flash-vision-exp","temperature":0,"messages":[{"role":"user","content":blocks}]}
        request=Request("https://api.deepseek.com/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"})
        raw=json.loads(urlopen(request,timeout=90).read()); text=raw["choices"][0]["message"]["content"]
        try: result=json.loads(text)
        except json.JSONDecodeError: raise RuntimeError(f"non-JSON response for {item['request_id']}: {text[:300]}")
        rank=result.get("ranked_candidate_ids",[]); scores=result.get("semantic_scores",[])
        if sorted(rank)!=list(range(8)) or len(scores)!=8: raise RuntimeError(f"invalid cache schema for {item['request_id']}: {result}")
        f.write(json.dumps({"request_id":item["request_id"],"model":"deepseek-v4-flash-vision-exp","ranking":rank,"semantic_scores":scores,"rationale":result.get("rationale","")},ensure_ascii=False)+"\n"); f.flush(); time.sleep(args.sleep_s)

if __name__ == "__main__": main()
