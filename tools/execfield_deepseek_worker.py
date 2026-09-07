#!/usr/bin/env python3
"""Local-only DeepSeek Vision worker for ExecField P0.

Run this on the trusted PC after copying the P0 directory there. It never opens
a port and reads DEEPSEEK_API_KEY only from the local environment.
"""
from __future__ import annotations
import argparse, base64, io, json, os, time
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw

PROMPT = """You are a semantic navigation-ranking module. This labelled contact sheet contains GOAL (the target place), CURRENT (the robot view), and C0 through C7 (candidate local-subgoal views). Rank C0..C7 by semantic likelihood of moving toward the place shown in GOAL. Do not infer coordinates, geometry, collision, or actions. Reply with JSON only: {\"ranked_candidate_ids\":[...],\"semantic_scores\":[8 floats indexed 0..7],\"rationale\":\"short\"}."""

def contact_sheet(root: Path, item: dict) -> str:
    """One labelled image avoids a slow 10-image API request per episode."""
    entries=[("GOAL", root/item["goal_image"]), ("CURRENT", root/item["current_rgb"])]
    entries += [(f"C{c['candidate_id']}", root/c["image"]) for c in item["candidates"]]
    canvas=Image.new("RGB", (1024, 768), "white"); draw=ImageDraw.Draw(canvas)
    for i,(label,path) in enumerate(entries):
        im=Image.open(path).convert("RGB").resize((256,256)); x=(i%4)*256; y=(i//4)*256
        canvas.paste(im,(x,y)); draw.rectangle((x,y,x+56,y+22),fill="black"); draw.text((x+4,y+3),label,fill="white")
    buf=io.BytesIO(); canvas.save(buf,format="JPEG",quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def parse_json(text: str) -> dict:
    text=text.strip()
    if text.startswith("```"):
        text=text.split("\n",1)[1].rsplit("```",1)[0].strip()
    return json.loads(text)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--p0-root", type=Path, required=True); ap.add_argument("--output", type=Path); ap.add_argument("--sleep-s", type=float, default=.2); ap.add_argument("--timeout-s", type=int, default=180); args=ap.parse_args()
    key=os.environ.get("DEEPSEEK_API_KEY")
    if not key: raise SystemExit("DEEPSEEK_API_KEY is required locally; never place it in the cloud repo.")
    root=args.p0_root; out=args.output or root/"vlm_cache.jsonl"; reqs=json.loads((root/"requests/vlm_request_package.json").read_text())
    done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip(): done.add(json.loads(line)["request_id"])
    with out.open("a") as f:
      for n,item in enumerate(reqs,1):
        if item["request_id"] in done: continue
        print(f"[{n}/{len(reqs)}] {item['request_id']}", flush=True)
        blocks=[{"type":"text","text":PROMPT},{"type":"image_url","image_url":{"url":contact_sheet(root,item),"detail":"original"}}]
        body={"model":"deepseek-v4-flash-vision-exp","temperature":0,"messages":[{"role":"user","content":blocks}]}
        request=Request("https://api.deepseek.com/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"})
        raw=json.loads(urlopen(request,timeout=args.timeout_s).read()); text=raw["choices"][0]["message"]["content"]
        try: result=parse_json(text)
        except json.JSONDecodeError: raise RuntimeError(f"non-JSON response for {item['request_id']}: {text[:300]}")
        rank=result.get("ranked_candidate_ids",[]); scores=result.get("semantic_scores",[])
        # Vision models often return C0...C7 despite the requested integer schema.
        try: rank=[int(str(x).lstrip("Cc")) for x in rank]
        except ValueError: pass
        if sorted(rank)!=list(range(8)) or len(scores)!=8: raise RuntimeError(f"invalid cache schema for {item['request_id']}: {result}")
        f.write(json.dumps({"request_id":item["request_id"],"model":"deepseek-v4-flash-vision-exp","ranking":rank,"semantic_scores":scores,"rationale":result.get("rationale","")},ensure_ascii=False)+"\n"); f.flush(); time.sleep(args.sleep_s)

if __name__ == "__main__": main()
