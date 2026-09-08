#!/usr/bin/env python3
"""Low-frequency, backend-agnostic VLM semantic intent bridge.

The VLM never receives metric poses, NavMesh data, candidate coordinates, or
control actions.  It produces qualitative goal cues and a left/forward/right
preference; the deterministic bridge maps that preference onto planner branches
and the learned Execution Bridge remains responsible for physical selection.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path


SYSTEM = """You are a high-level indoor robot semantic advisor. You do not control
the robot. Infer only qualitative visual task intent from the goal image,
current image and supplied task memory. Return strict JSON only with this exact
schema: {"goal_cues":[string],"heading":"left|forward|right|explore",
"confidence":number,"avoid_cues":[string],"requery_recommended":boolean,
"summary":string}. Never output a coordinate, waypoint, action, trajectory,
candidate ID, map, pose, or route."""


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"VLM did not return JSON: {text[:300]}")
    obj = json.loads(text[start:end + 1])
    required = {"goal_cues", "heading", "confidence", "avoid_cues", "requery_recommended", "summary"}
    if set(obj) != required or obj["heading"] not in {"left", "forward", "right", "explore"}:
        raise RuntimeError(f"invalid semantic intent schema: {obj}")
    obj["confidence"] = float(max(0., min(1., obj["confidence"])))
    return obj


def ollama(endpoint: str, model: str, prompt: str, images: list[str]) -> str:
    payload = {"model": model, "stream": False, "format": "json",
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt, "images": images}]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


def openai(endpoint: str, model: str, prompt: str, images: list[str], key: str) -> str:
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + x}} for x in images]
    payload = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-image", type=Path, required=True); ap.add_argument("--current-image", type=Path, required=True)
    ap.add_argument("--task-state", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    args = ap.parse_args()
    state = json.loads(args.task_state.read_text())
    prompt = "Task state (application-managed, no hidden geometry):\n" + json.dumps(state, ensure_ascii=False) + "\nImages are [goal, current]."
    images = [b64(args.goal_image), b64(args.current_image)]
    if args.backend == "ollama":
        raw = ollama(args.endpoint, args.model, prompt, images)
    else:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key: raise RuntimeError("DEEPSEEK_API_KEY is required for OpenAI-compatible backend")
        raw = openai(args.endpoint, args.model, prompt, images, key)
    result = parse(raw)
    result.update({"backend": args.backend, "model": args.model, "task_state_path": str(args.task_state), "goal_image": str(args.goal_image), "current_image": str(args.current_image)})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
