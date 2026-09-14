#!/usr/bin/env python3
"""Low-frequency semantic task-template extraction for GoalPose.

The output is deliberately qualitative.  It is a cached VLM prior, never a
metric target, route, action, candidate identifier, or supervision label.
"""
from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path


SYSTEM = """You are a semantic advisor for an indoor mobile robot. Infer an
object/scene description and one qualitative approach template from a goal
image, a current image, and a user task. You do not control the robot.
Return JSON only, with exactly this schema:
{
  \"target_description\": string,
  \"approach_template\": \"observe_doorway|approach_doorway|inspect_target|room_entrance\",
  \"desired_standoff\": \"close|moderate|far\",
  \"face_target\": boolean,
  \"confidence\": number,
  \"rationale\": string
}
Never output a coordinate, pose, waypoint, route, action, candidate ID,
trajectory, map, depth, NavMesh information, planner parameter, or control."""

TEMPLATES = {"observe_doorway", "approach_doorway", "inspect_target", "room_entrance"}
STANDOFFS = {"close", "moderate", "far"}


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_response(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"VLM did not return JSON: {raw[:300]}")
    result = json.loads(raw[start:end + 1])
    expected = {"target_description", "approach_template", "desired_standoff", "face_target", "confidence", "rationale"}
    if set(result) != expected:
        raise RuntimeError(f"invalid GoalPose semantic schema: {result}")
    if result["approach_template"] not in TEMPLATES or result["desired_standoff"] not in STANDOFFS:
        raise RuntimeError(f"invalid GoalPose template: {result}")
    result["confidence"] = float(max(0.0, min(1.0, float(result["confidence"]))))
    result["face_target"] = bool(result["face_target"])
    return result


def call_ollama(endpoint: str, model: str, prompt: str, images: list[str]) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt, "images": images},
        ],
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-image", type=Path, required=True)
    parser.add_argument("--current-image", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    args = parser.parse_args()

    prompt = (
        "User task: " + args.task + "\n"
        "The images are ordered [goal image, current robot image]. Return only the schema."
    )
    result = parse_response(call_ollama(
        args.endpoint, args.model, prompt,
        [image_b64(args.goal_image), image_b64(args.current_image)],
    ))
    result.update({
        "backend": "ollama",
        "model": args.model,
        "task": args.task,
        "goal_image": str(args.goal_image),
        "current_image": str(args.current_image),
        "semantic_role": "cached_task_template_only",
        "not_geometry_ground_truth": True,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
