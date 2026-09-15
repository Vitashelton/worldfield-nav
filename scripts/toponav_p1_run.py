#!/usr/bin/env python3
"""Run the TopoNav Harness P1 benchmark with real Habitat execution.

The VLM receives symbolic topology descriptions and selected RGB evidence only.
Metric anchors, NavMesh queries and hidden goal poses remain inside this
executor/evaluator.  Every VLM request/response is content-addressed in a
JSONL cache so a benchmark replay never changes a model call silently.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import habitat_sim
import numpy as np
from PIL import Image

from relationnav_p1_generate import make_sim, render, shortest, yaw_rotation
from toponav_harness import (AgentMemory, CompiledContext, ExecutionFeedback,
                             Outcome, RelationState, SelectiveEvidenceRetriever,
                             Tool, ToolCall, TopologyContextCompiler,
                             TopologyToolValidator)

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "outputs/formal/TopoNav/P1"
MANIFEST = P1 / "task_manifest.json"
GRAPH = ROOT / "outputs/formal/TopoNav/P1/semantic_topology.json"
METHODS = ("Direct-VLM", "FullTopo-VLM", "FullTopo+Validator", "History-Agent", "TopoNav-Harness")
MAX_STEPS = 6


def short(node_id: str) -> str:
    return node_id.rsplit(":", 1)[-1].replace("_", " ")


def rgb64(rgb: np.ndarray) -> str:
    import io
    b = io.BytesIO(); Image.fromarray(rgb).save(b, format="JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode()


def cache_load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists(): return {}
    return {r["key"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def qwen(prompt: dict[str, Any], images: list[np.ndarray], cache: dict[str, dict],
         cache_file: Path, timeout: int = 120) -> tuple[dict[str, Any], float, bool]:
    key = hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if key in cache and "error" not in cache[key].get("response", {}):
        return cache[key]["response"], float(cache[key].get("latency_s", 0.0)), True
    schema = {"tool": "NAVIGATE | OBSERVE | BACKTRACK | STOP", "target": "candidate ID or null"}
    system = ("You are a high-level indoor robot navigation agent. Return JSON only. "
              "Never output coordinates, paths, controls, or explanations. Choose one listed tool and, "
              "for NAVIGATE/OBSERVE, a listed candidate target. Do not invent IDs.")
    content = system + "\nRequired JSON schema: " + json.dumps(schema) + \
              "\nNavigation context:\n" + json.dumps(prompt, ensure_ascii=False)
    body = {"model": "qwen3-vl:8b-instruct", "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 64},
            "messages": [{"role": "user", "content": content,
                          "images": [rgb64(x) for x in images]}]}
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        raw = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
        text = raw["message"]["content"]
        response = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception as exc:
        detail = getattr(exc, "read", lambda: b"")()
        response = {"tool": "STOP", "target": None,
                    "error": f"VLM_ERROR:{type(exc).__name__}:{detail.decode(errors='replace')[:400]}"}
    latency = time.perf_counter() - start
    row = {"key": key, "prompt": prompt, "response": response, "latency_s": latency,
           "images": len(images), "backend": "ollama/qwen3-vl:8b-instruct"}
    with cache_file.open("a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cache[key] = row
    return response, latency, False


def scene_view(graph: dict, scene: str, full: bool) -> dict[str, Any]:
    nodes = [n for n in graph["nodes"] if n["scene_id"] == scene]
    edges = [e for e in graph["edges"] if e.get("edge_type") == "SPATIAL_ADJACENCY" and e["source"].startswith(scene + ":")]
    if not full:
        return {"node_ids": [n["node_id"] for n in nodes]}
    return {"nodes": [{"id": n["node_id"], "kind": n["kind"]} for n in nodes],
            "transitions": [{"id": e["via"], "from": e["source"], "to": e["target"], "type": e.get("semantic_type", "transition")} for e in edges]}


def set_agent(agent, pos: np.ndarray, yaw: float) -> None:
    state = agent.get_state(); state.position = pos.astype(np.float32); state.rotation = yaw_rotation(float(yaw))
    agent.set_state(state, reset_sensors=True)


def path_execute(sim, agent, target: np.ndarray, max_move: float = 30.0) -> tuple[ExecutionFeedback, list[list[float]], float]:
    start = np.asarray(agent.get_state().position, np.float32)
    ok, distance = shortest(sim, start, target)
    if not ok:
        return ExecutionFeedback(Outcome.NO_PROGRESS, None, 0., float("inf"), False, "CROSS", False), [start.tolist()], 0.
    query = habitat_sim.ShortestPath(); query.requested_start = start; query.requested_end = target
    sim.pathfinder.find_path(query)
    pts = [np.asarray(p, np.float32) for p in query.points]
    travelled, trace = 0.0, [pts[0].tolist()]
    for p in pts[1:]:
        if travelled + float(np.linalg.norm(p - np.asarray(trace[-1], np.float32))) > max_move: break
        travelled += float(np.linalg.norm(p - np.asarray(trace[-1], np.float32)))
        trace.append(p.tolist())
    end = np.asarray(trace[-1], np.float32)
    if len(trace) > 1:
        d = end - np.asarray(trace[-2], np.float32); yaw = math.atan2(float(-d[0]), float(-d[2]))
    else: yaw = 0.
    set_agent(agent, end, yaw)
    reached = bool(np.linalg.norm(end - target) <= 0.35)
    status = Outcome.SUCCESS if reached else Outcome.NO_PROGRESS
    return ExecutionFeedback(status, None, travelled, max(0., float(distance - travelled)), False, "CROSS", reached), trace, float(distance)


def transition_realization(sim, source: dict, target: dict, portal: dict) -> np.ndarray:
    """Make a metric realization for exactly one symbolic portal transition.

    The executor aims a short distance *past the selected portal* toward its
    declared target area, rather than aiming at a room centre.  Thus selecting
    a different named edge changes physical execution; the VLM never sees the
    coordinates used here.
    """
    p = np.asarray(portal["geometry"]["center_xz"], np.float32)
    dst = np.asarray(target["geometry"]["center_xz"], np.float32)
    direction = dst - p
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        direction = np.asarray(target["geometry"]["center_xz"], np.float32) - np.asarray(source["geometry"]["center_xz"], np.float32)
        norm = max(float(np.linalg.norm(direction)), 1e-5)
    inside = p + 0.85 * direction / norm
    return np.asarray(sim.pathfinder.snap_point(np.array([inside[0], 0., inside[1]], np.float32)), np.float32)


def source_realization(sim, source: dict, portal: dict) -> np.ndarray:
    """Deterministic valid start just inside a source region at a known portal.

    Structure-polygon centroids can fall in non-walkable furniture or holes.
    This fallback is still goal-independent: it uses only the declared source
    region and one adjacent portal, not task destination or NavMesh routing.
    """
    p = np.asarray(portal["geometry"]["center_xz"], np.float32)
    c = np.asarray(source["geometry"]["center_xz"], np.float32)
    d = c - p; norm = max(float(np.linalg.norm(d)), 1e-5)
    inside = p + .85 * d / norm
    return np.asarray(sim.pathfinder.snap_point(np.array([inside[0], 0., inside[1]], np.float32)), np.float32)


def parse(raw: dict[str, Any]) -> ToolCall | None:
    try:
        return ToolCall.parse(raw)
    except Exception:
        return None


def run_task(task: dict, method: str, graph: dict, nodes: dict, compiler: TopologyContextCompiler,
             validator: TopologyToolValidator, cache: dict, cache_file: Path, artifacts: Path) -> dict:
    sim = make_sim(task["scene_id"]); agent = sim.initialize_agent(0)
    try:
        start2 = np.asarray(task["start_anchor_xz"], np.float32)
        raw_start = np.array([start2[0], 0., start2[1]], np.float32)
        start = np.asarray(sim.pathfinder.snap_point(raw_start), np.float32)
        if not np.isfinite(start).all():
            candidates = compiler.adjacent(task["start_node"])
            if candidates:
                start = source_realization(sim, nodes[task["start_node"]], nodes[candidates[0].edge_id])
        if not np.isfinite(start).all():
            return {"task_id": task["task_id"], "method": method, "status": "invalid_start"}
        set_agent(agent, start, 0.)
        memory = AgentMemory(current_node=task["start_node"], goal_nodes={task["goal_node"]},
            route_nodes={task["goal_node"]}, relation=RelationState(task["relation_entity"], task["required_relation"], False))
        history: list[dict] = []; total_path = 0.; vlm_s = 0.; cache_hits = 0; frames: list[dict] = []
        for step in range(MAX_STEPS):
            pos = np.asarray(agent.get_state().position, np.float32)
            rgb, _, _, _ = render(sim, agent, pos, yaw_rotation(0.0))
            image_path = artifacts / "frames" / f"{task['task_id'].replace(':','_')}_{method.replace('+','_')}_{step:02d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(rgb).save(image_path)
            event = "START" if step == 0 else "JUNCTION"
            pool = SelectiveEvidenceRetriever.retrieve(event, str(image_path))
            previous = None
            if history:
                fb = history[-1].get("feedback")
                if isinstance(fb, dict):
                    previous = ExecutionFeedback(Outcome(fb["status"]), fb.get("target"),
                        float(fb["travelled_m"]), float(fb["remaining_m"]), bool(fb["collision"]),
                        fb.get("relation"), bool(fb["relation_complete"]))
            context = compiler.compile(task["instruction"], memory, previous, pool, event)
            candidate_ids = [x["edge_id"] for x in context.transitions]
            if method == "Direct-VLM":
                prompt = {"task": task["instruction"], "current": short(memory.current_node), "candidate_transitions": candidate_ids}
            elif method in {"FullTopo-VLM", "FullTopo+Validator"}:
                prompt = {"task": task["instruction"], "current": memory.current_node, "full_topology": scene_view(graph, task["scene_id"], True), "adjacent_candidates": candidate_ids}
            elif method == "History-Agent":
                prompt = {"task": task["instruction"], "current": memory.current_node, "full_topology": scene_view(graph, task["scene_id"], True), "raw_history": history, "adjacent_candidates": candidate_ids}
            else:
                prompt = {"task": task["instruction"], "current": memory.current_node,
                          "task_relevant_topology": context.as_prompt_object(), "adjacent_candidates": candidate_ids}
            raw, latency, hit = qwen(prompt, [rgb], cache, cache_file)
            vlm_s += latency; cache_hits += int(hit); call = parse(raw)
            valid, reason = (False, "parse_error") if call is None else validator.validate(call, memory)
            use_validator = method in {"FullTopo+Validator", "TopoNav-Harness"}
            if not use_validator and call and call.tool == Tool.NAVIGATE:
                # Baselines remain executable only for real adjacent transitions;
                # illegal choices are recorded, never silently repaired.
                valid = call.target in candidate_ids; reason = "executor_adjacency" if valid else "invalid_raw_action"
            if call is None or not valid:
                feedback = ExecutionFeedback.invalid(None if call is None else call.target, task["required_relation"])
                history.append({"raw": raw, "call": None if call is None else asdict(call), "feedback": asdict(feedback), "validation": reason})
                frames.append({"image": str(image_path.relative_to(ROOT)), "step": step, "raw": raw, "validation": reason, "feedback": asdict(feedback)})
                if method == "TopoNav-Harness" and step + 1 < MAX_STEPS: continue
                break
            if call.tool == Tool.STOP:
                feedback = ExecutionFeedback(Outcome.SUCCESS if memory.relation.complete else Outcome.RELATION_INCOMPLETE,
                    None, 0., 0., False, task["required_relation"], memory.relation.complete)
                history.append({"raw": raw, "call": asdict(call), "feedback": asdict(feedback), "validation": reason}); break
            if call.tool == Tool.OBSERVE:
                feedback = ExecutionFeedback(Outcome.SUCCESS, call.target, 0., 0., False, task["required_relation"], False)
                history.append({"raw": raw, "call": asdict(call), "feedback": asdict(feedback), "validation": reason}); continue
            if call.tool not in {Tool.NAVIGATE}:
                feedback = ExecutionFeedback.invalid(call.target, task["required_relation"])
                history.append({"raw": raw, "call": asdict(call), "feedback": asdict(feedback), "validation": "unsupported_tool"}); break
            edge = next(e for e in compiler.adjacent(memory.current_node) if e.edge_id == call.target)
            target = transition_realization(sim, nodes[memory.current_node], nodes[edge.target], nodes[edge.edge_id])
            feedback, trace, geodesic = path_execute(sim, agent, target)
            feedback = ExecutionFeedback(feedback.status, call.target, feedback.travelled_m, feedback.remaining_m,
                                         feedback.collision, task["required_relation"], feedback.status == Outcome.SUCCESS)
            total_path += feedback.travelled_m
            validator.apply_feedback(memory, call, feedback)
            history.append({"raw": raw, "call": asdict(call), "feedback": asdict(feedback), "validation": reason, "trace_xyz": trace})
            frames.append({"image": str(image_path.relative_to(ROOT)), "step": step, "raw": raw, "call": asdict(call), "feedback": asdict(feedback), "trace_xyz": trace})
            if memory.current_node == task["goal_node"] and memory.relation.complete:
                break
        success = memory.current_node == task["goal_node"] and memory.relation.complete
        goal2 = np.asarray(task["goal_anchor_xz"], np.float32); final = np.asarray(agent.get_state().position, np.float32)
        final_dtg = float(np.linalg.norm(final[[0,2]] - goal2))
        return {"task_id": task["task_id"], "scene_id": task["scene_id"], "task_type": task["task_type"], "method": method,
                "success": success, "path_length_m": total_path, "final_dtg_m": final_dtg, "tool_calls": len(history),
                "actual_start_xyz": start.tolist(),
                "valid_tool_calls": sum(x["validation"] in {"ok", "executor_adjacency"} for x in history),
                "invalid_tool_calls": sum(x["validation"] not in {"ok", "executor_adjacency"} for x in history),
                "relation_complete": memory.relation.complete, "vlm_latency_s": vlm_s, "cache_hits": cache_hits,
                "history": history, "frames": frames}
    finally:
        sim.close()


def summarize(rows: list[dict]) -> list[dict]:
    out=[]
    for method in METHODS:
        g=[r for r in rows if r.get("method")==method and "success" in r]
        if not g: continue
        s=np.asarray([r["success"] for r in g], float); pl=np.asarray([r["path_length_m"] for r in g],float)
        dtg=np.asarray([r["final_dtg_m"] for r in g],float); calls=np.asarray([r["tool_calls"] for r in g],float)
        denom=sum(r["tool_calls"] for r in g)
        out.append({"method":method,"episodes":len(g),"SR":float(s.mean()),"SPL_proxy":float(np.mean(s/(1+pl))),
                    "Final_DTG_m":float(dtg.mean()),"Path_Length_m":float(pl.mean()),"Relation_Completion":float(np.mean([r["relation_complete"] for r in g])),
                    "Tool_Validity_Rate":float(sum(r["valid_tool_calls"] for r in g)/max(1,denom)),
                    "Invalid_Tool_Rate":float(sum(r["invalid_tool_calls"] for r in g)/max(1,denom)),
                    "VLM_Calls":float(calls.mean()),"VLM_Latency_s":float(np.mean([r["vlm_latency_s"] for r in g]))})
    return out


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--methods",nargs="+",default=list(METHODS)); ap.add_argument("--limit",type=int); ap.add_argument("--task-id"); ap.add_argument("--out",default=str(P1)); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); artifacts=out/"artifacts"; cache_file=out/"vlm_cache.jsonl"; cache=cache_load(cache_file)
    manifest=json.loads(MANIFEST.read_text())
    graph_path = GRAPH if GRAPH.exists() else ROOT / "outputs/formal/RelationNav/topology/spatial_semantic_topology.json"
    graph=json.loads(graph_path.read_text()); nodes={n["node_id"]:n for n in graph["nodes"]}
    compiler=TopologyContextCompiler(graph, token_budget=1024); validator=TopologyToolValidator(compiler)
    tasks=manifest["tasks"]
    if args.task_id: tasks=[t for t in tasks if t["task_id"]==args.task_id]
    if args.limit: tasks=tasks[:args.limit]
    rows=[]
    for method in args.methods:
        for i, task in enumerate(tasks,1):
            print(f"[{method}] {i}/{len(tasks)} {task['task_id']}",flush=True)
            rows.append(run_task(task,method,graph,nodes,compiler,validator,cache,cache_file,artifacts))
    (out/"benchmark_episode_results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    summary={"protocol":{"backend":"qwen3-vl:8b-instruct","max_steps":MAX_STEPS,"methods":args.methods,"task_count":len(tasks)},"metrics":summarize(rows)}
    (out/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
