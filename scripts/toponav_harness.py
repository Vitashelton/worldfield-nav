"""Strict semantic-topology navigation-agent protocol.

This module is deliberately independent of Habitat, Nav2, or a particular VLM.
It turns a named semantic topology edge into a validated high-level action and
records typed execution feedback.  A simulator/robot executor is injected by
the caller; it alone owns metric motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class Tool(str, Enum):
    OBSERVE = "OBSERVE"
    NAVIGATE = "NAVIGATE"
    VERIFY = "VERIFY"
    STOP = "STOP"


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    NO_PROGRESS = "NO_PROGRESS"
    COLLISION = "COLLISION"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"
    RELATION_INCOMPLETE = "RELATION_INCOMPLETE"


@dataclass(frozen=True)
class ToolCall:
    tool: Tool
    target: str | None = None
    relation: str | None = None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "ToolCall":
        tool = Tool(str(raw.get("tool", "")).upper())
        target = raw.get("target")
        relation = raw.get("relation")
        if tool in {Tool.NAVIGATE, Tool.OBSERVE} and not isinstance(target, str):
            raise ValueError(f"{tool.value} requires a string target")
        return cls(tool=tool, target=target, relation=relation)


@dataclass
class EdgeState:
    edge_id: str
    source: str
    target: str
    semantic_type: str = "transition"
    distance_m: float | None = None
    status: str = "unexplored"
    failure_count: int = 0
    last_outcome: str = "UNKNOWN"
    evidence_refs: list[str] = field(default_factory=list)

    @property
    def retry_allowed(self) -> bool:
        return self.failure_count < 2


@dataclass
class RelationState:
    entity: str | None = None
    relation: str | None = None
    complete: bool = False

    def update(self, feedback: "ExecutionFeedback") -> None:
        self.complete = bool(feedback.relation_complete)


@dataclass(frozen=True)
class ExecutionFeedback:
    status: Outcome
    target: str | None
    travelled_m: float
    remaining_m: float
    collision: bool
    relation: str | None
    relation_complete: bool

    @classmethod
    def invalid(cls, target: str | None, relation: str | None = None) -> "ExecutionFeedback":
        return cls(Outcome.INVALID, target, 0.0, float("inf"), False, relation, False)


@dataclass
class AgentMemory:
    current_node: str
    goal_nodes: set[str]
    route_nodes: set[str] = field(default_factory=set)
    failed_edges: set[str] = field(default_factory=set)
    relation: RelationState = field(default_factory=RelationState)
    recent_nodes: list[str] = field(default_factory=list)


@dataclass
class CompiledContext:
    task: str
    current_node: str
    transitions: list[dict[str, Any]]
    selected_images: list[str]
    typed_feedback: dict[str, Any] | None
    relation_state: dict[str, Any]
    estimated_tokens: int

    def as_prompt_object(self) -> dict[str, Any]:
        return asdict(self)


class TopologyContextCompiler:
    """Deterministic relevance-budget compiler for local VLM context."""

    def __init__(self, graph: dict[str, Any], token_budget: int = 1024, gamma: float = 0.7):
        self.graph = graph
        self.token_budget = token_budget
        self.gamma = gamma
        self.nodes = {n["node_id"]: n for n in graph.get("nodes", [])}
        self.edges = [
            self._edge(e)
            for e in graph.get("edges", [])
            if e.get("edge_type", "SPATIAL_ADJACENCY") == "SPATIAL_ADJACENCY"
        ]

    @staticmethod
    def _edge(edge: dict[str, Any]) -> EdgeState:
        return EdgeState(
            edge_id=str(edge.get("via", f'{edge["source"]}->{edge["target"]}')),
            source=str(edge["source"]),
            target=str(edge["target"]),
            semantic_type=str(edge.get("semantic_type", edge.get("relation", "transition"))),
            distance_m=edge.get("distance_m"),
            evidence_refs=list(edge.get("evidence_refs", [])),
        )

    def adjacent(self, node_id: str) -> list[EdgeState]:
        return [e for e in self.edges if e.source == node_id]

    def compile(
        self,
        task: str,
        memory: AgentMemory,
        feedback: ExecutionFeedback | None = None,
        image_pool: dict[str, list[str]] | None = None,
        event: str = "START",
    ) -> CompiledContext:
        pool = image_pool or {}
        ranked: list[tuple[float, EdgeState]] = []
        for edge in self.adjacent(memory.current_node):
            task_gain = 1.0 if edge.target in memory.goal_nodes else 0.0
            route_gain = 0.7 if edge.target in memory.route_nodes else 0.0
            failure_gain = 1.0 if edge.edge_id in memory.failed_edges else 0.0
            recent_penalty = 0.35 if edge.target in memory.recent_nodes else 0.0
            score = task_gain + route_gain + failure_gain - recent_penalty
            if edge.edge_id in memory.failed_edges:
                score += 1.0
            ranked.append((score, edge))
        ranked.sort(key=lambda x: (-x[0], x[1].edge_id))

        transitions: list[dict[str, Any]] = []
        budget = max(128, self.token_budget - 220)
        spent = 0
        for score, edge in ranked:
            item = {
                "edge_id": edge.edge_id,
                "leads_to": edge.target,
                "type": edge.semantic_type,
                "distance_m": edge.distance_m,
                "status": "previously_failed" if edge.edge_id in memory.failed_edges else edge.status,
                "retry_allowed": edge.retry_allowed,
                "relevance": round(score, 3),
            }
            cost = max(45, len(str(item)) // 3)
            if transitions and spent + cost > budget:
                continue
            transitions.append(item)
            spent += cost

        images = list(pool.get("current", []))[:1]
        if event in {"START", "JUNCTION", "AMBIGUITY", "FAILURE"}:
            images.extend(pool.get(event.lower(), [])[:1])
        return CompiledContext(
            task=task,
            current_node=memory.current_node,
            transitions=transitions,
            selected_images=images[:2],
            typed_feedback=asdict(feedback) if feedback else None,
            relation_state=asdict(memory.relation),
            estimated_tokens=220 + spent + 180 * len(images[:2]),
        )


class TopologyToolValidator:
    """Pre-execution validator; it never creates a metric goal."""

    def __init__(self, compiler: TopologyContextCompiler):
        self.compiler = compiler

    def validate(self, call: ToolCall, memory: AgentMemory) -> tuple[bool, str]:
        if call.tool == Tool.STOP:
            return (memory.current_node in memory.goal_nodes and memory.relation.complete,
                    "goal_not_verified" if not (memory.current_node in memory.goal_nodes and memory.relation.complete) else "ok")
        if call.tool == Tool.VERIFY:
            return (bool(call.relation), "missing_relation" if not call.relation else "ok")
        if call.tool == Tool.OBSERVE:
            return (call.target in self.compiler.nodes, "unknown_node" if call.target not in self.compiler.nodes else "ok")
        if call.tool == Tool.NAVIGATE:
            matches = [e for e in self.compiler.adjacent(memory.current_node) if e.edge_id == call.target]
            if not matches:
                return False, "nonexistent_or_nonadjacent_edge"
            if not matches[0].retry_allowed or call.target in memory.failed_edges:
                return False, "failed_realization_blacklisted"
            return True, "ok"
        return False, "unknown_tool"

    def apply_feedback(self, memory: AgentMemory, call: ToolCall, feedback: ExecutionFeedback) -> None:
        memory.relation.update(feedback)
        if call.tool == Tool.NAVIGATE and call.target:
            if feedback.status == Outcome.SUCCESS:
                edge = next(e for e in self.compiler.adjacent(memory.current_node) if e.edge_id == call.target)
                memory.recent_nodes.append(memory.current_node)
                memory.recent_nodes = memory.recent_nodes[-4:]
                memory.current_node = edge.target
            elif feedback.status in {Outcome.NO_PROGRESS, Outcome.COLLISION, Outcome.BLOCKED, Outcome.TIMEOUT}:
                memory.failed_edges.add(call.target)


class SelectiveEvidenceRetriever:
    """Event-gated references only; image bytes remain outside the protocol."""

    EVENTS = {"START", "JUNCTION", "AMBIGUITY", "FAILURE"}

    @classmethod
    def retrieve(cls, event: str, current: str, node_keyframe: str | None = None,
                 failure_keyframe: str | None = None) -> dict[str, list[str]]:
        out = {"current": [current]}
        if event in cls.EVENTS:
            ref = failure_keyframe if event == "FAILURE" else node_keyframe
            if ref:
                out[event.lower()] = [ref]
        return out
