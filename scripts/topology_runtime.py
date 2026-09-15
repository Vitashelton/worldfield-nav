"""Execution-aware routing over the heterogeneous semantic topology."""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from typing import Any


@dataclass
class EdgeExecutionState:
    success_probability: float = 0.8
    blocked_confidence: float = 0.0
    failure_count: int = 0
    last_event: str = "unknown"

    def update(self, event: str) -> None:
        self.last_event = event
        if event in {"collision", "blocked", "no_progress", "timeout"}:
            self.failure_count += 1
            self.blocked_confidence = min(1.0, 0.45 + 0.15 * self.failure_count)
            self.success_probability = max(0.0, 1.0 - self.blocked_confidence)
        elif event == "success":
            self.success_probability = min(1.0, self.success_probability + 0.08)
            self.blocked_confidence = max(0.0, self.blocked_confidence - 0.08)

    @property
    def available(self) -> bool:
        return self.blocked_confidence < 0.7


@dataclass
class ExecutionAwareTopology:
    graph: dict[str, Any]
    edge_state: dict[str, EdgeExecutionState] = field(default_factory=dict)

    def _key(self, edge: dict[str, Any]) -> str:
        return str(edge.get("via", f"{edge['source']}->{edge['target']}"))

    def observe(self, edge_id: str, event: str) -> None:
        state = self.edge_state.setdefault(edge_id, EdgeExecutionState())
        state.update(event)

    def route(self, source: str, target: str) -> list[dict[str, Any]] | None:
        """Dijkstra on room/area nodes; each hop retains the portal realization."""
        adjacency: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
        for edge in self.graph.get("edges", []):
            if edge.get("edge_type") != "SPATIAL_ADJACENCY": continue
            key = self._key(edge); state = self.edge_state.get(key, EdgeExecutionState())
            if not state.available: continue
            adjacency.setdefault(edge["source"], []).append((1.0 + 0.1 * state.failure_count, edge["target"], edge))
        q=[(0.0, 0, source, [])]; seen={}; serial=1
        while q:
            cost, _, node, path = heapq.heappop(q)
            if node in seen and seen[node] <= cost: continue
            seen[node] = cost
            if node == target: return path
            for w, nxt, edge in adjacency.get(node, []):
                heapq.heappush(q, (cost+w, serial, nxt, path+[edge])); serial += 1
        return None
