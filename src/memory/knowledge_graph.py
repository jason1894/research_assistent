"""In-memory knowledge graph with JSON persistence."""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Stores research concepts and their relationships as a directed graph.

    The graph is kept in memory and optionally persisted to a JSON file so
    no external graph database is required.

    Args:
        graph_path: Path to the JSON file used for persistence.  Pass
            ``None`` to disable persistence.
    """

    def __init__(self, graph_path: Optional[str] = "./data/knowledge_graph.json") -> None:
        self.graph_path = Path(graph_path) if graph_path else None
        # nodes: concept_name -> metadata dict
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # edges: list of {source, target, relation_type, created_at}
        self._edges: List[Dict[str, str]] = []
        # adjacency index: concept_name -> list of neighbour names
        self._adjacency: Dict[str, List[str]] = {}

        if self.graph_path and self.graph_path.exists():
            self.import_graph(str(self.graph_path))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        domain: str = "",
        description: str = "",
        related_papers: Optional[List[str]] = None,
    ) -> None:
        """Add or update a concept node.

        Args:
            name: Unique name for the concept.
            domain: Research domain (e.g. ``"machine_learning"``).
            description: Brief description of the concept.
            related_papers: List of paper titles or file paths associated
                with this concept.
        """
        self._nodes[name] = {
            "name": name,
            "domain": domain,
            "description": description,
            "related_papers": related_papers or [],
            "created_at": datetime.utcnow().isoformat(),
        }
        if name not in self._adjacency:
            self._adjacency[name] = []
        logger.debug("Added concept: %s", name)
        self._auto_save()

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str = "related_to",
    ) -> None:
        """Create a directed relationship between two concepts.

        Both concepts are auto-created if they do not already exist.

        Args:
            source: Name of the source concept.
            target: Name of the target concept.
            relation_type: Type of relationship (e.g. ``"uses"``,
                ``"extends"``, ``"contradicts"``).
        """
        for node in (source, target):
            if node not in self._nodes:
                self.add_concept(node)

        edge = {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._edges.append(edge)

        if target not in self._adjacency[source]:
            self._adjacency[source].append(target)
        if source not in self._adjacency.get(target, []):
            self._adjacency.setdefault(target, []).append(source)

        logger.debug("Added relationship: %s -[%s]-> %s", source, relation_type, target)
        self._auto_save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_related_concepts(
        self, name: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        """BFS traversal to find concepts reachable within ``depth`` hops.

        Args:
            name: Starting concept name.
            depth: Maximum hop distance.

        Returns:
            List of concept metadata dicts (excluding the starting node).
        """
        if name not in self._nodes:
            return []

        visited: Set[str] = {name}
        queue: deque = deque([(name, 0)])
        related: List[Dict[str, Any]] = []

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for neighbour in self._adjacency.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    related.append(
                        {**self._nodes[neighbour], "distance": current_depth + 1}
                    )
                    queue.append((neighbour, current_depth + 1))

        return related

    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Case-insensitive substring search over concept names and descriptions.

        Args:
            query: Search string.

        Returns:
            List of matching concept metadata dicts.
        """
        query_lower = query.lower()
        results = [
            node
            for node in self._nodes.values()
            if query_lower in node["name"].lower()
            or query_lower in node.get("description", "").lower()
        ]
        return results

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a single concept.

        Args:
            name: Concept name.

        Returns:
            Metadata dict, or ``None`` if the concept is not found.
        """
        return self._nodes.get(name)

    def get_edges_for(self, name: str) -> List[Dict[str, str]]:
        """Return all edges involving a given concept.

        Args:
            name: Concept name.

        Returns:
            List of edge dicts where source or target equals ``name``.
        """
        return [e for e in self._edges if e["source"] == name or e["target"] == name]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_graph(self) -> Dict[str, Any]:
        """Return the full graph as a serialisable dict.

        Returns:
            Dict with ``nodes`` and ``edges`` keys.
        """
        return {
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
            "exported_at": datetime.utcnow().isoformat(),
        }

    def save_graph(self, path: Optional[str] = None) -> str:
        """Persist the graph to disk as JSON.

        Args:
            path: Output file path.  Falls back to ``self.graph_path``.

        Returns:
            Absolute path of the saved file.

        Raises:
            ValueError: If no path is available.
        """
        target = Path(path) if path else self.graph_path
        if target is None:
            raise ValueError("No graph_path configured and no path argument given.")

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.export_graph(), fh, ensure_ascii=False, indent=2)

        logger.info("Knowledge graph saved to: %s", target)
        return str(target.resolve())

    def import_graph(self, path: str) -> None:
        """Load a graph from a JSON file.

        Existing nodes and edges are preserved; duplicates are skipped.

        Args:
            path: Path to a JSON file previously written by :meth:`save_graph`.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for node in data.get("nodes", []):
            name = node.get("name", "")
            if name and name not in self._nodes:
                self._nodes[name] = node
                self._adjacency.setdefault(name, [])

        for edge in data.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src and tgt and edge not in self._edges:
                self._edges.append(edge)
                if tgt not in self._adjacency.get(src, []):
                    self._adjacency.setdefault(src, []).append(tgt)
                if src not in self._adjacency.get(tgt, []):
                    self._adjacency.setdefault(tgt, []).append(src)

        logger.info(
            "Imported graph: %d nodes, %d edges from '%s'.",
            len(self._nodes),
            len(self._edges),
            path,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Silently persist the graph if a path is configured."""
        if self.graph_path:
            try:
                self.save_graph()
            except Exception as exc:
                logger.warning("Auto-save failed: %s", exc)
