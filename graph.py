"""
Memory graph built on NetworkX MultiDiGraph.

Represents entities as nodes and claims as directed edges,
with evidence stored as edge attributes.

Features:
- MultiDiGraph: multiple directed edges between same nodes (different relations)
- Event time vs validity time tracked on edges
- Incremental ingestion with idempotency (skip existing IDs)
- Observability: extraction quality metrics
- Serialization to/from JSON
"""

import json
import os
from collections import Counter
from datetime import datetime

import networkx as nx

from schema import Claim, Entity, MemoryStore


class MemoryGraph:
    """In-memory knowledge graph backed by NetworkX."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._ingested_entity_ids = set()
        self._ingested_claim_ids = set()
        self._metrics = {
            "total_entities_ingested": 0,
            "total_claims_ingested": 0,
            "duplicate_entities_skipped": 0,
            "duplicate_claims_skipped": 0,
            "orphan_claims_skipped": 0,
        }

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_entity(self, entity: Entity):
        """Add an entity as a node. Idempotent: skips if already ingested."""
        if entity.id in self._ingested_entity_ids:
            self._metrics["duplicate_entities_skipped"] += 1
            return

        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            aliases=entity.aliases,
            first_seen=entity.first_seen,
        )
        self._ingested_entity_ids.add(entity.id)
        self._metrics["total_entities_ingested"] += 1

    def ingest_claim(self, claim: Claim):
        """Add a claim as a directed edge. Idempotent."""
        if claim.id in self._ingested_claim_ids:
            self._metrics["duplicate_claims_skipped"] += 1
            return

        # Check that both endpoints exist
        if claim.subject_id not in self._ingested_entity_ids:
            self._metrics["orphan_claims_skipped"] += 1
            return
        if claim.object_id not in self._ingested_entity_ids:
            self._metrics["orphan_claims_skipped"] += 1
            return

        # Serialize evidence for storage
        evidence_data = [ev.model_dump() for ev in claim.evidence]

        self.graph.add_edge(
            claim.subject_id,
            claim.object_id,
            key=claim.id,
            relation=claim.relation.value,
            evidence=evidence_data,
            confidence=claim.confidence,
            valid_from=claim.valid_from,
            valid_until=claim.valid_until,
            is_current=claim.is_current,
            extraction_version=claim.extraction_version,
        )
        self._ingested_claim_ids.add(claim.id)
        self._metrics["total_claims_ingested"] += 1

    def ingest_store(self, store: MemoryStore):
        """Ingest an entire MemoryStore into the graph."""
        print(f"Ingesting {len(store.entities)} entities and {len(store.claims)} claims...")

        for entity in store.entities.values():
            self.ingest_entity(entity)

        for claim in store.claims.values():
            self.ingest_claim(claim)

        print(f"Graph now has {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges.")

    # ── Querying ───────────────────────────────────────────────────────────

    def get_node(self, entity_id: str) -> dict:
        """Get node attributes."""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return {}

    def get_neighbors(self, entity_id: str, current_only: bool = True) -> list[dict]:
        """Get all claims (edges) connected to an entity."""
        results = []
        # Outgoing edges
        for _, target, key, data in self.graph.out_edges(entity_id, data=True, keys=True):
            if current_only and not data.get("is_current", True):
                continue
            results.append({
                "claim_id": key,
                "subject_id": entity_id,
                "object_id": target,
                "direction": "outgoing",
                **data,
            })
        # Incoming edges
        for source, _, key, data in self.graph.in_edges(entity_id, data=True, keys=True):
            if current_only and not data.get("is_current", True):
                continue
            results.append({
                "claim_id": key,
                "subject_id": source,
                "object_id": entity_id,
                "direction": "incoming",
                **data,
            })
        return results

    def find_entities_by_name(self, query: str) -> list[str]:
        """Find entity IDs whose name or aliases match a query (case-insensitive)."""
        query_lower = query.lower()
        results = []
        for nid, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            if query_lower in name or any(query_lower in a for a in aliases):
                results.append(nid)
        return results

    def get_subgraph(self, entity_ids: list[str], depth: int = 1) -> nx.MultiDiGraph:
        """Get a subgraph around the given entities up to a given depth."""
        nodes = set(entity_ids)
        for _ in range(depth):
            new_nodes = set()
            for nid in nodes:
                new_nodes.update(self.graph.successors(nid))
                new_nodes.update(self.graph.predecessors(nid))
            nodes.update(new_nodes)

        return self.graph.subgraph(nodes).copy()

    # ── Observability ──────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return extraction and graph quality metrics."""
        entity_types = Counter()
        relation_types = Counter()
        evidence_counts = []
        confidence_values = []
        current_claims = 0
        historical_claims = 0

        for nid, data in self.graph.nodes(data=True):
            entity_types[data.get("entity_type", "unknown")] += 1

        for u, v, key, data in self.graph.edges(data=True, keys=True):
            relation_types[data.get("relation", "unknown")] += 1
            evidence_counts.append(len(data.get("evidence", [])))
            confidence_values.append(data.get("confidence", 0))
            if data.get("is_current", True):
                current_claims += 1
            else:
                historical_claims += 1

        return {
            "graph_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "connected_components": nx.number_weakly_connected_components(self.graph),
            },
            "entity_type_distribution": dict(entity_types),
            "relation_type_distribution": dict(relation_types),
            "evidence_stats": {
                "total_evidence_pointers": sum(evidence_counts),
                "avg_evidence_per_claim": (
                    sum(evidence_counts) / len(evidence_counts)
                    if evidence_counts else 0
                ),
                "claims_with_no_evidence": sum(1 for c in evidence_counts if c == 0),
            },
            "confidence_stats": {
                "mean": (sum(confidence_values) / len(confidence_values)
                         if confidence_values else 0),
                "min": min(confidence_values) if confidence_values else 0,
                "max": max(confidence_values) if confidence_values else 0,
            },
            "temporal_stats": {
                "current_claims": current_claims,
                "historical_claims": historical_claims,
            },
            "ingestion_metrics": self._metrics,
        }

    def print_metrics(self):
        """Pretty-print quality metrics."""
        metrics = self.get_metrics()
        print("\n" + "=" * 60)
        print("MEMORY GRAPH OBSERVABILITY REPORT")
        print("=" * 60)
        for section, data in metrics.items():
            print(f"\n  {section}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {data}")
        print("=" * 60)

    # ── Serialization ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialize graph to JSON."""
        data = nx.node_link_data(self.graph)
        # Include metrics
        data["_metrics"] = self._metrics
        data["_ingested_entity_ids"] = list(self._ingested_entity_ids)
        data["_ingested_claim_ids"] = list(self._ingested_claim_ids)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MemoryGraph":
        """Load graph from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mg = cls()
        mg._metrics = data.pop("_metrics", mg._metrics)
        mg._ingested_entity_ids = set(data.pop("_ingested_entity_ids", []))
        mg._ingested_claim_ids = set(data.pop("_ingested_claim_ids", []))
        mg.graph = nx.node_link_graph(data)
        return mg


def build_graph(store_path: str, output_path: str) -> MemoryGraph:
    """Build graph from deduped MemoryStore."""
    store = MemoryStore.deserialize(store_path)

    mg = MemoryGraph()
    mg.ingest_store(store)
    mg.print_metrics()
    mg.save(output_path)

    return mg


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    store_path = os.path.join(data_dir, "deduped_store.json")
    graph_path = os.path.join(data_dir, "memory_graph.json")
    build_graph(store_path, graph_path)
