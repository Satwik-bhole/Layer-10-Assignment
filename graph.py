"""
The actual knowledge graph — built on top of NetworkX's MultiDiGraph.
Entities become nodes, claims become directed edges, and all the evidence
lives as edge attributes. Supports having multiple different relationships
between the same pair of entities, which happens a lot in real email data.
"""

import json
import os
from collections import Counter
from datetime import datetime

import networkx as nx

from schema import Claim, Entity, MemoryStore


class MemoryGraph:
    """The main knowledge graph. Wraps a NetworkX MultiDiGraph with ingestion and query methods."""

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

    # -- Adding data to the graph --

    def ingest_entity(self, entity: Entity):
        """Add an entity as a graph node. Safe to call multiple times — duplicates are skipped."""
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
        """Add a claim as a directed edge between two entities. Skips duplicates and orphans."""
        if claim.id in self._ingested_claim_ids:
            self._metrics["duplicate_claims_skipped"] += 1
            return

        # both the subject and object entities need to exist in the graph
        if claim.subject_id not in self._ingested_entity_ids:
            self._metrics["orphan_claims_skipped"] += 1
            return
        if claim.object_id not in self._ingested_entity_ids:
            self._metrics["orphan_claims_skipped"] += 1
            return

        # turn evidence into plain dicts for storage on the edge
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
        """Load a full MemoryStore into the graph in one go."""
        print(f"Ingesting {len(store.entities)} entities and {len(store.claims)} claims...")

        for entity in store.entities.values():
            self.ingest_entity(entity)

        for claim in store.claims.values():
            self.ingest_claim(claim)

        print(f"Graph now has {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges.")

    # -- Querying --

    def get_node(self, entity_id: str) -> dict:
        """Look up a node's attributes by ID."""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return {}

    def get_neighbors(self, entity_id: str, current_only: bool = True) -> list[dict]:
        """Get all the claims (edges) connected to a given entity, both incoming and outgoing."""
        results = []
        # outgoing
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
        # incoming
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
        """Search for entities whose name or aliases contain the query string."""
        query_lower = query.lower()
        results = []
        for nid, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            if query_lower in name or any(query_lower in a for a in aliases):
                results.append(nid)
        return results

    def get_subgraph(self, entity_ids: list[str], depth: int = 1) -> nx.MultiDiGraph:
        """Pull out a neighborhood around the given entities, going N hops deep."""
        nodes = set(entity_ids)
        for _ in range(depth):
            new_nodes = set()
            for nid in nodes:
                new_nodes.update(self.graph.successors(nid))
                new_nodes.update(self.graph.predecessors(nid))
            nodes.update(new_nodes)

        return self.graph.subgraph(nodes).copy()

    # -- Quality metrics --

    def get_metrics(self) -> dict:
        """Compute a bunch of stats about the graph for monitoring quality."""
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
        """Print out a nice summary of the graph's quality metrics."""
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

    # -- Save and load --

    def save(self, path: str):
        """Write the graph out to a JSON file."""
        data = nx.node_link_data(self.graph)
        # stash our tracking data alongside the graph
        data["_metrics"] = self._metrics
        data["_ingested_entity_ids"] = list(self._ingested_entity_ids)
        data["_ingested_claim_ids"] = list(self._ingested_claim_ids)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MemoryGraph":
        """Read a graph back from a saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mg = cls()
        mg._metrics = data.pop("_metrics", mg._metrics)
        mg._ingested_entity_ids = set(data.pop("_ingested_entity_ids", []))
        mg._ingested_claim_ids = set(data.pop("_ingested_claim_ids", []))
        mg.graph = nx.node_link_graph(data)
        return mg


def build_graph(store_path: str, output_path: str) -> MemoryGraph:
    """Create the graph from a deduplicated MemoryStore and save it."""
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
