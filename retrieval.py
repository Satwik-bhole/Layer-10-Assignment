"""
Retrieval engine — takes a natural language question, finds relevant entities
and claims in the graph, and packages everything up into a grounded context pack.
Answers always come with the source emails to back them up.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from graph import MemoryGraph
from schema import MemoryStore

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ContextItem:
    """One claim with its evidence, ready to be shown to the user or fed to an LLM."""
    claim_id: str
    subject: str
    relation: str
    object_str: str
    confidence: float
    is_current: bool
    valid_from: str
    valid_until: str
    evidence_snippets: list[dict] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class ContextPack:
    """A ranked bundle of claims and evidence that answers a query."""
    query: str
    matched_entities: list[dict] = field(default_factory=list)
    items: list[ContextItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "matched_entities": self.matched_entities,
            "items": [
                {
                    "claim_id": it.claim_id,
                    "subject": it.subject,
                    "relation": it.relation,
                    "object": it.object_str,
                    "confidence": it.confidence,
                    "is_current": it.is_current,
                    "valid_from": it.valid_from,
                    "valid_until": it.valid_until,
                    "evidence": it.evidence_snippets,
                    "relevance_score": round(it.relevance_score, 3),
                }
                for it in self.items
            ],
        }

    def format_for_llm(self) -> str:
        """Turn the context pack into plain text that an LLM can reason over."""
        lines = [f"CONTEXT FOR QUERY: {self.query}\n"]

        if self.matched_entities:
            lines.append("MATCHED ENTITIES:")
            for ent in self.matched_entities:
                aliases = ", ".join(ent.get("aliases", []))
                alias_str = f" (aliases: {aliases})" if aliases else ""
                lines.append(f"  - {ent['name']} [{ent['entity_type']}]{alias_str}")
            lines.append("")

        lines.append("RELEVANT CLAIMS (ranked by relevance):")
        for i, item in enumerate(self.items, 1):
            status = "CURRENT" if item.is_current else f"HISTORICAL (until {item.valid_until})"
            lines.append(
                f"\n  [{i}] {item.subject} --{item.relation}--> {item.object_str}"
                f"\n      Status: {status} | Confidence: {item.confidence}"
                f"\n      Valid from: {item.valid_from}"
            )
            for ev in item.evidence_snippets:
                lines.append(
                    f"      EVIDENCE [source: {ev.get('source_id', 'unknown')}]:"
                    f"\n        \"{ev.get('exact_quote', '')}\""
                    f"\n        -- {ev.get('author', 'unknown')} at {ev.get('timestamp', 'unknown')}"
                )

        return "\n".join(lines)


class Retriever:
    """Handles the whole flow from question to grounded context pack."""

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._node_embeddings = {}
        self._node_ids = []
        self._embedding_matrix = None
        self._build_index()

    def _build_index(self):
        """Pre-compute embeddings for every entity name and alias so search is fast."""
        print("Building retrieval index...")
        names = []
        ids = []
        for nid, data in self.graph.graph.nodes(data=True):
            name = data.get("name", "")
            if name:
                names.append(name)
                ids.append(nid)
                # index aliases too so we can match on those
                for alias in data.get("aliases", []):
                    names.append(alias)
                    ids.append(nid)

        if names:
            self._embedding_matrix = self.model.encode(
                names, normalize_embeddings=True, show_progress_bar=False
            )
            self._node_ids = ids
        print(f"  Indexed {len(names)} names/aliases across {self.graph.graph.number_of_nodes()} entities.")

    def _find_matching_entities(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Find entities that match the query using both keyword and semantic search.
        Returns (entity_id, score) pairs ranked by relevance.
        """
        if self._embedding_matrix is None or len(self._node_ids) == 0:
            return []

        # first, quick keyword matching against names and aliases
        keyword_matches = set()
        for nid, data in self.graph.graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            query_lower = query.lower()
            for word in query_lower.split():
                if len(word) >= 3 and (word in name or any(word in a for a in aliases)):
                    keyword_matches.add(nid)

        # then, semantic similarity using embeddings for fuzzier matching
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self._embedding_matrix, query_emb.T).flatten()

        # blend the two approaches together
        entity_scores = {}
        for idx in np.argsort(similarities)[::-1][:top_k * 3]:
            nid = self._node_ids[idx]
            score = float(similarities[idx])
            if nid not in entity_scores or score > entity_scores[nid]:
                entity_scores[nid] = score

        # give keyword matches a nice boost since exact matches are usually what people want
        for nid in keyword_matches:
            entity_scores[nid] = entity_scores.get(nid, 0.3) + 0.3

        # take the best unique entities
        ranked = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve(self, query: str, top_k: int = 10,
                 include_historical: bool = True) -> ContextPack:
        """The main retrieval method — give it a question, get back a grounded context pack."""
        pack = ContextPack(query=query)

        # find which entities are relevant to the question
        entity_matches = self._find_matching_entities(query, top_k=5)
        if not entity_matches:
            return pack

        # record what we matched for transparency
        for nid, score in entity_matches:
            node_data = self.graph.get_node(nid)
            if node_data:
                pack.matched_entities.append({
                    "id": nid,
                    "name": node_data.get("name", "unknown"),
                    "entity_type": node_data.get("entity_type", "unknown"),
                    "aliases": node_data.get("aliases", []),
                    "match_score": round(score, 3),
                })

        # pull in all claims connected to matching entities
        seen_claims = set()
        raw_items = []

        for nid, entity_score in entity_matches:
            neighbors = self.graph.get_neighbors(nid, current_only=False)
            for edge in neighbors:
                cid = edge.get("claim_id", "")
                if cid in seen_claims:
                    continue
                seen_claims.add(cid)

                is_current = edge.get("is_current", True)
                if not include_historical and not is_current:
                    continue

                # look up human-readable names for the entities in this claim
                subj_data = self.graph.get_node(edge["subject_id"])
                obj_data = self.graph.get_node(edge["object_id"])
                subject_name = subj_data.get("name", edge["subject_id"]) if subj_data else edge["subject_id"]
                object_name = obj_data.get("name", edge["object_id"]) if obj_data else edge["object_id"]

                # score = how well the entity matched * claim confidence * freshness bonus
                confidence = edge.get("confidence", 0.5)
                recency_boost = 1.2 if is_current else 0.8
                relevance = entity_score * confidence * recency_boost

                item = ContextItem(
                    claim_id=cid,
                    subject=subject_name,
                    relation=edge.get("relation", "related_to"),
                    object_str=object_name,
                    confidence=confidence,
                    is_current=is_current,
                    valid_from=edge.get("valid_from", ""),
                    valid_until=edge.get("valid_until", "") or "",
                    evidence_snippets=edge.get("evidence", []),
                    relevance_score=relevance,
                )
                raw_items.append(item)

        # rank everything and keep the best ones
        raw_items.sort(key=lambda x: x.relevance_score, reverse=True)
        pack.items = raw_items[:top_k]

        return pack

    def answer_question(self, query: str, top_k: int = 10) -> tuple[str, ContextPack]:
        """Answer a question by retrieving context from the graph and having Ollama generate a response."""
        import ollama as ollama_client

        pack = self.retrieve(query, top_k=top_k)

        if not pack.items:
            return ("I couldn't find any relevant information in the memory graph "
                    "for that query.", pack)

        context_text = pack.format_for_llm()

        prompt = f"""Answer the user's question using ONLY the provided context.
Cite the source_id for every fact you state.
If the context doesn't contain enough information, say so.

{context_text}

QUESTION: {query}

ANSWER (cite sources):"""

        try:
            response = ollama_client.chat(
                model="llama3.1",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 1024},
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = f"Error generating answer: {e}\n\nRaw context:\n{context_text}"

        return answer, pack


def generate_example_context_packs(graph_path: str, output_path: str):
    """Run a few sample queries and save the results so we can show them off."""
    mg = MemoryGraph.load(graph_path)
    retriever = Retriever(mg)

    sample_questions = [
        "What topics did Vince Kaminski discuss?",
        "Who are the most active email correspondents?",
        "What decisions were made about energy trading?",
        "What projects or deals were discussed?",
        "Who communicated with Kenneth Lay?",
    ]

    packs = []
    for q in sample_questions:
        print(f"\nQuery: {q}")
        pack = retriever.retrieve(q)
        print(f"  Found {len(pack.items)} relevant claims from {len(pack.matched_entities)} entities")
        packs.append(pack.to_dict())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(packs)} context packs to {output_path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    graph_path = os.path.join(data_dir, "memory_graph.json")
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "example_context_packs.json")
    generate_example_context_packs(graph_path, output_path)
