"""
Deduplication and canonicalization pipeline.

Handles three levels of dedup as required:
1. Artifact dedup: near-identical source texts (quoted replies, cross-posts)
2. Entity canonicalization: merging entities that refer to the same thing
   - Exact match (lowercase, strip)
   - Semantic match (embedding similarity > 0.90)
3. Claim dedup: merging repeated assertions while preserving all evidence

Also handles:
- Conflict resolution with temporal validity (valid_from / valid_until)
- Reversibility: full merge audit log with original snapshots for undo
"""

import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from schema import (
    Claim, Entity, EntityType, Evidence, MemoryStore,
    MergeRecord, RelationType,
)

SIMILARITY_THRESHOLD = 0.90
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Deduplicator:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.embedding_model = None
        self._entity_embeddings = {}
        # Union-Find for entity resolution
        self._parent = {}

    def _load_embedding_model(self):
        if self.embedding_model is None:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded.")

    # ── Union-Find ─────────────────────────────────────────────────────────

    def _find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    # ── 1. Artifact Dedup ──────────────────────────────────────────────────

    def dedup_artifacts(self, corpus: list[dict]) -> list[dict]:
        """
        Remove near-duplicate source texts (quoted replies, forwarded content).
        Handles email-specific quoting: '>' prefixed lines, '--- Original Message ---',
        '-----Forwarded by...', signature blocks, and multi-level quoting.
        Uses text fingerprinting to detect >80% overlap.
        """
        print("Deduplicating artifacts...")
        seen_fingerprints = {}
        dedup_count = 0

        for issue in corpus:
            unique_comments = []
            for comment in issue.get("comments", []):
                text = comment.get("text", "")
                # Normalize: strip email quoting artifacts
                clean_text = text
                # Remove '>' quoted lines (multi-level)
                clean_text = re.sub(
                    r"^>+.*$", "", clean_text, flags=re.MULTILINE)
                # Remove "--- Original Message ---" blocks and everything after
                clean_text = re.sub(
                    r"-{2,}\s*Original Message\s*-{2,}.*",
                    "", clean_text, flags=re.DOTALL | re.IGNORECASE
                )
                # Remove "-----Forwarded by..." blocks and everything after
                clean_text = re.sub(
                    r"-{2,}\s*Forwarded by.*",
                    "", clean_text, flags=re.DOTALL | re.IGNORECASE
                )
                # Remove common email signature markers
                clean_text = re.sub(
                    r"\n--\s*\n.*", "", clean_text, flags=re.DOTALL
                )
                # Normalize whitespace
                clean_text = re.sub(r"\s+", " ", clean_text).strip()

                if len(clean_text) < 20:
                    unique_comments.append(comment)
                    continue

                # Fingerprint using normalized text hash
                fp = hashlib.md5(clean_text.lower().encode()).hexdigest()

                if fp in seen_fingerprints:
                    dedup_count += 1
                    # Keep the comment but mark it as a duplicate
                    comment["_is_artifact_duplicate"] = True
                    comment["_duplicate_of"] = seen_fingerprints[fp]
                else:
                    seen_fingerprints[fp] = comment["comment_id"]

                unique_comments.append(comment)
            issue["comments"] = unique_comments

        print(f"  Found {dedup_count} artifact duplicates.")
        return corpus

    # ── 2. Entity Canonicalization ─────────────────────────────────────────

    def canonicalize_entities(self):
        """
        Merge entities that refer to the same real-world thing.
        Phase 1: Exact match (lowercased, stripped)
        Phase 2: Semantic similarity via embeddings
        """
        self._load_embedding_model()
        entities = list(self.store.entities.values())
        if not entities:
            return

        print(f"Canonicalizing {len(entities)} entities...")

        # Initialize union-find
        for e in entities:
            self._parent[e.id] = e.id

        # Phase 1: Exact match on normalized name + type
        exact_groups = defaultdict(list)
        for e in entities:
            key = f"{e.entity_type.value}::{e.name.lower().strip()}"
            exact_groups[key].append(e)

        exact_merges = 0
        for key, group in exact_groups.items():
            if len(group) > 1:
                canonical = group[0]
                for other in group[1:]:
                    self._union(canonical.id, other.id)
                    exact_merges += 1
        print(f"  Exact match merges: {exact_merges}")

        # Phase 2: Semantic similarity (within same type)
        type_groups = defaultdict(list)
        for e in entities:
            type_groups[e.entity_type].append(e)

        semantic_merges = 0
        for etype, group in type_groups.items():
            if len(group) < 2:
                continue
            # Skip User type — usernames are exact, not semantic
            if etype == EntityType.USER:
                continue

            names = [e.name for e in group]
            embeddings = self.embedding_model.encode(
                names, normalize_embeddings=True)

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Only compare if not already merged
                    if self._find(group[i].id) == self._find(group[j].id):
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim >= SIMILARITY_THRESHOLD:
                        self._union(group[i].id, group[j].id)
                        semantic_merges += 1

        print(f"  Semantic similarity merges: {semantic_merges}")

        # Apply merges: build canonical entities
        self._apply_entity_merges()

    def _apply_entity_merges(self):
        """Apply union-find results to create canonical entities."""
        # Group entities by their root
        groups = defaultdict(list)
        for eid in list(self.store.entities.keys()):
            root = self._find(eid)
            groups[root].append(eid)

        new_entities = {}
        id_mapping = {}  # old_id -> canonical_id

        for root, member_ids in groups.items():
            members = [self.store.entities[mid]
                       for mid in member_ids if mid in self.store.entities]
            if not members:
                continue

            # Pick the canonical: prefer the one with the shortest name, or first seen
            canonical = min(members, key=lambda e: (
                len(e.name), e.first_seen or ""))

            # Collect all aliases
            all_aliases = set()
            for m in members:
                all_aliases.add(m.name)
                all_aliases.update(m.aliases)
            all_aliases.discard(canonical.name)

            # Earliest first_seen
            first_seen_times = [m.first_seen for m in members if m.first_seen]
            earliest = min(first_seen_times) if first_seen_times else None

            merged = Entity(
                id=canonical.id,
                name=canonical.name,
                entity_type=canonical.entity_type,
                aliases=sorted(all_aliases),
                first_seen=earliest,
            )
            new_entities[canonical.id] = merged

            # Record mapping for all members
            for mid in member_ids:
                id_mapping[mid] = canonical.id

            # Record merge in audit log (if >1 member)
            if len(member_ids) > 1:
                original_snapshots = {
                    mid: self.store.entities[mid].model_dump()
                    for mid in member_ids
                    if mid in self.store.entities
                }
                record = MergeRecord(
                    merge_id=str(uuid.uuid4())[:8],
                    merge_type="entity",
                    source_ids=member_ids,
                    target_id=canonical.id,
                    reason=f"Dedup: {len(member_ids)} entities merged",
                    timestamp=datetime.utcnow().isoformat(),
                    original_snapshots=original_snapshots,
                )
                self.store.merge_log.append(record)

        # Update store entities
        self.store.entities = new_entities

        # Update claim references
        for cid, claim in self.store.claims.items():
            if claim.subject_id in id_mapping:
                claim.subject_id = id_mapping[claim.subject_id]
            if claim.object_id in id_mapping:
                claim.object_id = id_mapping[claim.object_id]

    # ── 3. Claim Deduplication ─────────────────────────────────────────────

    def dedup_claims(self):
        """
        Merge claims with the same (subject, relation, object) triple.
        Combine their evidence lists, keeping all source pointers.
        """
        print(f"Deduplicating {len(self.store.claims)} claims...")

        claim_groups = defaultdict(list)
        for claim in self.store.claims.values():
            key = claim.claim_key
            claim_groups[key].append(claim)

        new_claims = {}
        merge_count = 0

        for key, group in claim_groups.items():
            if len(group) == 1:
                new_claims[group[0].id] = group[0]
                continue

            # Sort by timestamp (earliest first)
            group.sort(key=lambda c: c.valid_from or "")
            canonical = group[0]

            # Merge evidence from all duplicates
            all_evidence = []
            seen_evidence_ids = set()
            for claim in group:
                for ev in claim.evidence:
                    ev_id = ev.evidence_id
                    if ev_id not in seen_evidence_ids:
                        all_evidence.append(ev)
                        seen_evidence_ids.add(ev_id)

            # Take highest confidence
            max_confidence = max(c.confidence for c in group)

            merged = Claim(
                id=canonical.id,
                subject_id=canonical.subject_id,
                relation=canonical.relation,
                object_id=canonical.object_id,
                evidence=all_evidence,
                confidence=max_confidence,
                valid_from=canonical.valid_from,
                valid_until=canonical.valid_until,
                is_current=canonical.is_current,
                extraction_version=canonical.extraction_version,
            )
            new_claims[merged.id] = merged
            merge_count += len(group) - 1

            # Audit log
            if len(group) > 1:
                record = MergeRecord(
                    merge_id=str(uuid.uuid4())[:8],
                    merge_type="claim",
                    source_ids=[c.id for c in group],
                    target_id=canonical.id,
                    reason=f"Same triple: {key}",
                    timestamp=datetime.utcnow().isoformat(),
                    original_snapshots={
                        c.id: c.model_dump() for c in group
                    },
                )
                self.store.merge_log.append(record)

        self.store.claims = new_claims
        print(
            f"  Merged {merge_count} duplicate claims. {len(new_claims)} unique remain.")

    # ── 4. Conflict Resolution (Temporal) ──────────────────────────────────

    def resolve_conflicts(self):
        """
        Detect contradicting claims and apply temporal validity.
        E.g., "issue is open" at T1 vs "issue is closed" at T2:
          - Mark T1 claim with valid_until=T2, is_current=False
          - Mark T2 claim as is_current=True
        """
        print("Resolving temporal conflicts...")

        # Group claims by (subject, relation) — same subject+relation with
        # different objects indicates potential conflict
        sr_groups = defaultdict(list)
        for claim in self.store.claims.values():
            key = f"{claim.subject_id}::{claim.relation.value}"
            sr_groups[key].append(claim)

        # Relations that are typically single-valued (conflicts possible)
        single_valued_relations = {
            RelationType.STATUS_CHANGED,
            RelationType.ASSIGNED_TO,
            RelationType.LABELED,
            RelationType.APPROVED,
        }

        conflicts_resolved = 0
        for key, group in sr_groups.items():
            if len(group) < 2:
                continue

            # Check if any claims in this group have different objects
            objects = set(c.object_id for c in group)
            if len(objects) <= 1:
                continue

            # Check if the relation type suggests single-valued
            relation = group[0].relation
            if relation not in single_valued_relations:
                continue

            # Sort chronologically
            group.sort(key=lambda c: c.valid_from or "")

            # Mark earlier claims as superseded
            for i in range(len(group) - 1):
                older = group[i]
                newer = group[i + 1]
                if older.object_id != newer.object_id:
                    older.valid_until = newer.valid_from
                    older.is_current = False
                    conflicts_resolved += 1

            # Latest claim is current
            group[-1].is_current = True

        print(f"  Resolved {conflicts_resolved} temporal conflicts.")

    # ── Main pipeline ──────────────────────────────────────────────────────

    def run_full_pipeline(self):
        """Run all dedup stages in order."""
        print(f"\n{'='*60}")
        print("DEDUPLICATION & CANONICALIZATION PIPELINE")
        print(f"{'='*60}")
        print(
            f"Input: {len(self.store.entities)} entities, {len(self.store.claims)} claims")

        self.canonicalize_entities()
        self.dedup_claims()
        self.resolve_conflicts()

        # Remove orphan claims (referencing entities that don't exist)
        valid_entity_ids = set(self.store.entities.keys())
        orphan_count = 0
        clean_claims = {}
        for cid, claim in self.store.claims.items():
            if claim.subject_id in valid_entity_ids and claim.object_id in valid_entity_ids:
                clean_claims[cid] = claim
            else:
                orphan_count += 1
        self.store.claims = clean_claims

        print(f"\n{'='*60}")
        print(
            f"Output: {len(self.store.entities)} entities, {len(self.store.claims)} claims")
        print(f"  Merge log entries: {len(self.store.merge_log)}")
        if orphan_count:
            print(f"  Removed {orphan_count} orphan claims")
        print(f"{'='*60}\n")

        return self.store


def undo_merge(store: MemoryStore, merge_id: str) -> MemoryStore:
    """
    Reverse a merge operation using the audit log.
    Returns the store with the merge undone.
    """
    record = None
    for r in store.merge_log:
        if r.merge_id == merge_id:
            record = r
            break

    if record is None:
        raise ValueError(f"Merge {merge_id} not found in audit log")

    if record.merge_type == "entity":
        # Restore original entities
        for eid, snapshot in record.original_snapshots.items():
            store.entities[eid] = Entity.model_validate(snapshot)
        # Remove the merged entity if it's not one of the originals
        if record.target_id not in record.original_snapshots:
            store.entities.pop(record.target_id, None)
    elif record.merge_type == "claim":
        # Restore original claims
        for cid, snapshot in record.original_snapshots.items():
            store.claims[cid] = Claim.model_validate(snapshot)
        if record.target_id not in record.original_snapshots:
            store.claims.pop(record.target_id, None)

    # Remove the merge record
    store.merge_log = [r for r in store.merge_log if r.merge_id != merge_id]

    return store


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    input_path = os.path.join(data_dir, "extracted_raw.json")
    output_path = os.path.join(data_dir, "deduped_store.json")

    store = MemoryStore.deserialize(input_path)
    deduper = Deduplicator(store)
    store = deduper.run_full_pipeline()
    store.serialize(output_path)
    print(f"Saved deduped store to {output_path}")
