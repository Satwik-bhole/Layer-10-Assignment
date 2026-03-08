"""
Deduplication and canonicalization pipeline.

Deals with three kinds of duplicate data that show up in email corpora:
  1. Artifact dedup — near-identical texts from quoted replies and forwards
  2. Entity canonicalization — merging different names for the same thing
  3. Claim dedup — combining repeated assertions while keeping all evidence

Also handles temporal conflict resolution (which version of a fact is current)
and keeps a full merge audit log so any merge can be reversed.
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

    # -- Union-Find helpers (for grouping entities that should merge) --

    def _find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression to keep it fast
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    # -- Artifact Dedup --

    def dedup_artifacts(self, corpus: list[dict]) -> list[dict]:
        """
        Spot and flag near-duplicate message texts. Emails get quoted and
        forwarded all the time, so we strip out the quoted junk, fingerprint
        what's left, and mark duplicates.
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
                # get rid of '>' quoted lines from replies
                clean_text = re.sub(
                    r"^>+.*$", "", clean_text, flags=re.MULTILINE)
                # chop off everything after "--- Original Message ---"
                clean_text = re.sub(
                    r"-{2,}\s*Original Message\s*-{2,}.*",
                    "", clean_text, flags=re.DOTALL | re.IGNORECASE
                )
                # same for forwarded-by blocks
                clean_text = re.sub(
                    r"-{2,}\s*Forwarded by.*",
                    "", clean_text, flags=re.DOTALL | re.IGNORECASE
                )
                # strip email signatures (the -- marker)
                clean_text = re.sub(
                    r"\n--\s*\n.*", "", clean_text, flags=re.DOTALL
                )
                # collapse whitespace
                clean_text = re.sub(r"\s+", " ", clean_text).strip()

                if len(clean_text) < 20:
                    unique_comments.append(comment)
                    continue

                # hash the cleaned text to quickly spot duplicates
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

    # -- Entity Canonicalization --

    def canonicalize_entities(self):
        """
        Merge entities that are really the same thing but got extracted with
        different names. First pass does exact matching (case-insensitive),
        second pass uses embedding similarity to catch fuzzy matches.
        """
        self._load_embedding_model()
        entities = list(self.store.entities.values())
        if not entities:
            return

        print(f"Canonicalizing {len(entities)} entities...")

        # Initialize union-find
        for e in entities:
            self._parent[e.id] = e.id

        # first pass: group by exact name+type match
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

        # second pass: use embeddings to catch semantically similar entities
        type_groups = defaultdict(list)
        for e in entities:
            type_groups[e.entity_type].append(e)

        semantic_merges = 0
        for etype, group in type_groups.items():
            if len(group) < 2:
                continue
            # user names/emails are already exact identifiers, skip semantic matching
            if etype == EntityType.USER:
                continue

            names = [e.name for e in group]
            embeddings = self.embedding_model.encode(
                names, normalize_embeddings=True)

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # no point comparing if they're already in the same group
                    if self._find(group[i].id) == self._find(group[j].id):
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim >= SIMILARITY_THRESHOLD:
                        self._union(group[i].id, group[j].id)
                        semantic_merges += 1

        print(f"  Semantic similarity merges: {semantic_merges}")

        # now actually apply all the merges we found
        self._apply_entity_merges()

    def _apply_entity_merges(self):
        """Take the union-find results and build the final merged entity list."""
        # figure out which entities ended up in the same group
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

            # pick the entity with the shortest name as the canonical one
            canonical = min(members, key=lambda e: (
                len(e.name), e.first_seen or ""))

            # gather up all the different names into aliases
            all_aliases = set()
            for m in members:
                all_aliases.add(m.name)
                all_aliases.update(m.aliases)
            all_aliases.discard(canonical.name)

            # use the earliest timestamp we've seen for any of these entities
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

            # remember which old IDs point to which canonical ID
            for mid in member_ids:
                id_mapping[mid] = canonical.id

            # log the merge so we can undo it later if needed
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

        self.store.entities = new_entities

        # update all claims to point to the new canonical entity IDs
        for cid, claim in self.store.claims.items():
            if claim.subject_id in id_mapping:
                claim.subject_id = id_mapping[claim.subject_id]
            if claim.object_id in id_mapping:
                claim.object_id = id_mapping[claim.object_id]

    # -- Claim Deduplication --

    def dedup_claims(self):
        """
        When multiple emails say the same thing (same subject-relation-object triple),
        merge them into one claim but keep all the evidence from each.
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

            # keep the earliest one as the base and merge the rest in
            group.sort(key=lambda c: c.valid_from or "")
            canonical = group[0]

            # combine all the evidence from every duplicate
            all_evidence = []
            seen_evidence_ids = set()
            for claim in group:
                for ev in claim.evidence:
                    ev_id = ev.evidence_id
                    if ev_id not in seen_evidence_ids:
                        all_evidence.append(ev)
                        seen_evidence_ids.add(ev_id)

            # use the best confidence score from any of the duplicates
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

            # keep a record of what we merged for transparency
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

    # -- Temporal Conflict Resolution --

    def resolve_conflicts(self):
        """
        When we have contradicting claims (like "issue is open" then later "issue is
        closed"), mark the older one as historical and the newer one as current.
        """
        print("Resolving temporal conflicts...")

        # group claims by subject + relation to find potential conflicts
        sr_groups = defaultdict(list)
        for claim in self.store.claims.values():
            key = f"{claim.subject_id}::{claim.relation.value}"
            sr_groups[key].append(claim)

        # these relations can only have one true value at a time
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

            # see if claims in this group point to different objects (= conflict)
            objects = set(c.object_id for c in group)
            if len(objects) <= 1:
                continue

            # only flag conflicts for relations where there should be one answer
            relation = group[0].relation
            if relation not in single_valued_relations:
                continue

            # line them up by time
            group.sort(key=lambda c: c.valid_from or "")

            # older claims get marked as superseded by newer ones
            for i in range(len(group) - 1):
                older = group[i]
                newer = group[i + 1]
                if older.object_id != newer.object_id:
                    older.valid_until = newer.valid_from
                    older.is_current = False
                    conflicts_resolved += 1

            # the most recent one wins
            group[-1].is_current = True

        print(f"  Resolved {conflicts_resolved} temporal conflicts.")

    # -- Run everything --

    def run_full_pipeline(self):
        """Run all the dedup stages in the right order."""
        print(f"\n{'='*60}")
        print("DEDUPLICATION & CANONICALIZATION PIPELINE")
        print(f"{'='*60}")
        print(
            f"Input: {len(self.store.entities)} entities, {len(self.store.claims)} claims")

        self.canonicalize_entities()
        self.dedup_claims()
        self.resolve_conflicts()

        # clean up any claims that reference entities we don't have anymore
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
    Roll back a specific merge using the snapshots we saved in the audit log.
    Puts the original entities/claims back the way they were.
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
