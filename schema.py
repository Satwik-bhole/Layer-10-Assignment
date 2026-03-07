"""
Pydantic schema / ontology for the memory graph.

Defines the typed, grounded objects that the extraction pipeline produces.
Every claim is traceable to evidence with source_id, exact quote, offsets, and timestamp.

Entity types:  User, Component, Bug, Feature, Issue, Label, Decision, Concept,
               Organization, Topic, Project
Relation types: created, reported, fixed, broke, depends_on, mentions, decided,
                assigned_to, labeled, status_changed, duplicate_of, related_to,
                caused_by, blocked_by, implements, reverted, sent_to, forwarded_to,
                discussed, approved, escalated_to
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    USER = "User"
    COMPONENT = "Component"
    BUG = "Bug"
    FEATURE = "Feature"
    ISSUE = "Issue"
    LABEL = "Label"
    DECISION = "Decision"
    CONCEPT = "Concept"
    ORGANIZATION = "Organization"
    TOPIC = "Topic"
    PROJECT = "Project"


class RelationType(str, Enum):
    CREATED = "created"
    REPORTED = "reported"
    FIXED = "fixed"
    BROKE = "broke"
    DEPENDS_ON = "depends_on"
    MENTIONS = "mentions"
    DECIDED = "decided"
    ASSIGNED_TO = "assigned_to"
    LABELED = "labeled"
    STATUS_CHANGED = "status_changed"
    DUPLICATE_OF = "duplicate_of"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    BLOCKED_BY = "blocked_by"
    IMPLEMENTS = "implements"
    REVERTED = "reverted"
    SENT_TO = "sent_to"
    FORWARDED_TO = "forwarded_to"
    DISCUSSED = "discussed"
    APPROVED = "approved"
    ESCALATED_TO = "escalated_to"


# ── Evidence (grounding) ───────────────────────────────────────────────────────

class Evidence(BaseModel):
    """A grounded pointer to the exact source text supporting a claim."""
    source_id: str = Field(
        ..., description="Unique ID: email message-ID or thread + message index"
    )
    exact_quote: str = Field(
        ..., description="Verbatim excerpt from the source"
    )
    char_offset_start: Optional[int] = Field(
        None, description="Character offset where the quote starts in the source text"
    )
    char_offset_end: Optional[int] = Field(
        None, description="Character offset where the quote ends in the source text"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of the source comment"
    )
    author: Optional[str] = Field(
        None, description="Author of the source comment"
    )

    @property
    def evidence_id(self) -> str:
        content = f"{self.source_id}:{self.exact_quote[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Entity ─────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    """A named thing extracted from the corpus."""
    id: str = Field(..., description="Unique entity identifier (auto-generated)")
    name: str = Field(..., description="Canonical name of the entity")
    entity_type: EntityType = Field(..., description="Type of the entity")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names / surface forms that resolved to this entity"
    )
    first_seen: Optional[str] = Field(
        None, description="ISO 8601 timestamp when first mentioned"
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip()

    def canonical_key(self) -> str:
        """Deterministic key for dedup: lowercased name + type."""
        return f"{self.entity_type.value}::{self.name.lower().strip()}"


# ── Claim ──────────────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """
    A factual assertion extracted from the corpus, linking two entities
    with a typed relation. Every claim carries grounding evidence.

    Supports temporal validity (valid_from / valid_until) for conflict resolution.
    """
    id: str = Field(..., description="Unique claim identifier")
    subject_id: str = Field(..., description="Entity ID of the subject")
    relation: RelationType = Field(..., description="The relationship type")
    object_id: str = Field(..., description="Entity ID of the object")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="List of evidence pointers grounding this claim"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence score (0-1); used as quality gate"
    )
    valid_from: Optional[str] = Field(
        None, description="ISO 8601 timestamp when this claim became true"
    )
    valid_until: Optional[str] = Field(
        None, description="ISO 8601 timestamp when this claim was superseded"
    )
    is_current: bool = Field(
        default=True,
        description="Whether this claim is the latest known truth"
    )
    extraction_version: str = Field(
        default="v1",
        description="Version of the extraction pipeline that produced this"
    )

    @property
    def claim_key(self) -> str:
        """Key for dedup: canonical (subject, relation, object) triple."""
        return f"{self.subject_id}::{self.relation.value}::{self.object_id}"


# ── Extraction output (what LLM returns per comment/issue) ─────────────────────

class ExtractionResult(BaseModel):
    """Output of the LLM extraction for a single source document."""
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


# ── Merge audit log (for reversibility) ───────────────────────────────────────

class MergeRecord(BaseModel):
    """Audit trail for entity or claim merges. Enables undo."""
    merge_id: str
    merge_type: str = Field(..., description="'entity' or 'claim'")
    source_ids: list[str] = Field(..., description="IDs that were merged")
    target_id: str = Field(..., description="Surviving canonical ID")
    reason: str = Field(..., description="Why the merge happened")
    timestamp: str = Field(..., description="When the merge occurred")
    original_snapshots: dict = Field(
        default_factory=dict,
        description="JSON snapshots of the originals before merge (for undo)"
    )


# ── Full memory store (serializable) ──────────────────────────────────────────

class MemoryStore(BaseModel):
    """Top-level container for the entire memory graph state."""
    entities: dict[str, Entity] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    merge_log: list[MergeRecord] = Field(default_factory=list)
    extraction_version: str = "v1"
    corpus_source: str = "enron:cmu-email-dataset"

    def add_entity(self, entity: Entity) -> str:
        self.entities[entity.id] = entity
        return entity.id

    def add_claim(self, claim: Claim) -> str:
        self.claims[claim.id] = claim
        return claim.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_claims_for_entity(self, entity_id: str) -> list[Claim]:
        results = []
        for claim in self.claims.values():
            if claim.subject_id == entity_id or claim.object_id == entity_id:
                results.append(claim)
        return results

    def serialize(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def deserialize(cls, path: str) -> "MemoryStore":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
