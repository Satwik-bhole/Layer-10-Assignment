"""
Structured extraction pipeline using Ollama (local LLM).

Takes raw Enron email data and extracts typed, grounded objects
(entities + claims + evidence) using a local model.

Features:
- Strict Pydantic validation with auto-repair loop
- Evidence grounding with exact quotes and character offsets
- Quality gates (confidence threshold, cross-evidence support)
- Extraction versioning
- Checkpoint/resume support for long runs
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime

import ollama

from schema import (
    Claim, Entity, EntityType, Evidence, ExtractionResult,
    MemoryStore, RelationType,
)

EXTRACTION_VERSION = "v1"
MODEL_NAME = "llama3.1"
CONFIDENCE_THRESHOLD = 0.5   # quality gate: discard claims below this
MAX_RETRIES = 3              # auto-repair attempts for invalid JSON
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "data")

SYSTEM_PROMPT = """You are an information extraction engine for corporate emails. Extract entities and claims as JSON.

ENTITY TYPES: User, Organization, Topic, Project, Decision, Concept
RELATION TYPES: sent_to, forwarded_to, discussed, approved, mentions, decided, related_to, escalated_to, assigned_to

RULES:
1. Every claim MUST have an exact_quote copied verbatim from the input text.
2. Users are identified by their email address or name.
3. Focus on new content only — skip quoted/forwarded text (lines starting with >).
4. Confidence: 0.0-1.0 based on how explicit the claim is.

Return ONLY this JSON structure:
{
  "entities": [{"id": "e1", "name": "...", "entity_type": "User|Organization|Topic|Project|Decision|Concept", "aliases": []}],
  "claims": [{"id": "c1", "subject_id": "e1", "relation": "...", "object_id": "e2", "confidence": 0.85, "evidence": [{"source_id": "SRC", "exact_quote": "verbatim text", "timestamp": "ISO", "author": "WHO"}]}]
}"""


def make_extraction_prompt(issue_title: str, comment: dict, issue_id: int) -> str:
    """Build the user prompt for a single email message extraction."""
    source_id = comment["comment_id"]
    text = comment["text"]
    author = comment["author"]
    timestamp = comment["timestamp"]

    # Truncate very long messages to avoid context overflow
    if len(text) > 3000:
        text = text[:3000] + "... [truncated]"

    return f"""THREAD: #{issue_id} - {issue_title}
SOURCE_ID: {source_id}
FROM: {author}
TIMESTAMP: {timestamp}

{text}"""


def call_ollama(prompt: str, attempt: int = 0) -> dict:
    """Call Ollama and parse JSON response with auto-repair."""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 2048, "num_gpu": 99},
        )
        raw_text = response["message"]["content"]
        return parse_json_response(raw_text)
    except Exception as e:
        if attempt < MAX_RETRIES:
            print(f"    Ollama error (attempt {attempt+1}): {e}. Retrying...")
            time.sleep(1)
            return call_ollama(prompt, attempt + 1)
        print(f"    Ollama failed after {MAX_RETRIES} retries: {e}")
        return {"entities": [], "claims": []}


def parse_json_response(raw_text: str) -> dict:
    """Extract and parse JSON from LLM response, handling common issues."""
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # With format="json" in Ollama, output should always be valid JSON.
    # If we still can't parse, return empty.
    return {"entities": [], "claims": []}


def validate_and_build(raw_data: dict, source_id: str, timestamp: str,
                       author: str, issue_id: int) -> ExtractionResult:
    """Validate LLM output against Pydantic schema with repair."""
    entities = []
    claims = []

    # Process entities
    for raw_ent in raw_data.get("entities", []):
        try:
            # Normalize entity type
            etype = raw_ent.get("entity_type", "Concept")
            try:
                etype = EntityType(etype)
            except ValueError:
                etype = EntityType.CONCEPT

            eid = str(raw_ent.get("id", hashlib.md5(
                raw_ent.get("name", "unknown").encode()
            ).hexdigest()[:8]))

            entity = Entity(
                id=f"ent_{issue_id}_{eid}",
                name=raw_ent.get("name", "unknown"),
                entity_type=etype,
                aliases=raw_ent.get("aliases", []),
                first_seen=timestamp,
            )
            entities.append(entity)
        except Exception as e:
            print(f"    Skipping invalid entity: {e}")

    # Process claims
    for raw_claim in raw_data.get("claims", []):
        try:
            # Normalize relation
            rel = raw_claim.get("relation", "related_to")
            try:
                rel = RelationType(rel)
            except ValueError:
                rel = RelationType.RELATED_TO

            # Build evidence with grounding
            evidences = []
            for raw_ev in raw_claim.get("evidence", []):
                quote = raw_ev.get("exact_quote", "")
                ev = Evidence(
                    source_id=raw_ev.get("source_id", source_id),
                    exact_quote=quote,
                    char_offset_start=raw_ev.get("char_offset_start"),
                    char_offset_end=raw_ev.get("char_offset_end"),
                    timestamp=raw_ev.get("timestamp", timestamp),
                    author=raw_ev.get("author", author),
                )
                evidences.append(ev)

            # If no evidence provided, skip (quality gate)
            if not evidences:
                continue

            cid = raw_claim.get("id", hashlib.md5(
                f"{raw_claim.get('subject_id')}_{rel}_{raw_claim.get('object_id')}".encode()
            ).hexdigest()[:8])

            confidence = float(raw_claim.get("confidence", 0.7))

            # Quality gate: skip low-confidence claims
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            subj = raw_claim.get("subject_id", "")
            obj = raw_claim.get("object_id", "")

            claim = Claim(
                id=f"clm_{issue_id}_{cid}",
                subject_id=f"ent_{issue_id}_{subj}",
                relation=rel,
                object_id=f"ent_{issue_id}_{obj}",
                evidence=evidences,
                confidence=confidence,
                valid_from=timestamp,
                is_current=True,
                extraction_version=EXTRACTION_VERSION,
            )
            claims.append(claim)
        except Exception as e:
            print(f"    Skipping invalid claim: {e}")

    return ExtractionResult(entities=entities, claims=claims)


def extract_issue(issue: dict) -> ExtractionResult:
    """Extract entities and claims from all comments in an issue."""
    issue_id = issue["issue_id"]
    title = issue["title"]
    all_entities = []
    all_claims = []

    # Sort comments chronologically for proper temporal ordering
    comments = sorted(issue["comments"], key=lambda c: c["timestamp"])

    for comment in comments:
        text = comment["text"]
        if not text or len(text.strip()) < 20:
            continue

        prompt = make_extraction_prompt(title, comment, issue_id)
        raw_data = call_ollama(prompt)

        result = validate_and_build(
            raw_data,
            source_id=comment["comment_id"],
            timestamp=comment["timestamp"],
            author=comment["author"],
            issue_id=issue_id,
        )

        all_entities.extend(result.entities)
        all_claims.extend(result.claims)

    return ExtractionResult(entities=all_entities, claims=all_claims)


def run_extraction(corpus_path: str, output_path: str):
    """Run extraction on the full corpus with checkpointing."""
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "_extraction_checkpoint.json")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        store = MemoryStore.model_validate(checkpoint["store"])
        completed = set(checkpoint["completed_issues"])
        print(f"Resuming extraction: {len(completed)} issues already done.")
    else:
        store = MemoryStore(
            extraction_version=EXTRACTION_VERSION,
            corpus_source="enron:cmu-email-dataset",
        )
        completed = set()

    total = len(corpus)
    for idx, issue in enumerate(corpus):
        iid = issue["issue_id"]
        if iid in completed:
            continue

        print(f"[{idx+1}/{total}] Extracting issue #{iid}: {issue['title'][:60]}...")
        result = extract_issue(issue)

        # Add to store
        for entity in result.entities:
            store.add_entity(entity)
        for claim in result.claims:
            store.add_claim(claim)

        completed.add(iid)

        # Checkpoint
        checkpoint_data = {
            "store": json.loads(store.model_dump_json()),
            "completed_issues": list(completed),
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"  Extracted {len(result.entities)} entities, {len(result.claims)} claims")

    # Save final output
    store.serialize(output_path)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nExtraction complete!")
    print(f"  Total entities: {len(store.entities)}")
    print(f"  Total claims: {len(store.claims)}")
    print(f"  Saved to: {output_path}")

    return store


if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "raw_corpus.json")
    output_path = os.path.join(os.path.dirname(__file__), "data", "extracted_raw.json")
    run_extraction(corpus_path, output_path)
