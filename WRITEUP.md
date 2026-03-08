# Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

## Project Overview

This system turns scattered corporate email communications (the **Enron Email Dataset** — internal discussions, decisions, escalations, project updates) into **grounded long-term memory**: a queryable knowledge graph where every claim is traceable to exact source evidence.

### Key Properties
- **Grounded**: Every claim points to verbatim source quotes with source IDs, character offsets, and timestamps
- **Deduplicated**: Multi-level dedup (artifact, entity, claim) with safe, reversible merges
- **Temporal**: Tracks when facts were true (validity time) vs when they were stated (event time)
- **Queryable**: Semantic + keyword retrieval returns ranked, grounded context packs
- **Explorable**: Interactive graph visualization with evidence click-through

---

## Corpus

**Source**: Enron Email Dataset (CMU mirror)  
**Obtained via**: `fetch_corpus.py` — downloads from `https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz`  
**Selection**: 7 active Enron employees' mailboxes (kaminski-v, dasovich-j, mann-k, shackleton-s, lay-k, germany-c, farmer-d) — covering executive decisions, legal discussions, energy trading, government affairs, and logistics  
**Processing**: Emails are parsed with Python's `email` module, grouped into conversation threads by normalized subject line, then filtered to threads with 12–15 messages  
**Total**: 12 email threads with 12–15 messages each, producing rich multi-party discussion data

### Why Enron?
The Enron Email Dataset is explicitly suggested by the assignment as a corpus that exercises:
- **Email threads**: Forwarding, quoting, and reply chains (artifact dedup challenge)
- **Identity resolution**: Same person appears with different email addresses, display names, and aliases
- **Rich relationships**: Discussions, decisions, escalations, approvals between people, teams, and projects
- **Temporal dynamics**: Discussions evolve over time; decisions are made, revised, and superseded

### How to Reproduce
```bash
python fetch_corpus.py
```
Downloads the Enron tar.gz from CMU (~1.7 GB, with resume support via `curl -C -`), extracts only the 7 target mailboxes, parses emails, groups into threads, and saves `data/raw_corpus.json`.

---

## Ontology / Schema (`schema.py`)

### Entity Types
| Type | Description | Examples |
|------|-------------|---------|
| `User` | Email sender/recipient | `vince.kaminski@enron.com`, `Kenneth Lay` |
| `Component` | System, department, or unit | `Trading Desk`, `Legal Department` |
| `Organization` | Company or external entity | `Enron Corp`, `FERC`, `PG&E` |
| `Topic` | Discussion subject / theme | `California Energy Crisis`, `Risk Management` |
| `Project` | Business project or deal | `Project Raptor`, `Broadband deal` |
| `Decision` | A decision or policy | `Approve the contract`, `Reject the proposal` |
| `Concept` | A general concept or term | `Derivatives`, `Mark-to-market` |
| `Bug` / `Feature` / `Issue` / `Label` | Retained for generality | (used if discussion references software/systems) |

### Relation Types
**Core**: `created`, `reported`, `fixed`, `broke`, `depends_on`, `mentions`, `decided`, `assigned_to`, `labeled`, `status_changed`, `duplicate_of`, `related_to`, `caused_by`, `blocked_by`, `implements`, `reverted`

**Email-specific**: `sent_to`, `forwarded_to`, `discussed`, `approved`, `escalated_to`

### Evidence Model
Every claim carries a list of `Evidence` objects:
```python
class Evidence:
    source_id: str          # Email message-ID or thread + message index
    exact_quote: str        # Verbatim excerpt from the email body
    char_offset_start: int  # Character offset in source text
    char_offset_end: int    # Character offset end
    timestamp: str          # ISO 8601, when the email was sent
    author: str             # Email sender
```

### Temporal Model
Claims have dual-time tracking:
- `valid_from`: When the fact became true (event time)
- `valid_until`: When the fact was superseded (validity time)
- `is_current`: Whether this is the latest known truth

This handles cases like: A project is "approved" at T1, then "cancelled" at T2. Both claims exist; T1's claim has `valid_until=T2, is_current=False`.

---

## Extraction Pipeline (`extraction.py`)

### How It Works
1. For each email message, construct a structured prompt instructing the LLM to extract entities and claims
2. Call Ollama (Llama 3.1 8B) with temperature=0.1 for deterministic output
3. Parse JSON response with auto-repair (strip markdown fences, regex extraction, LLM self-repair)
4. Validate against Pydantic schema; skip invalid/low-confidence items
5. Checkpoint after each email thread for resumability

### Email-Specific Extraction
- **People**: Extracted from From/To/CC headers AND mentioned in body text
- **Organizations**: Companies, regulatory bodies, counterparties
- **Topics/Projects**: Business deals, energy markets, regulatory proceedings
- **Decisions**: Approvals, rejections, escalations found in email body
- **Forwarded content**: The LLM is instructed to focus on new content and ignore quoted/forwarded blocks

### Validation & Repair
- `ValidationError` from Pydantic triggers auto-repair prompts
- Invalid entity types fall back to `Concept`
- Invalid relation types fall back to `related_to`
- Claims without evidence are discarded (quality gate)
- Claims below confidence threshold (0.5) are discarded

### Versioning
- Every claim carries `extraction_version` (currently `v1`)
- Changing the prompt/model requires bumping the version
- Backfilling: delete `extracted_raw.json` and re-run extraction; the checkpoint system handles incremental progress

### Quality Gates
- **Confidence threshold**: Claims with confidence < 0.5 are dropped
- **Evidence requirement**: Claims without any evidence are dropped
- **Schema validation**: Invalid JSON is auto-repaired or skipped
- **Cross-evidence**: During dedup, claims supported by multiple independent emails get boosted confidence

---

## Deduplication and Canonicalization (`deduplication.py`)

### Three Levels of Dedup

#### 1. Artifact Dedup
Detects near-identical source texts from email quoting patterns:
- Strip `>` prefixed quoted lines (multi-level quoting)
- Remove `--- Original Message ---` forwarded blocks
- Remove `-----Forwarded by...` blocks
- Strip email signature blocks (`--` marker)
- MD5 fingerprint the cleaned text
- Mark duplicates with pointers to originals

#### 2. Entity Canonicalization (Union-Find)
**Exact Match**: Lowercase + strip whitespace. `"vince kaminski"` and `"Vince Kaminski"` → same entity.

**Semantic Match**: Embed entity names with `all-MiniLM-L6-v2` (sentence-transformers). If cosine similarity > 0.90, merge into canonical entity. Example: `"Energy Trading Desk"` and `"ENA Trading"` → merged.

**Implementation**: Union-Find data structure with path compression for efficient grouping. User entities are exempt from semantic matching (email addresses are exact).

**Result**: Canonical entity gets the shortest name; all original names become aliases.

#### 3. Claim Dedup
Claims with identical `(subject_id, relation, object_id)` triples are merged:
- Evidence lists are **combined** (not replaced) — the merged claim has all source pointers
- Highest confidence is kept
- Earliest `valid_from` is preserved
- All merge operations are logged

### Conflict Resolution (Temporal)
For single-valued relations (`status_changed`, `assigned_to`, `labeled`, `approved`):
- Claims are sorted chronologically
- Earlier claims get `valid_until = newer.valid_from` and `is_current = False`
- The latest claim is marked `is_current = True`

### Reversibility
Every merge creates a `MergeRecord` in the audit log containing:
- Original entity/claim snapshots (full JSON)
- Merge reason and timestamp
- Source and target IDs

The `undo_merge()` function restores originals from snapshots.

---

## Memory Graph Design (`graph.py`)

### Structure
- **Library**: NetworkX `MultiDiGraph` (multiple directed edges between same nodes)
- **Nodes**: Canonical entities with attributes (name, type, aliases, first_seen)
- **Edges**: Claims with attributes (relation, evidence list, confidence, temporal validity)

### Properties
- **Queryable**: Find entities by name/alias, traverse neighbors, extract subgraphs
- **Grounded**: Every edge carries evidence pointers back to source email text
- **Maintainable**: Incremental ingestion with idempotency (duplicate IDs are skipped)

### Time Model
- **Event time**: When an email was sent (`timestamp` in evidence)
- **Validity time**: When a claim was true (`valid_from` / `valid_until` on edges)
- Queries can filter by `is_current` to see only latest truth, or include historical data

### Idempotency
Entity and claim IDs are tracked in `_ingested_entity_ids` / `_ingested_claim_ids`. Re-ingesting the same store is a no-op.

### Observability
`get_metrics()` reports:
- Graph statistics (nodes, edges, connected components)
- Entity/relation type distributions
- Evidence quality (avg evidence per claim, claims with no evidence)
- Confidence distribution
- Temporal coverage (current vs historical claims)
- Ingestion metrics (duplicates skipped, orphans cleaned)

### Permissions (Conceptual)
In a production system for Layer10's target environment, permissions would be enforced at retrieval time:
- Each evidence pointer includes `source_id` traceable to the original email
- A permission layer would check if the querying user has access to the underlying mailbox
- Claims whose evidence is entirely from restricted sources would be filtered out
- This ensures memory retrieval is constrained by access to underlying sources

---

## Retrieval and Grounding (`retrieval.py`)

### Query Flow
1. **Embed query** using `all-MiniLM-L6-v2`
2. **Entity matching**: Combine keyword matching (substring search) with semantic similarity against pre-computed entity name embeddings
3. **Graph traversal**: For matched entities, gather all neighboring claims (edges)
4. **Ranking**: Score each claim by `entity_match_score × confidence × recency_boost`
   - `recency_boost = 1.2` for current claims, `0.8` for historical
5. **Context packing**: Top-K claims formatted with full evidence chains
6. **Answer generation** (optional): Send context pack to Ollama with citation instructions

### Handling Ambiguity and Conflicts
- When multiple entities match, all are included with match scores
- Conflicting claims (historical vs current) are both shown, clearly labeled
- The user can see both sides and click through to evidence

### Context Pack Format
```json
{
  "query": "What did Vince Kaminski discuss about risk?",
  "matched_entities": [...],
  "items": [
    {
      "claim_id": "...",
      "subject": "Vince Kaminski",
      "relation": "discussed",
      "object": "Risk Management",
      "confidence": 0.9,
      "is_current": true,
      "valid_from": "2001-03-15T...",
      "evidence": [
        {"source_id": "msg_123", "exact_quote": "...", "timestamp": "..."}
      ]
    }
  ]
}
```

---

## Visualization (`app.py`)

### Streamlit App with 5 Pages

1. **Graph View**: Interactive pyvis graph with filters (entity type, relation type, confidence, current-only). Nodes colored by entity type; edges show relation labels; hover reveals evidence.

2. **Query / Chat**: Text input → retrieves grounded context pack → shows matched entities, claims with evidence expandable panels, and (optionally) an LLM-generated answer citing sources.

3. **Evidence Inspector**: Search for any entity → see all connected claims with their exact source quotes, authors, and timestamps.

4. **Merge Inspector**: View entities with aliases (merged), browse the full merge audit log, inspect original snapshots before merge.

5. **Quality Metrics**: Dashboard showing graph stats, evidence quality, confidence distribution, temporal coverage, entity/relation type distributions.

### How to Run
```bash
streamlit run app.py
```

---

## Layer10 Considerations

### Adapting to Slack, Jira/Linear, Other Sources

**Ontology Changes**:
- Add entity types: `Team`, `Channel`, `Thread`, `Document`, `Ticket`
- Add relations: `resolved_in`, `referenced_from`, `rejected`
- Tickets get structured fields (status, priority, assignee) extracted as typed claims

**Extraction Contract**:
- Slack: Thread structure gives natural conversation boundaries; reactions and edits tracked as temporal claims
- Jira/Linear: Status transitions are first-class events; custom fields map to the ontology
- Cross-system: Same decision discussed in email + documented in Jira → merged with evidence from both

**Dedup Strategy**:
- **Unstructured + structured fusion**: Cross-reference Slack discussion threads with Jira tickets via entity canonicalization
- **Email chains**: Artifact dedup handles quoted replies, forwarded messages, and signatures
- **Identity resolution**: `Vince Kaminski <vince.kaminski@enron.com>` and `@vkaminski` on Slack → same entity
- **Cross-system dedup**: Same decision discussed in email + documented in Jira → merged with evidence from both

**Grounding & Safety**:
- Provenance chains: every claim traces back through a chain of evidence to original sources
- Deletions/redactions: when a source is deleted, claims grounded in it are marked with `redacted = True` but the claim structure is preserved
- Citations in generated answers always reference source IDs that can be looked up

**Long-Term Memory**:
- **Durable vs ephemeral**: High-confidence claims with multiple independent evidence sources become durable memory; single-evidence, low-confidence claims decay over time
- **Drift prevention**: Extraction version tracking; re-extraction with new models creates new `extraction_version` claims that can be compared against previous versions
- **Incremental updates**: Idempotent ingestion allows processing new emails without reprocessing the full history

**Permissions**:
- Memory retrieval constrained by source access: if a user can't access the original email thread, claims grounded only in that thread are filtered out
- Claims with mixed evidence (some accessible, some not) show only accessible evidence

**Operational Reality**:
- **Scaling**: Graph partitioning by project/team; embedding index uses approximate nearest neighbors (FAISS/Annoy) for large corpora
- **Cost**: Local Ollama models for extraction (no API costs); embedding model is lightweight (~80MB)
- **Evaluation/regression**: Track extraction quality metrics over time; alert on drift (e.g., avg confidence drops, evidence-per-claim drops)

---

## Reproducibility

### Full Pipeline (end-to-end)

```bash
# 1. Clone and setup
cd Layer_10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Ensure Ollama is running with llama3.1
ollama pull llama3.1

# 3. Run the full pipeline
python run_pipeline.py

# 4. Launch visualization
streamlit run app.py
```

### Step-by-Step
```bash
# Fetch corpus (Enron email dataset from CMU)
python fetch_corpus.py

# Extract entities/claims via Ollama
python extraction.py

# Deduplicate and canonicalize
python deduplication.py

# Build memory graph
python graph.py

# Generate example context packs
python retrieval.py

# Launch visualization
streamlit run app.py
```

### Output Files
- `data/raw_corpus.json` — Raw corpus (12 email threads, 12–15 messages each)
- `data/extracted_raw.json` — Raw extraction output (entities + claims)
- `data/deduped_store.json` — Deduplicated memory store
- `data/memory_graph.json` — Serialized NetworkX graph
- `outputs/deduped_store.json` — Copy of final store
- `outputs/memory_graph.json` — Copy of final graph
- `outputs/example_context_packs.json` — Sample retrieval results

---

## Technology Stack

| Component | Tool | Justification |
|-----------|------|---------------|
| Extraction | Ollama + Llama 3.1 (8B) | Free, local, good at JSON/instruction-following |
| Schema | Pydantic v2 | Strict typed validation, JSON serialization |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Lightweight, local, good for semantic similarity |
| Graph | NetworkX MultiDiGraph | In-memory, multi-edge support, Python-native |
| Visualization | Streamlit + pyvis | Interactive web UI, graph rendering, zero frontend code |
| Data format | JSON | Human-readable, portable, no database dependencies |
