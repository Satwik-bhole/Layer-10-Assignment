# Layer10 Take-Home: Grounded Long-Term Memory

A system that turns the **Enron Email Dataset** (corporate email discussions, decisions, escalations) into grounded long-term memory via structured extraction, robust deduplication, and a queryable context graph.

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ensure Ollama is running
ollama pull llama3.1

# Run full pipeline
python run_pipeline.py

# Launch visualization
streamlit run app.py
```

## Architecture

```
fetch_corpus.py  →  extraction.py  →  deduplication.py  →  graph.py  →  retrieval.py
    (Enron          (entities +        (canonical +        (NetworkX     (context
     emails)         claims +           merged +            graph)        packs)
                     evidence)          temporal)
                                                                    ↓
                                                              app.py (Streamlit UI)
```

## Files

| File | Purpose |
|------|---------|
| `schema.py` | Pydantic ontology (Entity, Claim, Evidence, MergeRecord) |
| `fetch_corpus.py` | Download & process Enron Email Dataset (CMU source) |
| `extraction.py` | LLM extraction pipeline (Ollama + validation) |
| `deduplication.py` | Entity/claim dedup, conflict resolution, reversibility |
| `graph.py` | NetworkX memory graph with observability |
| `retrieval.py` | Semantic retrieval → grounded context packs |
| `app.py` | Streamlit visualization (graph, chat, evidence, merges, metrics) |
| `run_pipeline.py` | End-to-end pipeline runner |
| `WRITEUP.md` | Detailed write-up (ontology, dedup strategy, Layer10 adaptation) |

## Corpus

**Enron Email Dataset** (CMU mirror) — 12 email threads from 7 active Enron employees' mailboxes, covering executive decisions, legal discussions, energy trading, government affairs, and logistics. Parsed, threaded, and filtered from the original maildir format.

See [WRITEUP.md](WRITEUP.md) for full details.
