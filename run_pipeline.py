"""
Runs the whole pipeline from start to finish: fetch data, extract, dedup, build graph,
generate example queries, and copy everything to the outputs folder.

Usage:
    python run_pipeline.py              # run everything
    python run_pipeline.py --skip-fetch # skip the download step
    python run_pipeline.py --skip-extract # skip the LLM extraction step
"""

import argparse
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_fetch():
    """Download and process the Enron email corpus."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING CORPUS (Enron Email Dataset)")
    print("=" * 60)
    raw_path = os.path.join(DATA_DIR, "raw_corpus.json")
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            data = json.load(f)
        print(f"  Corpus already exists: {len(data)} email threads. Skipping fetch.")
        return
    from fetch_corpus import main as fetch_main
    fetch_main()


def step_extract():
    """Run the LLM to pull entities and claims out of each email."""
    print("\n" + "=" * 60)
    print("STEP 2: STRUCTURED EXTRACTION (Ollama)")
    print("=" * 60)
    corpus_path = os.path.join(DATA_DIR, "raw_corpus.json")
    output_path = os.path.join(DATA_DIR, "extracted_raw.json")
    if os.path.exists(output_path):
        print(f"  Extracted data already exists. Skipping extraction.")
        return
    from extraction import run_extraction
    run_extraction(corpus_path, output_path)


def step_dedup():
    """Clean up duplicates and merge equivalent entities."""
    print("\n" + "=" * 60)
    print("STEP 3: DEDUPLICATION & CANONICALIZATION")
    print("=" * 60)
    input_path = os.path.join(DATA_DIR, "extracted_raw.json")
    output_path = os.path.join(DATA_DIR, "deduped_store.json")

    from schema import MemoryStore
    from deduplication import Deduplicator

    store = MemoryStore.deserialize(input_path)
    deduper = Deduplicator(store)
    store = deduper.run_full_pipeline()
    store.serialize(output_path)
    print(f"  Saved to {output_path}")


def step_graph():
    """Turn the deduped store into a proper NetworkX graph."""
    print("\n" + "=" * 60)
    print("STEP 4: BUILD MEMORY GRAPH")
    print("=" * 60)
    store_path = os.path.join(DATA_DIR, "deduped_store.json")
    graph_path = os.path.join(DATA_DIR, "memory_graph.json")

    from graph import build_graph
    build_graph(store_path, graph_path)


def step_retrieval_examples():
    """Run some sample queries to show off what the retrieval can do."""
    print("\n" + "=" * 60)
    print("STEP 5: GENERATE EXAMPLE CONTEXT PACKS")
    print("=" * 60)
    graph_path = os.path.join(DATA_DIR, "memory_graph.json")
    output_path = os.path.join(OUTPUT_DIR, "example_context_packs.json")

    from retrieval import generate_example_context_packs
    generate_example_context_packs(graph_path, output_path)


def step_serialize_outputs():
    """Copy the final files into the outputs directory for easy access."""
    print("\n" + "=" * 60)
    print("STEP 6: SERIALIZE OUTPUTS")
    print("=" * 60)
    import shutil
    for fname in ["deduped_store.json", "memory_graph.json"]:
        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname} to outputs/")

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
    print("  To launch the visualization: streamlit run app.py")


def main():
    parser = argparse.ArgumentParser(description="Run the memory graph pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching step")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip LLM extraction step")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_fetch:
        step_fetch()
    else:
        print("Skipping fetch step.")

    if not args.skip_extract:
        step_extract()
    else:
        print("Skipping extraction step.")

    step_dedup()
    step_graph()
    step_retrieval_examples()
    step_serialize_outputs()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Start the visualization: streamlit run app.py")
    print("  2. Open your browser to the displayed URL")
    print("  3. Explore the graph, query it, and inspect evidence")


if __name__ == "__main__":
    main()
