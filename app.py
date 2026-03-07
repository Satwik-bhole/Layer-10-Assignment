"""
Streamlit visualization app for the Memory Graph.

Features:
- Interactive graph visualization via pyvis
- Chat interface for querying the graph
- Evidence panel: click a claim to see exact source quotes
- Inspect duplicates/merges (aliases, merged entities, merged claims)
- Filter by time, relation type, confidence
- Observability dashboard with quality metrics
"""

import json
import os
import tempfile

import networkx as nx
import streamlit as st
from pyvis.network import Network

# Must be first Streamlit call
st.set_page_config(page_title="Memory Graph Explorer", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRAPH_PATH = os.path.join(DATA_DIR, "memory_graph.json")
STORE_PATH = os.path.join(DATA_DIR, "deduped_store.json")


# ── Caching ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    from graph import MemoryGraph
    return MemoryGraph.load(GRAPH_PATH)


@st.cache_resource
def load_store():
    from schema import MemoryStore
    return MemoryStore.deserialize(STORE_PATH)


@st.cache_resource
def load_retriever():
    from retrieval import Retriever
    mg = load_graph()
    return Retriever(mg)


# ── Color map for entity types ─────────────────────────────────────────────────

ENTITY_COLORS = {
    "User": "#4CAF50",
    "Component": "#2196F3",
    "Bug": "#f44336",
    "Feature": "#FF9800",
    "Issue": "#9C27B0",
    "Label": "#607D8B",
    "Decision": "#E91E63",
    "Concept": "#00BCD4",
    "Organization": "#FF5722",
    "Topic": "#3F51B5",
    "Project": "#8BC34A",
}

RELATION_COLORS = {
    "created": "#4CAF50",
    "reported": "#FF9800",
    "fixed": "#2196F3",
    "broke": "#f44336",
    "mentions": "#9E9E9E",
    "decided": "#E91E63",
    "status_changed": "#9C27B0",
    "assigned_to": "#795548",
    "related_to": "#607D8B",
    "caused_by": "#f44336",
    "duplicate_of": "#FF5722",
    "sent_to": "#4CAF50",
    "forwarded_to": "#FF9800",
    "discussed": "#3F51B5",
    "approved": "#8BC34A",
    "escalated_to": "#E91E63",
}


# ── Pyvis graph rendering ─────────────────────────────────────────────────────

def render_pyvis_graph(subgraph: nx.MultiDiGraph, height: str = "600px"):
    """Render a NetworkX subgraph as an interactive pyvis HTML graph."""
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    # Physics settings for better layout
    net.set_options(json.dumps({
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 200,
        },
    }))

    # Add nodes
    for nid, data in subgraph.nodes(data=True):
        name = data.get("name", nid)
        etype = data.get("entity_type", "Concept")
        aliases = data.get("aliases", [])
        color = ENTITY_COLORS.get(etype, "#00BCD4")

        title = f"<b>{name}</b><br>Type: {etype}"
        if aliases:
            title += f"<br>Aliases: {', '.join(aliases[:5])}"

        net.add_node(
            nid,
            label=name[:30],
            title=title,
            color=color,
            size=15 + min(subgraph.degree(nid) * 3, 30),
        )

    # Add edges
    for u, v, key, data in subgraph.edges(data=True, keys=True):
        relation = data.get("relation", "related_to")
        confidence = data.get("confidence", 0.5)
        is_current = data.get("is_current", True)
        evidence = data.get("evidence", [])
        color = RELATION_COLORS.get(relation, "#9E9E9E")

        if not is_current:
            color = "#555555"

        title = f"<b>{relation}</b><br>Confidence: {confidence:.2f}"
        title += f"<br>Current: {is_current}"
        title += f"<br>Evidence: {len(evidence)} source(s)"
        if evidence:
            for ev in evidence[:3]:
                quote = ev.get("exact_quote", "")[:100]
                title += f"<br>📝 \"{quote}...\""

        net.add_edge(
            u, v,
            title=title,
            label=relation,
            color=color,
            width=max(1, confidence * 3),
            dashes=not is_current,
        )

    # Save to temp file and read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        return f.name


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("🔍 Memory Graph Explorer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Graph View", "Query / Chat", "Evidence Inspector",
         "Merge Inspector", "Quality Metrics"],
    )
    return page


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_graph_view():
    st.header("Graph View")

    mg = load_graph()
    G = mg.graph

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        entity_types = sorted(set(
            d.get("entity_type", "unknown") for _, d in G.nodes(data=True)
        ))
        selected_types = st.multiselect(
            "Entity Types", entity_types, default=entity_types
        )

    with col2:
        relation_types = sorted(set(
            d.get("relation", "unknown") for _, _, d in G.edges(data=True)
        ))
        selected_relations = st.multiselect(
            "Relation Types", relation_types, default=relation_types
        )

    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        current_only = st.checkbox("Current claims only", value=False)

    # Build filtered subgraph
    filtered_nodes = {
        nid for nid, d in G.nodes(data=True)
        if d.get("entity_type", "unknown") in selected_types
    }

    filtered_graph = nx.MultiDiGraph()
    for nid in filtered_nodes:
        filtered_graph.add_node(nid, **G.nodes[nid])

    for u, v, key, data in G.edges(data=True, keys=True):
        if u in filtered_nodes and v in filtered_nodes:
            if data.get("relation", "unknown") in selected_relations:
                if data.get("confidence", 0) >= min_confidence:
                    if not current_only or data.get("is_current", True):
                        filtered_graph.add_edge(u, v, key=key, **data)

    # Remove isolated nodes
    isolates = list(nx.isolates(filtered_graph))
    filtered_graph.remove_nodes_from(isolates)

    st.info(f"Showing {filtered_graph.number_of_nodes()} entities, "
            f"{filtered_graph.number_of_edges()} claims "
            f"(filtered from {G.number_of_nodes()}/{G.number_of_edges()})")

    # Limit nodes for performance
    if filtered_graph.number_of_nodes() > 200:
        # Take top 200 by degree
        top_nodes = sorted(
            filtered_graph.nodes(),
            key=lambda n: filtered_graph.degree(n),
            reverse=True
        )[:200]
        filtered_graph = filtered_graph.subgraph(top_nodes).copy()
        st.warning(f"Graph truncated to top 200 nodes by degree for performance.")

    if filtered_graph.number_of_nodes() > 0:
        html_path = render_pyvis_graph(filtered_graph)
        with open(html_path, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
        os.unlink(html_path)
    else:
        st.warning("No data to display. Adjust filters or check that data files exist.")


def page_query_chat():
    st.header("Query the Memory Graph")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if query := st.chat_input("Ask a question about the Enron emails..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        retriever = load_retriever()

        with st.chat_message("assistant"):
            with st.spinner("Searching memory graph..."):
                pack = retriever.retrieve(query)

            if not pack.items:
                response = "No relevant information found in the memory graph."
                st.markdown(response)
            else:
                # Show matched entities
                st.markdown("**Matched Entities:**")
                for ent in pack.matched_entities:
                    st.markdown(f"- **{ent['name']}** [{ent['entity_type']}] "
                                f"(score: {ent['match_score']:.2f})")

                st.markdown("---")
                st.markdown("**Relevant Claims:**")

                for item in pack.items:
                    status = "✅ Current" if item.is_current else "⏰ Historical"
                    st.markdown(
                        f"**{item.subject}** → *{item.relation}* → **{item.object_str}** "
                        f"({status}, confidence: {item.confidence:.2f})"
                    )
                    for ev in item.evidence_snippets:
                        with st.expander(f"📝 Evidence: {ev.get('source_id', 'unknown')}"):
                            st.markdown(f"> {ev.get('exact_quote', 'N/A')}")
                            st.caption(f"Author: {ev.get('author', 'unknown')} | "
                                       f"Time: {ev.get('timestamp', 'unknown')}")

                # Try generating an answer via Ollama
                try:
                    with st.spinner("Generating grounded answer..."):
                        answer, _ = retriever.answer_question(query)
                    st.markdown("---")
                    st.markdown("**Generated Answer:**")
                    st.markdown(answer)
                    response = answer
                except Exception:
                    response = "Retrieved context shown above."

            st.session_state.messages.append({
                "role": "assistant",
                "content": response if isinstance(response, str) else "See results above.",
            })


def page_evidence_inspector():
    st.header("Evidence Inspector")
    st.markdown("Inspect the evidence grounding for each claim in the graph.")

    mg = load_graph()
    G = mg.graph

    # Let user search for an entity
    search = st.text_input("Search entity by name:")

    if search:
        matches = mg.find_entities_by_name(search)
        if not matches:
            st.warning("No matching entities found.")
            return

        for nid in matches[:10]:
            node_data = mg.get_node(nid)
            name = node_data.get("name", nid)
            etype = node_data.get("entity_type", "unknown")
            aliases = node_data.get("aliases", [])

            with st.expander(f"🔹 {name} [{etype}]", expanded=True):
                if aliases:
                    st.caption(f"Aliases: {', '.join(aliases)}")

                neighbors = mg.get_neighbors(nid, current_only=False)
                if not neighbors:
                    st.info("No connected claims.")
                    continue

                for edge in neighbors:
                    subj_data = mg.get_node(edge["subject_id"])
                    obj_data = mg.get_node(edge["object_id"])
                    subj = subj_data.get("name", edge["subject_id"]) if subj_data else edge["subject_id"]
                    obj = obj_data.get("name", edge["object_id"]) if obj_data else edge["object_id"]
                    rel = edge.get("relation", "?")
                    conf = edge.get("confidence", 0)
                    is_current = edge.get("is_current", True)
                    status = "Current" if is_current else "Historical"

                    st.markdown(f"**{subj}** → *{rel}* → **{obj}** "
                                f"(confidence: {conf:.2f}, {status})")

                    evidence = edge.get("evidence", [])
                    for ev in evidence:
                        st.markdown(f"> 📝 \"{ev.get('exact_quote', 'N/A')}\"")
                        st.caption(f"Source: {ev.get('source_id', '?')} | "
                                   f"Author: {ev.get('author', '?')} | "
                                   f"Time: {ev.get('timestamp', '?')}")
                    st.markdown("---")


def page_merge_inspector():
    st.header("Merge / Dedup Inspector")
    st.markdown("Inspect entity merges, aliases, and claim deduplication history.")

    store = load_store()

    # Merged entities
    st.subheader("Entities with Aliases (merged)")
    entities_with_aliases = [
        e for e in store.entities.values() if e.aliases
    ]

    if not entities_with_aliases:
        st.info("No merged entities found.")
    else:
        for ent in entities_with_aliases[:50]:
            with st.expander(f"🔀 {ent.name} [{ent.entity_type.value}]"):
                st.markdown(f"**Canonical name:** {ent.name}")
                st.markdown(f"**Aliases:** {', '.join(ent.aliases)}")
                st.markdown(f"**First seen:** {ent.first_seen}")

    # Merge audit log
    st.subheader("Merge Audit Log")
    if not store.merge_log:
        st.info("No merge records found.")
    else:
        for record in store.merge_log[:50]:
            color = "🟢" if record.merge_type == "entity" else "🔵"
            with st.expander(
                f"{color} {record.merge_type.upper()} merge: "
                f"{len(record.source_ids)} items → {record.target_id}"
            ):
                st.markdown(f"**Merge ID:** {record.merge_id}")
                st.markdown(f"**Reason:** {record.reason}")
                st.markdown(f"**Time:** {record.timestamp}")
                st.markdown(f"**Source IDs:** {', '.join(record.source_ids[:10])}")
                if st.button(f"Show originals", key=record.merge_id):
                    st.json(record.original_snapshots)


def page_quality_metrics():
    st.header("Quality & Observability Metrics")

    mg = load_graph()
    metrics = mg.get_metrics()

    # Graph stats
    col1, col2, col3 = st.columns(3)
    gs = metrics["graph_stats"]
    col1.metric("Entities", gs["nodes"])
    col2.metric("Claims", gs["edges"])
    col3.metric("Components", gs["connected_components"])

    # Evidence stats
    st.subheader("Evidence Quality")
    es = metrics["evidence_stats"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evidence Pointers", es["total_evidence_pointers"])
    col2.metric("Avg Evidence/Claim", f"{es['avg_evidence_per_claim']:.2f}")
    col3.metric("Claims with No Evidence", es["claims_with_no_evidence"])

    # Confidence stats
    st.subheader("Confidence Distribution")
    cs = metrics["confidence_stats"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Confidence", f"{cs['mean']:.2f}")
    col2.metric("Min", f"{cs['min']:.2f}")
    col3.metric("Max", f"{cs['max']:.2f}")

    # Temporal stats
    st.subheader("Temporal Coverage")
    ts = metrics["temporal_stats"]
    col1, col2 = st.columns(2)
    col1.metric("Current Claims", ts["current_claims"])
    col2.metric("Historical Claims", ts["historical_claims"])

    # Entity type distribution
    st.subheader("Entity Type Distribution")
    st.bar_chart(metrics["entity_type_distribution"])

    # Relation type distribution
    st.subheader("Relation Type Distribution")
    st.bar_chart(metrics["relation_type_distribution"])

    # Ingestion metrics
    st.subheader("Ingestion Metrics")
    st.json(metrics["ingestion_metrics"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()

    # Check data exists
    if not os.path.exists(GRAPH_PATH) or not os.path.exists(STORE_PATH):
        st.error(
            "Data files not found. Please run the pipeline first:\n"
            "1. `python fetch_corpus.py`\n"
            "2. `python extraction.py`\n"
            "3. `python deduplication.py`\n"
            "4. `python graph.py`"
        )
        return

    if page == "Graph View":
        page_graph_view()
    elif page == "Query / Chat":
        page_query_chat()
    elif page == "Evidence Inspector":
        page_evidence_inspector()
    elif page == "Merge Inspector":
        page_merge_inspector()
    elif page == "Quality Metrics":
        page_quality_metrics()


if __name__ == "__main__":
    main()
