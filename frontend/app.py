import streamlit as st
import requests

st.set_page_config(
    page_title="Hybrid Fusion RAG Studio", 
    page_icon="⚡", 
    layout="wide"
)

st.title("⚡ Hybrid Fusion RAG Studio (Decoupled)")
st.caption("Integrate unstructured knowledge into dual-engine storage via FastAPI Backend.")

# Update this URL if needed or map via docker-compose network service name
BACKEND_URL = "http://backend:8000"

# ---------------------------------------------------------
# Section 1: Knowledge Base Ingestion
# ---------------------------------------------------------
st.header("📄 1. Knowledge Base Ingestion")
uploaded_file = st.file_uploader(
    "Upload reference documentation (PDF or TXT)", 
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    if st.button("🚀 Process & Index Knowledge Base"):
        with st.spinner("Parsing document and populating FAISS & Neo4j..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{BACKEND_URL}/ingest", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success("Document successfully processed!")
                    
                    st.subheader("📊 Engine Summary & Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Indexed Chunks", data.get("chunks", 0))
                    col2.metric("Extracted Nodes", data.get("nodes", 0))
                    col3.metric("Extracted Relationships", data.get("relationships", 0))
                else:
                    st.error(f"Ingestion failed: {res.text}")
            except Exception as e:
                st.error(f"Backend connection error: {e}")

st.divider()

# ---------------------------------------------------------
# Section 2: Hybrid Fusion Retrieval & AI Generation
# ---------------------------------------------------------
st.header("🔎 2. Hybrid Fusion Retrieval")

# Using st.form prevents premature page re-runs and manages state cleanly
with st.form(key="search_form"):
    user_query = st.text_input(
        "Ask a question based on your knowledge base:", 
        placeholder="e.g., What databases are used for relational data and caching?"
    )
    submit_button = st.form_submit_button(label="🔍 Search & Generate Answer")

if submit_button and user_query:
    with st.spinner("Querying FAISS, Neo4j, and synthesizing AI answer..."):
        try:
            response = requests.post(f"{BACKEND_URL}/generate-answer", params={"query": user_query})
            if response.status_code == 200:
                data = response.json()
                
                # 1. Final AI Synthesized Answer
                st.markdown("### 💡 AI Synthesized Answer")
                st.success(data.get("answer", "No answer generated."))
                
                st.divider()
                
                col_vector, col_graph = st.columns(2)
                
                # 2. FAISS Vector Store Results
                with col_vector:
                    st.subheader("🔍 FAISS Vector Store Results")
                    vector_data = data.get("vector_results", [])
                    if vector_data:
                        for idx, item in enumerate(vector_data, 1):
                            chunk_text = item.get("page_content", "")
                            score = item.get("score", "N/A")
                            with st.expander(f"Result {idx} | Score: {score}", expanded=(idx == 1)):
                                st.write(chunk_text)
                    else:
                        st.info("No relevant vector chunks retrieved.")
                        
                # 3. Knowledge Graph Results (Neo4j)
                with col_graph:
                    st.subheader("🕸️ Knowledge Graph Results (Neo4j)")
                    graph_data = data.get("graph_results", [])
                    if graph_data:
                        for idx, path in enumerate(graph_data, 1):
                            source = path.get("Source", "Unknown")
                            relation = path.get("Relation", "RELATED_TO")
                            target = path.get("Target", "Unknown")
                            st.info(f"**Path {idx}:** `{source}` ➔ **{relation}** ➔ `{target}`")
                    else:
                        st.warning("No dynamic relational paths found for this query.")
            else:
                st.error(f"Search failed with status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to backend service: {e}")