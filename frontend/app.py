import streamlit as st
import requests

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="Enterprise Fusion RAG Studio", page_icon="⚡", layout="wide")

st.markdown('<div class="main-header" style="font-size:2.25rem; font-weight:700;">⚡ Hybrid Fusion RAG Studio (Decoupled)</div>', unsafe_allow_html=True)
st.markdown("Integrate unstructured knowledge into dual-engine storage via **FastAPI Backend**.", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 1. Knowledge Base Ingestion")
    uploaded_file = st.file_uploader("Upload reference documentation (PDF or TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Index Knowledge Base", type="primary"):
            with st.spinner("Processing document through backend pipeline..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{BACKEND_URL}/process", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["processed"] = True
                    st.session_state["metrics"] = data
                    st.success("Successfully processed document!")
                else:
                    st.error(f"Error: {response.text}")

with col2:
    st.markdown("### 📊 Engine Summary & Metrics")
    if st.session_state.get("processed"):
        m = st.session_state["metrics"]
        st.markdown(f"* **Indexed Chunks:** `{m.get('total_chunks', 0)}`")
        st.markdown(f"* **Extracted Nodes:** `{m.get('total_nodes', 0)}`")
        st.markdown(f"* **Extracted Relationships:** `{m.get('total_rels', 0)}`")
    else:
        st.info("Upload and process a document to view metrics.")

st.divider()
st.markdown("### 🔎 2. Hybrid Fusion Retrieval")
user_query = st.text_input("Ask a question based on your knowledge base:")

if user_query:
    with st.spinner("Querying vector store and knowledge graph..."):
        res = requests.post(f"{BACKEND_URL}/retrieve", json={"query": user_query})
        if res.status_code == 200:
            data = res.json()
            col_v, col_g = st.columns(2, gap="large")
            
            with col_v:
                st.markdown("#### 🔍 FAISS Vector Store Results")
                faiss_res = data.get("faiss", [])
                if faiss_res:
                    for i, item in enumerate(faiss_res):
                        with st.expander(f"Result {i+1} | Distance: {item['score']:.4f}"):
                            st.write(item["content"])
                else:
                    st.info("No semantic matches found.")
                    
            with col_g:
                st.markdown("#### 🕸️ Knowledge Graph Results (Neo4j)")
                neo4j_res = data.get("neo4j", [])
                if neo4j_res:
                    for i, item in enumerate(neo4j_res):
                        with st.expander(f"Path {i+1}: {item['source']} ➔ {item['relationship']} ➔ {item['target']}"):
                            st.write(f"Source: {item['source']}")
                            st.write(f"Relation: {item['relationship']}")
                            st.write(f"Target: {item['target']}")
                else:
                    st.info("No relational paths found.")