import os
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain and Vector/Graph Libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

app = FastAPI(
    title="Hybrid Fusion RAG Backend",
    description="Decoupled FastAPI backend handling FAISS vector storage and Neo4j Knowledge Graph extraction.",
    version="1.0.0"
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "c06e9654")

FAISS_INDEX_DIR = "/app/faiss_index"
TEMP_DOCS_DIR = "/app/temp_docs"

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(TEMP_DOCS_DIR, exist_ok=True)

# Initialize OpenAI Models
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# Initialize Direct Neo4j Driver
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def get_faiss_vector_store():
    """Load existing FAISS index or return None if not initialized."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(
            FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    return None


@app.get("/health")
def health_check():
    """Verify backend and Neo4j connection status."""
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 AS status")
            status = result.single()["status"]
            neo4j_ok = (status == 1)
    except Exception as e:
        neo4j_ok = False
        print(f"Healthcheck Neo4j Error: {e}")

    return {
        "status": "online",
        "neo4j_connected": neo4j_ok,
        "faiss_index_exists": os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    }


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document (PDF or TXT):
    1. Chunk text and add to local FAISS index.
    2. Extract entities/relationships via LLM and write to Neo4j.
    """
    file_path = os.path.join(TEMP_DOCS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Load Document
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")

        docs = loader.load()

        # 2. Split Document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        # 3. Save / Update FAISS Index
        vector_store = get_faiss_vector_store()
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)

        vector_store.save_local(FAISS_INDEX_DIR)

        # 4. LLM Graph Extraction for Knowledge Graph
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)

        # Save to Neo4j via LangChain Neo4jGraph wrapper
        graph_db = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        graph_db.add_graph_documents(graph_documents)

        nodes_count = sum(len(gd.nodes) for gd in graph_documents)
        relationships_count = sum(len(gd.relationships) for gd in graph_documents)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "nodes": nodes_count,
            "relationships": relationships_count,
            "message": "Knowledge Base updated successfully!"
        }

    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/retrieve")
def retrieve(query: str = Query(..., description="User search prompt")) -> Dict[str, Any]:
    # 1. FAISS Vector Search
    vector_store = get_faiss_vector_store()
    vector_results = []
    
    if vector_store:
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
            for doc, score in results:
                vector_results.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(float(score), 4)
                })
        except Exception as e:
            print(f"FAISS retrieval error: {e}")

    # 2. Dynamic Neo4j Graph Retrieval (Strict keyword match only)
    search_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
    
    cypher_query = """
    MATCH (n)-[r]->(m)
    WHERE any(term IN $search_terms WHERE toLower(coalesce(n.id, n.name, '')) CONTAINS term
       OR toLower(coalesce(m.id, m.name, '')) CONTAINS term
       OR toLower(type(r)) CONTAINS term)
    RETURN coalesce(n.id, n.name) AS Source, type(r) AS Relation, coalesce(m.id, m.name) AS Target
    LIMIT 12
    """

    graph_results = []
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            db_res = session.run(cypher_query, search_terms=search_terms)
            for record in db_res:
                graph_results.append({
                    "Source": record["Source"],
                    "Relation": record["Relation"],
                    "Target": record["Target"]
                })
    except Exception as e:
        print(f"Neo4j dynamic retrieval error: {e}")

    return {
        "vector_results": vector_results,
        "graph_results": graph_results
    }

@app.post("/generate-answer")
def generate_answer(query: str = Query(..., description="User question")):
    """
    Combines hybrid retrieval results (FAISS chunks + Neo4j graph paths)
    and uses OpenAI to synthesize a precise natural language response.
    """
    retrieval_data = retrieve(query)
    
    context_texts = [item["page_content"] for item in retrieval_data["vector_results"]]
    graph_texts = [f"{p['Source']} - [{p['Relation']}] -> {p['Target']}" for p in retrieval_data["graph_results"]]
    
    combined_context = "\n---\n".join(context_texts + graph_texts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical assistant. Answer the user's question accurately using ONLY the provided context from vector chunks and knowledge graph paths."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": combined_context, "question": query})
    
    return {
        "answer": answer,
        "vector_results": retrieval_data["vector_results"],
        "graph_results": retrieval_data["graph_results"]
    }