from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
import networkx as nx

load_dotenv()

app = FastAPI(title="Fusion RAG Backend", version="1.0")

FAISS_INDEX_PATH = "/app/faiss_index"
TEMP_DIR = "/app/temp_docs"
os.makedirs(TEMP_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. FAISS Pipeline
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
            
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(docs, embeddings)
        faiss_db.save_local(FAISS_INDEX_PATH)
        
        # 2. Knowledge Graph Pipeline
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        graph_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = graph_transformer.convert_to_graph_documents(docs[:15])
        
        neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        neo4j_graph.add_graph_documents(graph_documents, include_source=True)
        
        nx_graph = nx.DiGraph()
        total_nodes = 0
        total_rels = 0
        for gdoc in graph_documents:
            for node in gdoc.nodes:
                nx_graph.add_node(node.id, type=node.type)
                total_nodes += 1
            for rel in gdoc.relationships:
                nx_graph.add_edge(rel.source.id, rel.target.id, type=rel.type)
                total_rels += 1

        return {
            "status": "success",
            "total_chunks": len(docs),
            "total_nodes": total_nodes,
            "total_rels": total_rels
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_data(payload: QueryRequest):
    query = payload.query
    results = {"faiss": [], "neo4j": []}
    
    # 1. Query FAISS
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            embeddings = OpenAIEmbeddings()
            loaded_faiss = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            docs_and_scores = loaded_faiss.similarity_search_with_score(query, k=3)
            for doc, score in docs_and_scores:
                results["faiss"].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
    except Exception as e:
        results["faiss_error"] = str(e)
        
    # 2. Query Neo4j
    try:
        neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        cypher_query = """
        MATCH (s)-[r]->(t)
        RETURN s.id AS source, type(r) AS relationship, t.id AS target
        LIMIT 10
        """
        graph_results = neo4j_graph.query(cypher_query)
        for record in graph_results:
            results["neo4j"].append({
                "source": record["source"],
                "relationship": record["relationship"],
                "target": record["target"]
            })
    except Exception as e:
        results["neo4j_error"] = str(e)
        
    return results