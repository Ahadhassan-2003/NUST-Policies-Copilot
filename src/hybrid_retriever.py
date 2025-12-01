# hybrid_retriever.py

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# LangChain and embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document 
from dotenv import load_dotenv

# LangSmith tracing
from langsmith import traceable
import os

# Cross-encoder for reranking
from sentence_transformers import CrossEncoder

load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NUST Policies Copilot"

# ========= PATHS ========= #
PDF_JSONL = Path("./data/processed/pdf_chunks.jsonl")
HTML_JSONL = Path("./data/processed/html_chunks.jsonl")
INDEX_A_DIR = "./data/vectorstores/chroma_index_openAI"
INDEX_B_DIR = "./data/vectorstores/chroma_index_bge_m3"
BM25_INDEX_PATH = Path("./data/vectorstores/bm25_index.pkl")

# ========= LOAD CHUNKS ========= #
@traceable(name="load_chunks")
def load_chunks() -> List[Dict[str, Any]]:
    """Load all chunks from PDF and HTML JSONL files."""
    chunks = []
    
    # Helper to load from a file
    def load_from_file(file_path: Path, source_type: str):
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist.")
            return []
        with file_path.open("r", encoding="utf-8") as f:
            loaded = []
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get("chunk_text") or item.get("text", "")
                    if text.strip():
                        # Preserve all other keys as metadata
                        metadata = {k: v for k, v in item.items() if k not in ["chunk_text", "text"]}
                        metadata["source_type"] = source_type
                        loaded.append({"text": text, "metadata": metadata})
                except json.JSONDecodeError:
                    pass  # Skip invalid lines
            return loaded
    
    # Load PDF and HTML
    chunks.extend(load_from_file(PDF_JSONL, "pdf"))
    chunks.extend(load_from_file(HTML_JSONL, "html"))
    
    print(f"Loaded {len(chunks)} chunks.")
    return chunks

# ========= LOAD VECTOR INDICES ========= #
@traceable(name="load_vector_indices")
def load_vector_indices():
    """Load OpenAI and BGE vector stores from disk."""
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    openai_vs = Chroma(persist_directory=INDEX_A_DIR, embedding_function=openai_embeddings)
    bge_vs = Chroma(persist_directory=INDEX_B_DIR, embedding_function=bge_embeddings)
    
    print("Vector indices loaded.")
    return openai_vs, bge_vs

# ========= LOAD BM25 INDEX ========= #
@traceable(name="load_bm25_index")
def load_bm25_index():
    """Load pre-built BM25 index from disk."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Please run create_embeddings.py first.")
    
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    print("BM25 index loaded.")
    return bm25_retriever

# ========= HYBRID SEARCH ========= #
@traceable(name="hybrid_search")
def hybrid_search(
    query: str, 
    bm25_retriever: BM25Retriever,
    openai_vs: Chroma, 
    bge_vs: Chroma, 
    k: int = 10,
    bm25_weight: float = 0.3,
    openai_weight: float = 0.35,
    bge_weight: float = 0.35
) -> List[Dict]:
    """
    Perform hybrid search combining BM25, OpenAI embeddings, and BGE embeddings.
    
    Args:
        query: Search query string
        bm25_retriever: Pre-loaded BM25 retriever
        openai_vs: OpenAI vector store
        bge_vs: BGE vector store
        k: Number of final results to return
        bm25_weight: Weight for BM25 scores (default 0.35)
        openai_weight: Weight for OpenAI scores (default 0.35)
        bge_weight: Weight for BGE scores (default 0.30)
    
    Returns:
        List of dictionaries with text, metadata, and score
    """
    
    # BM25 retrieval
    bm25_retriever.k = 50
    bm25_docs = bm25_retriever.invoke(query)
    
    bm25_results = {}
    for i, doc in enumerate(bm25_docs):
        # Assign scores based on rank (higher rank = higher score)
        score = 1.0 / (i + 1)
        bm25_results[doc.page_content] = {
            "score": score, 
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        }

    # OpenAI dense retrieval
    openai_similarities = openai_vs.similarity_search_with_score(query, k=50)
    openai_results = {}
    for doc, dist in openai_similarities:
        sim = 1 / (1 + dist)  # Convert distance to similarity
        openai_results[doc.page_content] = {"score": sim, "metadata": doc.metadata}

    # BGE dense retrieval
    bge_similarities = bge_vs.similarity_search_with_score(query, k=50)
    bge_results = {}
    for doc, dist in bge_similarities:
        sim = 1 / (1 + dist)  # Convert distance to similarity
        bge_results[doc.page_content] = {"score": sim, "metadata": doc.metadata}

    # Union texts
    all_texts = set(bm25_results.keys()) | set(openai_results.keys()) | set(bge_results.keys())

    # Normalize scores (min-max, higher better)
    def normalize(d: Dict) -> Dict:
        if not d:
            return {}
        scores = [v["score"] for v in d.values()]
        min_s, max_s = min(scores), max(scores)
        if min_s == max_s:
            return {t: 0.5 for t in d}
        return {t: (v["score"] - min_s) / (max_s - min_s) for t, v in d.items()}

    bm25_norm = normalize(bm25_results)
    openai_norm = normalize(openai_results)
    bge_norm = normalize(bge_results)

    # Fusion with configurable weights
    fused = []
    for text in all_texts:
        score = (bm25_weight * bm25_norm.get(text, 0) + 
                openai_weight * openai_norm.get(text, 0) + 
                bge_weight * bge_norm.get(text, 0))
        
        # Get metadata from any available source
        meta = {}
        if text in bm25_results:
            meta = bm25_results[text]["metadata"]
        elif text in openai_results:
            meta = openai_results[text]["metadata"]
        elif text in bge_results:
            meta = bge_results[text]["metadata"]
        
        fused.append({"text": text, "metadata": meta, "score": score})

    # Top 30 before reranking
    fused.sort(key=lambda x: x["score"], reverse=True)
    fused = fused[:30]

    # Rerank with cross-encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, d["text"]] for d in fused]
    rerank_scores = cross_encoder.predict(pairs)
    for i, s in enumerate(rerank_scores):
        fused[i]["score"] = float(s)

    # Final top k
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]

# ========= INITIALIZATION HELPER ========= #
@traceable(name="initialize_retriever")
def initialize_retriever():
    """
    Initialize and return all components needed for hybrid search.
    
    Returns:
        tuple: (bm25_retriever, openai_vs, bge_vs)
    """
    print("Initializing hybrid retriever components...")
    
    bm25_retriever = load_bm25_index()
    openai_vs, bge_vs = load_vector_indices()
    
    print("✓ All retriever components initialized successfully\n")
    
    return bm25_retriever, openai_vs, bge_vs

# ========= MAIN (FOR TESTING) ========= #
if __name__ == "__main__":
    print("Testing hybrid retriever...")
    
    # Initialize
    bm25_retriever, openai_vs, bge_vs = initialize_retriever()
    
    # Test queries
    test_queries = [
        "What are the hostel rules at NUST?",
        "Tell me about the BSCS 2025 curriculum",
        "What scholarships are available?",
        "How much are the tuition fees?"
    ]
    
    print("\n" + "="*70)
    print("RUNNING TEST QUERIES")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        results = hybrid_search(
            query=query,
            bm25_retriever=bm25_retriever,
            openai_vs=openai_vs,
            bge_vs=bge_vs,
            k=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']:.4f}):")
            print(f"    Doc ID: {result['metadata'].get('doc_id', 'N/A')}")
            print(f"    Section: {result['metadata'].get('section', 'N/A')[:50]}")
            print(f"    Text: {result['text'][:150]}...")
    
    print("\n" + "="*70)
    print("✓ Test complete!")