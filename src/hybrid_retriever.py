import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import math

# LangChain and embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
# NEW IMPORT: Necessary to correctly pass metadata to BM25
from langchain_core.documents import Document 
from dotenv import load_dotenv

# LangSmith tracing
from langsmith import Client
from langsmith.wrappers import wrap_openai
import langsmith

load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NUST Policies Copilot"
# Make sure LANGCHAIN_API_KEY is set in your .env file

# Cross-encoder for reranking
from sentence_transformers import CrossEncoder

# ========= PATHS ========= #
PDF_JSONL = Path("./data/processed/pdf_chunks.jsonl")
HTML_JSONL = Path("./data/processed/html_chunks.jsonl")
INDEX_A_DIR = "./data/vectorstores/chroma_index_openAI"
INDEX_B_DIR = "./data/vectorstores/chroma_index_bge_m3"
GOLD_SET_PATH = Path("./eval/gold_set.jsonl")

# ========= LOAD CHUNKS ========= #
def load_chunks() -> List[Dict[str, Any]]:
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
def load_vector_indices():
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    openai_vs = Chroma(persist_directory=INDEX_A_DIR, embedding_function=openai_embeddings)
    bge_vs = Chroma(persist_directory=INDEX_B_DIR, embedding_function=bge_embeddings)
    
    print("Vector indices loaded.")
    return openai_vs, bge_vs

# ========= HYBRID SEARCH ========= #
@langsmith.traceable(name="hybrid_search")
def hybrid_search(query: str, chunks: List[Dict], openai_vs: Chroma, bge_vs: Chroma, k: int = 10) -> List[Dict]:
    lc_docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]

    # BM25 retrieval
    # texts = [c["text"] for c in chunks] # Removed: We now use lc_docs
    # bm25_retriever = BM25Retriever.from_texts(texts) # Removed: Loses metadata
    bm25_retriever = BM25Retriever.from_documents(lc_docs) # New: Preserves metadata
    bm25_retriever.k = 50
    bm25_docs = bm25_retriever.invoke(query)
    
    bm25_results = {}
    for i, doc in enumerate(bm25_docs):
        # Assign scores based on rank (higher rank = higher score)
        score = 1.0 / (i + 1)
        # doc.metadata is now correctly populated!
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

    # Fusion
    fused = []
    for text in all_texts:
        score = 0.35 * bm25_norm.get(text, 0) + 0.35 * openai_norm.get(text, 0) + 0.30 * bge_norm.get(text, 0)
        # Get metadata from any available source
        meta = {}
        if text in bm25_results:
            meta = bm25_results[text]["metadata"]
        elif text in openai_results:
            meta = openai_results[text]["metadata"]
        elif text in bge_results:
            meta = bge_results[text]["metadata"]
        
        fused.append({"text": text, "metadata": meta, "score": score})

    # Top 30
    fused.sort(key=lambda x: x["score"], reverse=True)
    fused = fused[:30]

    # Rerank
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, d["text"]] for d in fused]
    rerank_scores = cross_encoder.predict(pairs)
    for i, s in enumerate(rerank_scores):
        fused[i]["score"] = float(s)

    # Final top k
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]

# ========= EVALUATION HELPERS ========= #
def normalize_doc_id(doc_id: Any) -> str:
    """
    Normalize document ID for comparison while preserving important numbers.
    Only removes file extensions and compression suffixes.
    """
    if doc_id is None:
        return ""
    
    doc_id = str(doc_id)
    doc_id = doc_id.lower().strip()
    
    # Remove file extensions (only at the end)
    for ext in ['.pdf', '.html', '.docx', '.txt', '.doc', '.xlsx', '.pptx']:
        if doc_id.endswith(ext):
            doc_id = doc_id[:-len(ext)]
            break
    
    # Remove compression suffix (only at the end)
    if doc_id.endswith('_compressed'):
        doc_id = doc_id[:-len('_compressed')]
    
    return doc_id

def extract_doc_id_variants(doc_id: str) -> list:
    normalized = normalize_doc_id(doc_id)
    variants = [normalized]
    
    # Split by common separators
    parts = normalized.replace('_', '-').split('-')
    
    meaningful_parts = [p for p in parts if len(p) > 2 and p not in ['v', 'vol', 'ver']]
    
    if meaningful_parts:
        # Add the main name (first substantial part)
        if len(meaningful_parts) >= 1:
            variants.append(meaningful_parts[0])
        
        # Add name + year patterns (if year exists)
        for i, part in enumerate(meaningful_parts):
            # Check if it's a year (4 digits starting with 19 or 20)
            if part.isdigit() and len(part) == 4 and part.startswith(('19', '20')):
                if i > 0:
                    # Add "name-year" variant
                    variants.append(f"{meaningful_parts[i-1]}-{part}")
                    # Add "name year" variant (space separated)
                    variants.append(f"{meaningful_parts[i-1]} {part}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            unique_variants.append(v)
    
    return unique_variants

def is_relevant(retrieved: Dict, gold: Dict, debug: bool = False) -> bool:
    ret_doc = retrieved["metadata"].get("doc_id", "")
    ret_source = retrieved["metadata"].get("source", "")
    ret_sec = retrieved["metadata"].get("section", "")
    ret_text = retrieved["text"].lower()
    
    # Handle expected_doc_ids as list or string
    exp_docs_raw = gold.get("expected_doc_ids", [])
    if isinstance(exp_docs_raw, str):
        exp_docs = [exp_docs_raw]
    elif isinstance(exp_docs_raw, list):
        exp_docs = exp_docs_raw
    else:
        exp_docs = []
    
    # Handle expected_section
    exp_secs_raw = gold.get("expected_section", "")
    if isinstance(exp_secs_raw, str):
        exp_secs = [s.strip() for s in exp_secs_raw.split(",")]
    elif isinstance(exp_secs_raw, list):
        exp_secs = exp_secs_raw
    else:
        exp_secs = []
    
    # Handle expected_keywords
    exp_kws_raw = gold.get("expected_keywords", [])
    if isinstance(exp_kws_raw, list):
        exp_kws = [k.lower() for k in exp_kws_raw]
    elif isinstance(exp_kws_raw, str):
        exp_kws = [exp_kws_raw.lower()]
    else:
        exp_kws = []
    
    # Get all variants for retrieved doc_ids
    ret_doc_variants = extract_doc_id_variants(ret_doc) if ret_doc else []
    ret_source_variants = extract_doc_id_variants(ret_source) if ret_source else []
    all_ret_variants = set(ret_doc_variants + ret_source_variants)
    
    # Get all variants for expected doc_ids
    all_exp_variants = set()
    for exp_doc in exp_docs:
        all_exp_variants.update(extract_doc_id_variants(exp_doc))
    
    # Check doc match (exact or variant overlap)
    doc_match = bool(all_ret_variants & all_exp_variants)  # Set intersection
    
    # For debugging, track match type
    exact_match = normalize_doc_id(ret_doc) in [normalize_doc_id(d) for d in exp_docs]
    exact_match = exact_match or (normalize_doc_id(ret_source) in [normalize_doc_id(d) for d in exp_docs])
    
    # Check section match
    sec_match = any(
        sec.lower() in ret_sec.lower() or ret_sec.lower() in sec.lower()
        for sec in exp_secs if sec
    ) if ret_sec and exp_secs else False
    
    # Check keyword match
    kw_matches = [kw for kw in exp_kws if kw in ret_text]
    single_kw_match = len(kw_matches) > 0
    strong_kw_match = len(kw_matches) >= 2  # At least 2 keywords
    
    # Relevance logic with multiple strategies
    relevant = False
    match_reason = ""
    
    if doc_match and (sec_match or single_kw_match):
        relevant = True
        match_reason = f"{'exact' if exact_match else 'variant'}_doc + {'section' if sec_match else 'keyword'}"
    elif strong_kw_match:
        relevant = True
        match_reason = f"strong_keywords({len(kw_matches)})"
    
    if debug and relevant:
        print(f"  MATCH ({match_reason}):")
        print(f"    ret_doc: {ret_doc[:50]}")
        print(f"    ret_variants: {ret_doc_variants[:3]}")
        print(f"    exp_variants: {list(all_exp_variants)[:3]}")
        print(f"    section: {ret_sec[:30]}")
        print(f"    keywords matched: {kw_matches[:3]}")
    
    return relevant

def compute_total_relevant(gold: Dict, all_chunks: List[Dict]) -> int:
    """Count total relevant chunks for a gold standard query."""
    count = 0
    for c in all_chunks:
        if is_relevant({"text": c["text"], "metadata": c["metadata"]}, gold):
            count += 1
    return count

# ========= DIAGNOSTIC FUNCTIONS ========= #
def diagnose_metadata(chunks: List[Dict], sample_size: int = 5):
    """
    Print sample metadata to understand the structure.
    """
    print("\n" + "="*50)
    print("METADATA DIAGNOSIS")
    print("="*50)
    
    if not chunks:
        print("No chunks available!")
        return
    
    # Sample random chunks
    import random
    samples = random.sample(chunks, min(sample_size, len(chunks)))
    
    for i, chunk in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Metadata keys: {list(chunk['metadata'].keys())}")
        for key, value in chunk['metadata'].items():
            value_str = str(value)[:100]
            print(f"    {key}: {value_str}")
        print(f"  Text preview: {chunk['text'][:100]}...")
    print("="*50 + "\n")

def diagnose_query_retrieval(query: str, gold: Dict, chunks: List[Dict], 
                             openai_vs: Chroma, bge_vs: Chroma, k: int = 10):
    print(f"\n{'='*70}")
    print(f"DIAGNOSING QUERY: {query}")
    print(f"{'='*70}")
    
    print(f"Expected doc_ids: {gold.get('expected_doc_ids')}")
    print(f"Expected section: {gold.get('expected_section')}")
    print(f"Expected keywords: {gold.get('expected_keywords')[:3] if isinstance(gold.get('expected_keywords'), list) else gold.get('expected_keywords')}")
    
    # Retrieve
    retrieved = hybrid_search(query, chunks, openai_vs, bge_vs, k=k)
    
    print(f"\nTop {k} Retrieved Results:")
    for i, result in enumerate(retrieved, 1):
        is_rel = is_relevant(result, gold, debug=False)
        rel_marker = "✓" if is_rel else "✗"
        
        doc_id = result['metadata'].get('doc_id', 'N/A')[:40]
        source = result['metadata'].get('source', 'N/A')[:40]
        section = result['metadata'].get('section', 'N/A')[:40]
        
        print(f"\n  {rel_marker} Result {i} (score: {result['score']:.4f}):")
        print(f"      doc_id: {doc_id}")
        print(f"      source: {source}")
        print(f"      section: {section}")
        print(f"      text: {result['text'][:150]}...")
        
        # Check why it's relevant/not relevant
        if is_rel:
            is_relevant(result, gold, debug=True)
    
    # Count total relevant
    total_rel = compute_total_relevant(gold, chunks)
    print(f"\n  Total relevant chunks in corpus: {total_rel}")
    print(f"{'='*70}\n")

# ========= EVALUATION ========= #
def load_gold_set(gold_path: Path) -> List[Dict]:
    """
    Load gold set with support for both proper JSONL and pretty-printed JSON.
    """
    with gold_path.open('r', encoding='utf-8') as f:
        content = f.read()
    
    golds = []
    
    # First, try to parse as proper JSONL (one object per line)
    lines = content.strip().split('\n')
    if lines:
        try:
            # Try parsing first line
            json.loads(lines[0])
            # If successful, parse all lines
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    golds.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            
            if golds:
                return golds
        except json.JSONDecodeError:
            pass
    
    # If JSONL parsing failed, try parsing as multi-line JSON objects
    print("Attempting to parse multi-line JSON format...")
    current_obj = ""
    brace_count = 0
    
    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            
            if brace_count == 0 and current_obj.strip():
                try:
                    obj = json.loads(current_obj.strip())
                    golds.append(obj)
                    current_obj = ""
                except json.JSONDecodeError:
                    pass
    
    return golds

@langsmith.traceable(name="evaluate_gold_set")
def evaluate_gold_set(gold_path: Path, chunks: List[Dict], openai_vs: Chroma, bge_vs: Chroma) -> Dict:
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold set not found at {gold_path}")
    
    golds = load_gold_set(gold_path)
    print(f"Loaded {len(golds)} gold queries.")
    
    if not golds:
        print("No valid gold queries found!")
        return {"recall": {}, "ndcg": {}}
    
    metrics = {
        "recall": {1: [], 3: [], 5: [], 10: []},
        "ndcg": {3: [], 5: [], 10: []}
    }
    
    for gold in tqdm(golds, desc="Evaluating"):
        query = gold.get("query")
        if not query:
            print(f"Warning: Gold item missing query: {gold.get('id', 'unknown')}")
            continue
        
        retrieved = hybrid_search(query, chunks, openai_vs, bge_vs, k=10)
        
        total_rel = compute_total_relevant(gold, chunks)
        if total_rel == 0:
            print(f"Warning: No relevant chunks found for query '{query[:50]}...'")
            # Skip this query instead of adding zeros
            continue
        
        rel_scores = [1 if is_relevant(r, gold) else 0 for r in retrieved]
        
        # Recall@k
        for kk in [1, 3, 5, 10]:
            rel_in_k = sum(rel_scores[:kk])
            metrics["recall"][kk].append(rel_in_k / total_rel)
        
        # nDCG@k
        for kk in [3, 5, 10]:
            dcg = sum(rs / math.log2(i+2) for i, rs in enumerate(rel_scores[:kk]))
            idcg = sum(1 / math.log2(i+2) for i in range(min(kk, total_rel)))
            metrics["ndcg"][kk].append(dcg / idcg if idcg > 0 else 0.0)
    
    # Check if we have any results
    if not metrics["recall"][1]:
        print("Warning: No queries with relevant documents were evaluated!")
        return {"recall": {k: 0.0 for k in [1,3,5,10]}, "ndcg": {k: 0.0 for k in [3,5,10]}}
    
    # Averages
    avg_recall = {k: np.mean(v) if v else 0.0 for k, v in metrics["recall"].items()}
    avg_ndcg = {k: np.mean(v) if v else 0.0 for k, v in metrics["ndcg"].items()}
    
    # Print
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Recall@1:  {avg_recall[1]:.4f}")
    print(f"Recall@3:  {avg_recall[3]:.4f}")
    print(f"Recall@5:  {avg_recall[5]:.4f}")
    print(f"Recall@10: {avg_recall[10]:.4f}")
    print(f"nDCG@3:    {avg_ndcg[3]:.4f}")
    print(f"nDCG@5:    {avg_ndcg[5]:.4f}")
    print(f"nDCG@10:   {avg_ndcg[10]:.4f}")
    print("="*50)
    
    return {"recall": avg_recall, "ndcg": avg_ndcg}

# ========= MAIN ========= #
if __name__ == "__main__":
    print("Starting hybrid retriever evaluation...")
    chunks = load_chunks()
    
    if not chunks:
        print("Error: No chunks loaded. Please check your data files.")
        exit(1)
    
    openai_vs, bge_vs = load_vector_indices()
    
    # Diagnostic mode: Check if first argument is "diagnose"
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        print("\n🔍 RUNNING DIAGNOSTIC MODE 🔍\n")
        
        # Show metadata structure
        diagnose_metadata(chunks, sample_size=3)
        
        # Load one gold query and diagnose
        golds = load_gold_set(GOLD_SET_PATH)
        if golds:
            print(f"Testing with first gold query...")
            diagnose_query_retrieval(
                golds[0]['query'], 
                golds[0], 
                chunks, 
                openai_vs, 
                bge_vs, 
                k=10
            )
            
            if len(golds) > 1:
                print(f"\nTesting with a query that had no relevant results...")
                # Find a query that might have issues
                for gold in golds[15:20]:  # Check middle queries
                    print(f"Testing: {gold['query'][:60]}...")
                    diagnose_query_retrieval(
                        gold['query'], 
                        gold, 
                        chunks, 
                        openai_vs, 
                        bge_vs, 
                        k=5
                    )
                    break
        
        print("\n✓ Diagnostic complete. Run without 'diagnose' argument for full evaluation.\n")
    else:
        # Normal evaluation mode
        results = evaluate_gold_set(GOLD_SET_PATH, chunks, openai_vs, bge_vs)
        
        # Save results
        results_path = Path("./eval/evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")