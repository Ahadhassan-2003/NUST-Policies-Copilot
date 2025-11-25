# evaluate_retriever.py

import json
import os
import math
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from dotenv import load_dotenv
from langsmith import traceable

# Import from hybrid_retriever
from hybrid_retriever import initialize_retriever, hybrid_search, load_chunks

load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NUST Policies Copilot"

# ========= PATHS ========= #
GOLD_SET_PATH = Path("./eval/gold_set.jsonl")

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
    """Extract multiple variants of a document ID for flexible matching."""
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

@traceable(name="is_relevant")
def is_relevant(retrieved: Dict, gold: Dict, debug: bool = False) -> bool:
    """
    Check if a retrieved result is relevant to the gold standard.
    
    Args:
        retrieved: Retrieved result dict with text and metadata
        gold: Gold standard query dict with expected_doc_ids, expected_section, expected_keywords
        debug: If True, print debug information
    
    Returns:
        bool: True if relevant, False otherwise
    """
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

@traceable(name="compute_total_relevant")
def compute_total_relevant(gold: Dict, all_chunks: List[Dict]) -> int:
    """Count total relevant chunks for a gold standard query."""
    count = 0
    for c in all_chunks:
        if is_relevant({"text": c["text"], "metadata": c["metadata"]}, gold):
            count += 1
    return count

# ========= DIAGNOSTIC FUNCTIONS ========= #

def diagnose_metadata(chunks: List[Dict], sample_size: int = 5):
    """Print sample metadata to understand the structure."""
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

@traceable(name="diagnose_query_retrieval")
def diagnose_query_retrieval(
    query: str, 
    gold: Dict, 
    chunks: List[Dict], 
    bm25_retriever,
    openai_vs, 
    bge_vs, 
    k: int = 10
):
    """Diagnose retrieval for a specific query."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING QUERY: {query}")
    print(f"{'='*70}")
    
    print(f"Expected doc_ids: {gold.get('expected_doc_ids')}")
    print(f"Expected section: {gold.get('expected_section')}")
    print(f"Expected keywords: {gold.get('expected_keywords')[:3] if isinstance(gold.get('expected_keywords'), list) else gold.get('expected_keywords')}")
    
    # Retrieve
    retrieved = hybrid_search(query, bm25_retriever, openai_vs, bge_vs, k=k)
    
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

@traceable(name="evaluate_gold_set")
def evaluate_gold_set(
    gold_path: Path, 
    chunks: List[Dict], 
    bm25_retriever,
    openai_vs, 
    bge_vs
) -> Dict:
    """
    Evaluate retriever performance on gold set.
    
    Returns:
        Dict with recall and nDCG metrics
    """
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
        
        retrieved = hybrid_search(query, bm25_retriever, openai_vs, bge_vs, k=10)
        
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
    import sys
    
    print("Starting retriever evaluation...")
    
    # Load chunks and initialize retriever
    chunks = load_chunks()
    if not chunks:
        print("Error: No chunks loaded. Please check your data files.")
        exit(1)
    
    bm25_retriever, openai_vs, bge_vs = initialize_retriever()
    
    # Diagnostic mode: Check if first argument is "diagnose"
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
                bm25_retriever,
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
                        bm25_retriever, 
                        openai_vs, 
                        bge_vs, 
                        k=5
                    )
                    break
        
        print("\n✓ Diagnostic complete. Run without 'diagnose' argument for full evaluation.\n")
    else:
        # Normal evaluation mode
        results = evaluate_gold_set(GOLD_SET_PATH, chunks, bm25_retriever, openai_vs, bge_vs)
        
        # Save results
        results_path = Path("./eval/evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")