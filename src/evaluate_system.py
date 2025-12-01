# evaluate_system.py

import json
import os
import sys
import math
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from dotenv import load_dotenv
from langsmith import traceable

# Import from other modules
from hybrid_retriever import initialize_retriever, hybrid_search, load_chunks
from generation_module import generate_answer_with_retrieval
from evaluation_metrics import (
    evaluate_single_query,
    aggregate_metrics,
    print_evaluation_summary,
    track_latency
)

load_dotenv()

# ========= PATHS ========= #
GOLD_SET_PATH = Path("./eval/final_gold_set.jsonl")
RESULTS_PATH = Path("./eval/results")

# ========= HELPER FUNCTIONS FROM OLD EVALUATOR ========= #

def normalize_doc_id(doc_id: Any) -> str:
    """Normalize document ID for comparison."""
    if doc_id is None:
        return ""
    doc_id = str(doc_id).lower().strip()
    for ext in ['.pdf', '.html', '.docx', '.txt']:
        if doc_id.endswith(ext):
            doc_id = doc_id[:-len(ext)]
            break
    if doc_id.endswith('_compressed'):
        doc_id = doc_id[:-len('_compressed')]
    return doc_id

def extract_doc_id_variants(doc_id: str) -> list:
    """Extract multiple variants of a document ID for flexible matching."""
    normalized = normalize_doc_id(doc_id)
    variants = [normalized]
    parts = normalized.replace('_', '-').split('-')
    meaningful_parts = [p for p in parts if len(p) > 2 and p not in ['v', 'vol', 'ver']]
    
    if meaningful_parts:
        if len(meaningful_parts) >= 1:
            variants.append(meaningful_parts[0])
        for i, part in enumerate(meaningful_parts):
            if part.isdigit() and len(part) == 4 and part.startswith(('19', '20')):
                if i > 0:
                    variants.append(f"{meaningful_parts[i-1]}-{part}")
                    variants.append(f"{meaningful_parts[i-1]} {part}")
    
    seen = set()
    unique_variants = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants

@traceable(name="is_relevant")
def is_relevant(retrieved: Dict, gold: Dict) -> bool:
    """Check if a retrieved result is relevant (for retrieval metrics)."""
    ret_doc = retrieved["metadata"].get("doc_id", "")
    ret_source = retrieved["metadata"].get("source", "")
    ret_sec = retrieved["metadata"].get("section", "")
    ret_text = retrieved["text"].lower()
    
    exp_docs_raw = gold.get("expected_doc_ids", [])
    if isinstance(exp_docs_raw, str):
        exp_docs = [exp_docs_raw]
    elif isinstance(exp_docs_raw, list):
        exp_docs = exp_docs_raw
    else:
        exp_docs = []
    
    exp_secs_raw = gold.get("expected_section", "")
    if isinstance(exp_secs_raw, str):
        exp_secs = [s.strip() for s in exp_secs_raw.split(",")]
    elif isinstance(exp_secs_raw, list):
        exp_secs = exp_secs_raw
    else:
        exp_secs = []
    
    exp_kws_raw = gold.get("expected_keywords", [])
    if isinstance(exp_kws_raw, list):
        exp_kws = [k.lower() for k in exp_kws_raw]
    elif isinstance(exp_kws_raw, str):
        exp_kws = [exp_kws_raw.lower()]
    else:
        exp_kws = []
    
    ret_doc_variants = extract_doc_id_variants(ret_doc) if ret_doc else []
    ret_source_variants = extract_doc_id_variants(ret_source) if ret_source else []
    all_ret_variants = set(ret_doc_variants + ret_source_variants)
    
    all_exp_variants = set()
    for exp_doc in exp_docs:
        all_exp_variants.update(extract_doc_id_variants(exp_doc))
    
    doc_match = bool(all_ret_variants & all_exp_variants)
    
    sec_match = any(
        sec.lower() in ret_sec.lower() or ret_sec.lower() in sec.lower()
        for sec in exp_secs if sec
    ) if ret_sec and exp_secs else False
    
    kw_matches = [kw for kw in exp_kws if kw in ret_text]
    single_kw_match = len(kw_matches) > 0
    strong_kw_match = len(kw_matches) >= 2
    
    if doc_match and (sec_match or single_kw_match):
        return True
    elif strong_kw_match:
        return True
    
    return False

# ========= GOLD SET LOADING ========= #

def load_gold_set(gold_path: Path) -> List[Dict]:
    """Load enhanced gold set with all new fields."""
    with gold_path.open('r', encoding='utf-8') as f:
        content = f.read()
    
    golds = []
    lines = content.strip().split('\n')
    
    if lines:
        try:
            json.loads(lines[0])
            for line in lines:
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
    
    # Multi-line JSON fallback
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

# ========= RETRIEVAL EVALUATION (RENAMED: Hit Rate -> Recall) ========= #

@traceable(name="evaluate_retrieval_metrics")
def evaluate_retrieval_metrics(
    golds: List[Dict],
    chunks: List[Dict],
    bm25_retriever,
    openai_vs,
    bge_vs,
    k: int = 10
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance using Recall@k and nDCG@k.
    
    Recall@k (formerly Hit Rate@k):
    - Binary metric: 1 if at least one relevant doc in top k, 0 otherwise
    - More realistic for RAG than traditional recall (relevant_in_k / total_relevant_in_corpus)
    
    nDCG@k:
    - Measures ranking quality
    - Higher scores = relevant docs ranked higher
    """
    metrics = {
        "recall": {1: [], 3: [], 5: [], 10: []},  # RENAMED from "hit_rate"
        "ndcg": {3: [], 5: [], 10: []}
    }
    
    for gold in tqdm(golds, desc="Evaluating Retrieval"):
        query = gold.get("query")
        if not query:
            continue
        
        retrieved = hybrid_search(query, bm25_retriever, openai_vs, bge_vs, k=k)
        
        # Get relevance scores for each retrieved doc
        rel_scores = [1 if is_relevant(r, gold) else 0 for r in retrieved]
        
        # Recall@k: Did we find at least one relevant doc in top k?
        for kk in [1, 3, 5, 10]:
            has_relevant = any(rel_scores[:kk])
            metrics["recall"][kk].append(1.0 if has_relevant else 0.0)
        
        # nDCG@k: How well are relevant docs ranked?
        # Only compute if we have at least one relevant doc
        if any(rel_scores):
            for kk in [3, 5, 10]:
                # DCG: sum of (relevance / log2(rank+1))
                dcg = sum(rs / math.log2(i+2) for i, rs in enumerate(rel_scores[:kk]))
                
                # IDCG: perfect ranking (all relevant docs first)
                num_relevant_in_k = sum(rel_scores[:kk])
                idcg = sum(1 / math.log2(i+2) for i in range(num_relevant_in_k))
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                metrics["ndcg"][kk].append(ndcg)
        else:
            # No relevant docs found at all
            for kk in [3, 5, 10]:
                metrics["ndcg"][kk].append(0.0)
    
    # Compute averages
    avg_recall = {k: np.mean(v) if v else 0.0 for k, v in metrics["recall"].items()}
    avg_ndcg = {k: np.mean(v) if v else 0.0 for k, v in metrics["ndcg"].items()}
    
    return {
        "recall": avg_recall,  # RENAMED from "hit_rate"
        "ndcg": avg_ndcg
    }

# ========= GENERATION EVALUATION (NEW METRICS) ========= #

@traceable(name="evaluate_generation_metrics")
def evaluate_generation_metrics(
    golds: List[Dict],
    bm25_retriever,
    openai_vs,
    bge_vs,
    strict_mode: bool = False,
    k: int = 10,
    faithfulness_top_k: int = 2
) -> Dict[str, Any]:
    """
    Evaluate generation quality with new metrics:
    - Citation Precision
    - Faithfulness/AIS (LLM-based, top K sources only)
    - Abstention Appropriateness (COMMENTED OUT)
    - Latency
    
    Args:
        golds: Gold standard queries
        bm25_retriever: BM25 retriever
        openai_vs: OpenAI vector store
        bge_vs: BGE vector store
        strict_mode: Enable strict mode for generation
        k: Number of documents to retrieve
        faithfulness_top_k: Number of top sources to use for faithfulness eval (default: 2)
    """
    
    all_results = []
    latency_results = []
    
    print(f"\nEvaluating with faithfulness check on top {faithfulness_top_k} sources...")
    
    for gold in tqdm(golds, desc="Evaluating Generation"):
        query = gold.get("query")
        if not query:
            continue
        
        try:
            # Track latency
            t_start = time.time()
            
            # Retrieval
            t0 = time.time()
            retrieved_docs = hybrid_search(query, bm25_retriever, openai_vs, bge_vs, k=k)
            retrieval_time = (time.time() - t0) * 1000
            
            # Generation
            t1 = time.time()
            result = generate_answer_with_retrieval(
                query=query,
                context=retrieved_docs,
                strict_mode=strict_mode,
                thread_id=f"eval_{gold.get('id', 'unknown')}"
            )
            generation_time = (time.time() - t1) * 1000
            
            total_time = (time.time() - t_start) * 1000
            
            # Store latency
            latency_results.append({
                'query_id': gold.get('id'),
                'retrieval_ms': round(retrieval_time, 2),
                'generation_ms': round(generation_time, 2),
                'total_ms': round(total_time, 2)
            })
            
            # Evaluate answer quality
            answer = result['answer']
            sources = result['sources']
            
            eval_result = evaluate_single_query(
                query=query,
                answer=answer,
                sources=sources,
                gold=gold,
                faithfulness_top_k=faithfulness_top_k
            )
            
            # Add latency info
            eval_result['latency'] = latency_results[-1]
            
            all_results.append(eval_result)
            
        except Exception as e:
            print(f"\nError evaluating query {gold.get('id')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_results)
    
    # Add latency statistics
    if latency_results:
        retrieval_times = [r['retrieval_ms'] for r in latency_results]
        generation_times = [r['generation_ms'] for r in latency_results]
        total_times = [r['total_ms'] for r in latency_results]
        
        aggregated['latency'] = {
            'retrieval_ms': {
                'mean': round(np.mean(retrieval_times), 2),
                'median': round(np.median(retrieval_times), 2),
                'std': round(np.std(retrieval_times), 2),
                'p95': round(np.percentile(retrieval_times, 95), 2),
                'p99': round(np.percentile(retrieval_times, 99), 2)
            },
            'generation_ms': {
                'mean': round(np.mean(generation_times), 2),
                'median': round(np.median(generation_times), 2),
                'std': round(np.std(generation_times), 2),
                'p95': round(np.percentile(generation_times, 95), 2),
                'p99': round(np.percentile(generation_times, 99), 2)
            },
            'total_ms': {
                'mean': round(np.mean(total_times), 2),
                'median': round(np.median(total_times), 2),
                'std': round(np.std(total_times), 2),
                'p95': round(np.percentile(total_times, 95), 2),
                'p99': round(np.percentile(total_times, 99), 2)
            }
        }
    
    return {
        'individual_results': all_results,
        'aggregated': aggregated
    }

# ========= FULL EVALUATION ========= #

@traceable(name="evaluate_full_system")
def evaluate_full_system(
    gold_path: Path,
    chunks: List[Dict],
    bm25_retriever,
    openai_vs,
    bge_vs,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run complete evaluation: retrieval + generation metrics.
    
    Args:
        gold_path: Path to enhanced gold set
        chunks: All document chunks
        bm25_retriever: BM25 retriever
        openai_vs: OpenAI vector store
        bge_vs: BGE vector store
        config: Evaluation configuration
    
    Returns:
        Dict with all evaluation results
    """
    
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold set not found at {gold_path}")
    
    # Load gold set
    golds = load_gold_set(gold_path)
    print(f"\nLoaded {len(golds)} queries from gold set")
    
    if not golds:
        raise ValueError("No valid queries found in gold set!")
    
    # Configuration
    k = config.get('k', 10)
    strict_mode = config.get('strict_mode', False)
    faithfulness_top_k = config.get('faithfulness_top_k', 2)
    
    print(f"\nEvaluation Config:")
    print(f"  k (retrieval): {k}")
    print(f"  strict_mode: {strict_mode}")
    print(f"  faithfulness_top_k: {faithfulness_top_k}")
    
    # 1. Evaluate Retrieval
    print("\n" + "="*70)
    print("PHASE 1: RETRIEVAL EVALUATION")
    print("="*70)
    
    retrieval_metrics = evaluate_retrieval_metrics(
        golds, chunks, bm25_retriever, openai_vs, bge_vs, k=k
    )
    
    print("\nRetrieval Metrics (Recall = % queries with ≥1 relevant doc in top k):")
    print(f"  Recall@1:  {retrieval_metrics['recall'][1]:.4f}")
    print(f"  Recall@3:  {retrieval_metrics['recall'][3]:.4f}")
    print(f"  Recall@5:  {retrieval_metrics['recall'][5]:.4f}")
    print(f"  Recall@10: {retrieval_metrics['recall'][10]:.4f}")
    print(f"  nDCG@3:    {retrieval_metrics['ndcg'][3]:.4f}")
    print(f"  nDCG@5:    {retrieval_metrics['ndcg'][5]:.4f}")
    print(f"  nDCG@10:   {retrieval_metrics['ndcg'][10]:.4f}")
    
    # 2. Evaluate Generation
    print("\n" + "="*70)
    print("PHASE 2: GENERATION EVALUATION")
    print("="*70)
    
    generation_results = evaluate_generation_metrics(
        golds, bm25_retriever, openai_vs, bge_vs,
        strict_mode=strict_mode,
        k=k,
        faithfulness_top_k=faithfulness_top_k
    )
    
    # Print summary - FIXED: Pass total_queries
    aggregated_with_total = generation_results['aggregated'].copy()
    aggregated_with_total['total_queries'] = len(golds)
    print_evaluation_summary(aggregated_with_total)
    
    # Print latency summary
    if 'latency' in generation_results['aggregated']:
        print("\n--- Latency Statistics ---")
        lat = generation_results['aggregated']['latency']
        print(f"  Retrieval: {lat['retrieval_ms']['mean']:.2f}ms (±{lat['retrieval_ms']['std']:.2f})")
        print(f"  Generation: {lat['generation_ms']['mean']:.2f}ms (±{lat['generation_ms']['std']:.2f})")
        print(f"  Total: {lat['total_ms']['mean']:.2f}ms (±{lat['total_ms']['std']:.2f})")
        print(f"  P95: {lat['total_ms']['p95']:.2f}ms")
        print(f"  P99: {lat['total_ms']['p99']:.2f}ms")
    
    # Combine results
    full_results = {
        'config': config,
        'total_queries': len(golds),
        'retrieval': retrieval_metrics,
        'generation': generation_results['aggregated'],
        'individual_results': generation_results['individual_results']
    }
    
    return full_results

# ========= MAIN ========= #

if __name__ == "__main__":
    print("Starting comprehensive system evaluation...")
    
    # Check for gold set
    if not GOLD_SET_PATH.exists():
        print(f"Error: Enhanced gold set not found at {GOLD_SET_PATH}")
        print("Please create the enhanced gold set first with expected_answer, expected_citations, etc.")
        exit(1)
    
    # Load chunks and retriever
    print("\nLoading chunks and retriever...")
    chunks = load_chunks()
    if not chunks:
        print("Error: No chunks loaded!")
        exit(1)
    
    bm25_retriever, openai_vs, bge_vs = initialize_retriever()
    
    # Configuration
    config = {
        'k': 10,
        'strict_mode': False,
        'faithfulness_top_k': 2  # Only use top 2 sources for faithfulness eval
    }
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'strict':
            config['strict_mode'] = True
            print("Running in STRICT MODE")
        elif sys.argv[1].startswith('top'):
            # Allow: python evaluate_system.py top3
            try:
                config['faithfulness_top_k'] = int(sys.argv[1][3:])
                print(f"Using top {config['faithfulness_top_k']} sources for faithfulness")
            except:
                pass
    
    # Run evaluation
    results = evaluate_full_system(
        GOLD_SET_PATH,
        chunks,
        bm25_retriever,
        openai_vs,
        bge_vs,
        config
    )
    
    # Save results
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = RESULTS_PATH / "full_evaluation_results.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to {results_file}")
    
    # Save summary only
    summary_file = RESULTS_PATH / "evaluation_summary.json"
    summary = {
        'config': results['config'],
        'total_queries': results['total_queries'],
        'retrieval': results['retrieval'],
        'generation_summary': results['generation']
    }
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
