# evaluation_metrics.py

import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from pathlib import Path

from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# ========= CITATION EXTRACTION ========= #

@traceable(name="extract_citations_detailed")
def extract_citations_detailed(answer: str) -> List[Dict[str, Any]]:
    """
    Extract detailed citations from answer text.
    Supports formats:
    - [1]
    - [1: doc_id]
    - [1: doc_id, p.5]
    - [1: doc_id, §section]
    - [1: doc_id, p.5, §section]
    """
    citations = []
    
    # Pattern 1: Detailed citation [N: doc_id, p.X, §section]
    detailed_pattern = r'\[(\d+):\s*([^\],]+?)(?:,\s*p\.(\d+))?(?:,\s*§([^\]]+?))?\]'
    for match in re.finditer(detailed_pattern, answer):
        citation_num = int(match.group(1))
        doc_id = match.group(2).strip()
        page = int(match.group(3)) if match.group(3) else None
        section = match.group(4).strip() if match.group(4) else None
        
        citations.append({
            'number': citation_num,
            'doc_id': doc_id,
            'page': page,
            'section': section,
            'format': 'detailed'
        })
    
    # Pattern 2: Simple citation [N]
    simple_pattern = r'\[(\d+)\]'
    for match in re.finditer(simple_pattern, answer):
        citation_num = int(match.group(1))
        
        # Don't add if already captured in detailed
        if not any(c['number'] == citation_num for c in citations):
            citations.append({
                'number': citation_num,
                'doc_id': None,
                'page': None,
                'section': None,
                'format': 'simple'
            })
    
    return citations

def extract_citation_numbers(answer: str) -> List[int]:
    """Extract just the citation numbers [1], [2], etc."""
    pattern = r'\[(\d+)(?::|(?:\]))'
    numbers = [int(m.group(1)) for m in re.finditer(pattern, answer)]
    return sorted(set(numbers))

# ========= ABSTENTION DETECTION ========= #

@traceable(name="detect_abstention")
def detect_abstention(answer: str) -> bool:
    """
    Detect if the system abstained from answering.
    
    Returns:
        bool: True if abstention detected, False otherwise
    """
    abstention_phrases = [
        "don't have sufficient",
        "don't have enough",
        "insufficient information",
        "insufficient current information",
        "recommend checking",
        "recommend contacting",
        "unable to answer",
        "cannot answer",
        "not enough current information",
        "cannot provide",
        "please verify",
        "please contact",
        "i apologize, but i cannot",
        "i'm unable to",
        "i don't have",
        "no relevant documents",
        "no context available"
    ]
    
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in abstention_phrases)

# ========= CITATION PRECISION ========= #

@traceable(name="evaluate_citation_precision")
def evaluate_citation_precision(
    answer: str,
    sources: List[Dict[str, Any]],
    gold: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate citation quality and precision.
    
    Metrics:
    - Citation count: Number of citations in answer
    - Citation coverage: % of sentences with citations
    - Expected sources found: Did we cite the expected documents?
    - Page accuracy: If pages provided, are they correct?
    - Format quality: Are citations detailed or simple?
    """
    
    # Extract citations from answer
    citations = extract_citations_detailed(answer)
    citation_numbers = extract_citation_numbers(answer)
    
    # Count sentences (rough proxy for claims)
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    sentences = [s for s in sentences if len(s) > 10]  # Filter very short sentences
    
    # Citation coverage
    citation_coverage = len(citation_numbers) / len(sentences) if sentences else 0
    
    # Check if expected sources are cited
    expected_doc_ids = gold.get('expected_doc_ids', [])
    if isinstance(expected_doc_ids, str):
        expected_doc_ids = [expected_doc_ids]
    
    # Normalize doc IDs for comparison
    def normalize_doc_id(doc_id):
        if not doc_id:
            return ""
        return str(doc_id).lower().replace('_compressed', '').replace('.pdf', '')
    
    # Get actual cited docs
    cited_doc_ids = set()
    for cit in citations:
        if cit['doc_id']:
            cited_doc_ids.add(normalize_doc_id(cit['doc_id']))
    
    # Also check source documents
    for num in citation_numbers:
        if 0 < num <= len(sources):
            source_doc_id = sources[num-1]['metadata'].get('doc_id', '')
            cited_doc_ids.add(normalize_doc_id(source_doc_id))
    
    expected_doc_ids_normalized = {normalize_doc_id(d) for d in expected_doc_ids}
    expected_sources_found = len(cited_doc_ids & expected_doc_ids_normalized)
    expected_sources_recall = expected_sources_found / len(expected_doc_ids_normalized) if expected_doc_ids_normalized else 0
    
    # Check page accuracy
    expected_citations = gold.get('expected_citations', [])
    page_matches = 0
    page_total = 0
    
    for exp_cit in expected_citations:
        exp_page = exp_cit.get('page')
        if exp_page is not None:
            page_total += 1
            # Check if any citation mentions this page
            for cit in citations:
                if cit['page'] == exp_page:
                    page_matches += 1
                    break
    
    page_accuracy = page_matches / page_total if page_total > 0 else None
    
    # Format quality
    detailed_citations = sum(1 for c in citations if c['format'] == 'detailed')
    format_quality = detailed_citations / len(citations) if citations else 0
    
    # Citation out of bounds check
    max_citation = max(citation_numbers) if citation_numbers else 0
    citations_valid = max_citation <= len(sources)
    
    return {
        'citation_count': len(citation_numbers),
        'unique_citations': len(set(citation_numbers)),
        'citation_coverage': round(citation_coverage, 3),
        'expected_sources_found': expected_sources_found,
        'expected_sources_total': len(expected_doc_ids_normalized),
        'expected_sources_recall': round(expected_sources_recall, 3),
        'page_accuracy': round(page_accuracy, 3) if page_accuracy is not None else None,
        'format_quality': round(format_quality, 3),
        'citations_valid': citations_valid,
        'detailed_citations': detailed_citations,
        'simple_citations': len(citations) - detailed_citations
    }

# ========= FAITHFULNESS / AIS ========= #

@traceable(name="evaluate_faithfulness_simple")
def evaluate_faithfulness_simple(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple faithfulness check using ngram overlap.
    
    Measures what percentage of answer content appears in sources.
    """
    
    # Remove citations and clean answer
    answer_clean = re.sub(r'\[(?:\d+:?[^\]]*)\]', '', answer)
    answer_clean = answer_clean.lower().strip()
    
    # Extract source texts
    source_texts = ' '.join([doc['text'].lower() for doc in sources])
    
    # Compute word overlap
    answer_words = set(answer_clean.split())
    source_words = set(source_texts.split())
    
    # Filter out common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                   'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                   'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                   'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those'}
    
    answer_words_filtered = answer_words - common_words
    
    if not answer_words_filtered:
        return {'word_overlap': 0.0, 'ngram_overlap_2': 0.0, 'ngram_overlap_3': 0.0}
    
    # Word overlap
    word_overlap = len(answer_words_filtered & source_words) / len(answer_words_filtered)
    
    # Bigram overlap
    def get_ngrams(text, n):
        words = text.split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    answer_bigrams = get_ngrams(answer_clean, 2)
    source_bigrams = get_ngrams(source_texts, 2)
    bigram_overlap = len(answer_bigrams & source_bigrams) / len(answer_bigrams) if answer_bigrams else 0
    
    # Trigram overlap
    answer_trigrams = get_ngrams(answer_clean, 3)
    source_trigrams = get_ngrams(source_texts, 3)
    trigram_overlap = len(answer_trigrams & source_trigrams) / len(answer_trigrams) if answer_trigrams else 0
    
    return {
        'word_overlap': round(word_overlap, 3),
        'ngram_overlap_2': round(bigram_overlap, 3),
        'ngram_overlap_3': round(trigram_overlap, 3),
        'faithfulness_score': round((word_overlap + bigram_overlap + trigram_overlap) / 3, 3)
    }

@traceable(name="evaluate_faithfulness_llm")
def evaluate_faithfulness_llm(
    answer: str, 
    sources: List[Dict[str, Any]], 
    query: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Advanced faithfulness check using LLM as judge.
    
    The LLM evaluates if the answer is fully supported by sources.
    """
    
    try:
        llm = ChatOpenAI(model=model, temperature=0)
        
        # Format sources
        sources_text = "\n\n".join([
            f"Source {i+1}:\n{doc['text'][:500]}"
            for i, doc in enumerate(sources[:5])  # Limit to top 5 sources
        ])
        
        prompt = f"""You are evaluating if an AI assistant's answer is fully supported by the provided sources.

Query: {query}

Answer: {answer}

Sources:
{sources_text}

Evaluate the answer on a scale of 0-10:
- 10: Every claim in the answer is directly supported by the sources
- 7-9: Most claims supported, minor unsupported details
- 4-6: Some claims supported, some unsupported or inferred
- 1-3: Few claims supported, mostly unsupported
- 0: Answer contradicts sources or is completely unsupported

Respond in this exact format:
Score: [0-10]
Reasoning: [one sentence explaining the score]
Unsupported claims: [list any claims not found in sources, or "None"]"""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response
        score_match = re.search(r'Score:\s*(\d+)', response_text)
        score = int(score_match.group(1)) / 10 if score_match else 0.5
        
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', response_text)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        unsupported_match = re.search(r'Unsupported claims:\s*(.+?)(?:\n\n|$)', response_text, re.DOTALL)
        unsupported = unsupported_match.group(1).strip() if unsupported_match else ""
        
        return {
            'faithfulness_score_llm': round(score, 3),
            'reasoning': reasoning,
            'unsupported_claims': unsupported,
            'llm_response': response_text
        }
    
    except Exception as e:
        print(f"Warning: LLM faithfulness evaluation failed: {e}")
        return {
            'faithfulness_score_llm': None,
            'reasoning': f"Error: {str(e)}",
            'unsupported_claims': "Error during evaluation",
            'llm_response': None
        }

# ========= ABSTENTION APPROPRIATENESS ========= #

@traceable(name="evaluate_abstention")
def evaluate_abstention(
    answer: str,
    query: str,
    sources: List[Dict[str, Any]],
    gold: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate if abstention was appropriate.
    
    Checks:
    - Did system abstain when it should?
    - Did system answer when it should abstain?
    """
    
    # Detect if system abstained
    system_abstained = detect_abstention(answer)
    
    # Check gold standard
    should_abstain = gold.get('should_abstain', False)
    
    # Determine appropriateness
    if system_abstained and should_abstain:
        result = "correct_abstention"
        appropriate = True
    elif not system_abstained and not should_abstain:
        result = "correct_answer"
        appropriate = True
    elif system_abstained and not should_abstain:
        result = "false_abstention"  # Abstained but shouldn't have
        appropriate = False
    else:  # not system_abstained and should_abstain
        result = "false_confidence"  # Answered but should have abstained
        appropriate = False
    
    # Additional context
    no_sources = len(sources) == 0
    weak_sources = len(sources) > 0 and all(doc.get('score', 0) < 0.5 for doc in sources)
    
    return {
        'system_abstained': system_abstained,
        'should_abstain': should_abstain,
        'appropriate': appropriate,
        'result': result,
        'no_sources': no_sources,
        'weak_sources': weak_sources,
        'num_sources': len(sources)
    }

# ========= LATENCY TRACKING ========= #

@traceable(name="track_latency")
def track_latency(
    query: str,
    retrieval_fn,
    generation_fn,
    k: int = 10
) -> Dict[str, float]:
    """
    Track latency for retrieval and generation.
    
    Returns:
        Dict with timing information in milliseconds
    """
    
    # Retrieval timing
    t0 = time.time()
    retrieved_docs = retrieval_fn(query, k=k)
    retrieval_time = (time.time() - t0) * 1000
    
    # Generation timing
    t1 = time.time()
    answer_result = generation_fn(query, retrieved_docs)
    generation_time = (time.time() - t1) * 1000
    
    total_time = retrieval_time + generation_time
    
    return {
        'retrieval_ms': round(retrieval_time, 2),
        'generation_ms': round(generation_time, 2),
        'total_ms': round(total_time, 2),
        'retrieval_sec': round(retrieval_time / 1000, 3),
        'generation_sec': round(generation_time / 1000, 3),
        'total_sec': round(total_time / 1000, 3)
    }

# ========= ANSWER QUALITY (OPTIONAL) ========= #

@traceable(name="evaluate_answer_quality")
def evaluate_answer_quality(
    answer: str,
    expected_answer: str,
    query: str
) -> Dict[str, Any]:
    """
    Optional: Compare generated answer with expected answer.
    
    Uses simple text similarity metrics.
    """
    
    # Clean texts
    def clean_text(text):
        text = re.sub(r'\[(?:\d+:?[^\]]*)\]', '', text)  # Remove citations
        return text.lower().strip()
    
    answer_clean = clean_text(answer)
    expected_clean = clean_text(expected_answer)
    
    # Word overlap
    answer_words = set(answer_clean.split())
    expected_words = set(expected_clean.split())
    
    if not expected_words:
        return {'word_overlap': 0.0, 'key_terms_coverage': 0.0}
    
    word_overlap = len(answer_words & expected_words) / len(expected_words)
    
    # Key terms (longer words are likely more important)
    key_terms = {w for w in expected_words if len(w) > 5}
    key_terms_coverage = len(answer_words & key_terms) / len(key_terms) if key_terms else 0
    
    return {
        'word_overlap_with_expected': round(word_overlap, 3),
        'key_terms_coverage': round(key_terms_coverage, 3)
    }

# ========= AGGREGATE EVALUATION ========= #

@traceable(name="evaluate_single_query")
def evaluate_single_query(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    gold: Dict[str, Any],
    use_llm_faithfulness: bool = False
) -> Dict[str, Any]:
    """
    Run all evaluation metrics on a single query.
    
    Returns:
        Dict with all metrics
    """
    
    results = {
        'query_id': gold.get('id'),
        'query': query,
        'answer_length': len(answer),
        'num_sources': len(sources)
    }
    
    # Citation Precision
    citation_metrics = evaluate_citation_precision(answer, sources, gold)
    results['citation'] = citation_metrics
    
    # Faithfulness (simple version always, LLM optional)
    faithfulness_simple = evaluate_faithfulness_simple(answer, sources)
    results['faithfulness'] = faithfulness_simple
    
    if use_llm_faithfulness and sources:
        faithfulness_llm = evaluate_faithfulness_llm(answer, sources, query)
        results['faithfulness_llm'] = faithfulness_llm
    
    # Abstention
    abstention_metrics = evaluate_abstention(answer, query, sources, gold)
    results['abstention'] = abstention_metrics
    
    # Answer Quality (if expected answer provided)
    if gold.get('expected_answer'):
        quality_metrics = evaluate_answer_quality(answer, gold['expected_answer'], query)
        results['answer_quality'] = quality_metrics
    
    return results

# ========= UTILITY FUNCTIONS ========= #

def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all queries."""
    
    if not all_results:
        return {}
    
    aggregated = {
        'total_queries': len(all_results),
        'citation': {},
        'faithfulness': {},
        'abstention': {},
        'answer_quality': {}
    }
    
    # Citation metrics
    citation_keys = ['citation_count', 'citation_coverage', 'expected_sources_recall', 
                     'format_quality', 'page_accuracy']
    for key in citation_keys:
        values = [r['citation'][key] for r in all_results if r['citation'].get(key) is not None]
        if values:
            aggregated['citation'][f'{key}_mean'] = round(np.mean(values), 3)
            aggregated['citation'][f'{key}_std'] = round(np.std(values), 3)
    
    # Faithfulness metrics
    faith_keys = ['word_overlap', 'ngram_overlap_2', 'ngram_overlap_3', 'faithfulness_score']
    for key in faith_keys:
        values = [r['faithfulness'][key] for r in all_results if key in r.get('faithfulness', {})]
        if values:
            aggregated['faithfulness'][f'{key}_mean'] = round(np.mean(values), 3)
            aggregated['faithfulness'][f'{key}_std'] = round(np.std(values), 3)
    
    # Abstention metrics
    abstention_counts = Counter(r['abstention']['result'] for r in all_results)
    total = len(all_results)
    
    aggregated['abstention'] = {
        'abstention_rate': round(sum(r['abstention']['system_abstained'] for r in all_results) / total, 3),
        'should_abstain_rate': round(sum(r['abstention']['should_abstain'] for r in all_results) / total, 3),
        'appropriate_rate': round(sum(r['abstention']['appropriate'] for r in all_results) / total, 3),
        'correct_abstentions': abstention_counts.get('correct_abstention', 0),
        'correct_answers': abstention_counts.get('correct_answer', 0),
        'false_abstentions': abstention_counts.get('false_abstention', 0),
        'false_confidence': abstention_counts.get('false_confidence', 0)
    }
    
    # Answer quality (if available)
    if any('answer_quality' in r for r in all_results):
        quality_values = [r['answer_quality']['word_overlap_with_expected'] 
                         for r in all_results if 'answer_quality' in r]
        if quality_values:
            aggregated['answer_quality']['word_overlap_mean'] = round(np.mean(quality_values), 3)
    
    return aggregated

def print_evaluation_summary(aggregated: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nTotal Queries Evaluated: {aggregated['total_queries']}")
    
    # Citation Metrics
    print("\n--- Citation Quality ---")
    cit = aggregated['citation']
    print(f"  Average Citations per Answer: {cit.get('citation_count_mean', 0):.2f}")
    print(f"  Citation Coverage: {cit.get('citation_coverage_mean', 0):.3f}")
    print(f"  Expected Sources Recall: {cit.get('expected_sources_recall_mean', 0):.3f}")
    print(f"  Format Quality (detailed): {cit.get('format_quality_mean', 0):.3f}")
    if cit.get('page_accuracy_mean') is not None:
        print(f"  Page Accuracy: {cit.get('page_accuracy_mean', 0):.3f}")
    
    # Faithfulness Metrics
    print("\n--- Faithfulness / Grounding ---")
    faith = aggregated['faithfulness']
    print(f"  Word Overlap: {faith.get('word_overlap_mean', 0):.3f}")
    print(f"  Bigram Overlap: {faith.get('ngram_overlap_2_mean', 0):.3f}")
    print(f"  Trigram Overlap: {faith.get('ngram_overlap_3_mean', 0):.3f}")
    print(f"  Overall Faithfulness Score: {faith.get('faithfulness_score_mean', 0):.3f}")
    
    # Abstention Metrics
    print("\n--- Abstention Behavior ---")
    abst = aggregated['abstention']
    print(f"  System Abstention Rate: {abst['abstention_rate']:.3f}")
    print(f"  Should Abstain Rate (gold): {abst['should_abstain_rate']:.3f}")
    print(f"  Appropriate Behavior Rate: {abst['appropriate_rate']:.3f}")
    print(f"  Breakdown:")
    print(f"    Correct Answers: {abst['correct_answers']}")
    print(f"    Correct Abstentions: {abst['correct_abstentions']}")
    print(f"    False Abstentions: {abst['false_abstentions']}")
    print(f"    False Confidence: {abst['false_confidence']}")
    
    # Answer Quality
    if aggregated.get('answer_quality'):
        print("\n--- Answer Quality ---")
        qual = aggregated['answer_quality']
        print(f"  Word Overlap with Expected: {qual.get('word_overlap_mean', 0):.3f}")
    
    print("\n" + "="*70)