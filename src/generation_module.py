# generation_module.py

import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import langsmith
from dotenv import load_dotenv

load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_PROJECT"] = "NUST Policies Copilot"

# ========= SYSTEM PROMPT TEMPLATE ========= #
SYSTEM_PROMPT = """You are the NUST Policies Copilot, an AI assistant specifically designed to help students, faculty, and staff understand university policies at the National University of Sciences and Technology (NUST).

# CORE RESPONSIBILITIES
1. **Answer policy questions** using ONLY the provided context documents
2. **Cite your sources** inline using numbered citations [1], [2], etc.
3. **Be concise** - provide direct, clear answers without unnecessary elaboration
4. **Stay current** - always prioritize the most recent policy documents
5. **Abstain when uncertain** - if evidence is weak, outdated, or missing, clearly state limitations

# CITATION REQUIREMENTS (CRITICAL)
- Every factual claim MUST be supported by a numbered citation: [1], [2], etc.
- Place citations immediately after the claim: "Students must maintain 2.5 CGPA [1]."
- Multiple sources for one claim: "Attendance is mandatory [1][2]."
- The citation number corresponds to the document number in the context
- NEVER make claims without citations from the provided context
- If a source doesn't support your claim, DO NOT cite it

# ANSWER STRUCTURE
1. **Direct Answer** (2-3 sentences with inline citations)
2. **Key Details** (if needed, bullet points with citations)
3. **See Also** (related documents or sections, if relevant)

# STRICT MODE BEHAVIOR
When strict_mode is enabled:
- Be MORE aggressive about requiring strong evidence
- Abstain if ANY doubt exists about source support
- Verify each citation supports its specific claim
- Provide warnings about outdated information more prominently
- Suggest where users can find authoritative answers if you cannot provide them

# TEMPORAL AWARENESS
- Always check document dates (academic_year, last_updated, etc.)
- Prefer documents from the current or most recent academic year
- Explicitly warn users when citing policies older than 2 years
- Format: "According to the 2023-24 policy [1], ..." or "Note: This policy is from 2021 and may be outdated [2]."

# SAFETY & ETHICAL GUIDELINES
- NEVER share personal data about individuals
- NEVER provide advice beyond what's explicitly stated in university policies
- NEVER make up policy information - abstain if unsure
- For sensitive topics (disciplinary actions, financial aid, medical issues), direct users to official offices
- Clearly distinguish between official policy and general guidance

# WHEN TO ABSTAIN
Abstain and provide guidance if:
- No relevant documents found in context
- Documents are significantly outdated (>2 years old) without recent confirmation
- Evidence is weak or contradictory
- Question requires interpretation beyond policy text
- Topic involves personal circumstances requiring official guidance

**Abstention Template:**
"I don't have sufficient current information to answer this accurately. I recommend:
- Checking [specific office/department]
- Visiting [official website]
- Contacting [relevant authority]
[If partial info available]: Based on older documents, [brief summary with year], but please verify current status."

# RESPONSE FORMATTING
- Use clear, professional language
- Keep paragraphs short (2-3 sentences)
- Use bullet points for lists
- Bold important terms or deadlines
- Include "See Also:" section for related topics

# CURRENT CONTEXT
- Current Date: {current_date}
- Current Academic Year: {current_academic_year}

Remember: Your primary goal is accuracy and helpfulness. When in doubt, abstain and guide users to authoritative sources rather than providing potentially incorrect information.
"""

# ========= CITATION VERIFICATION PROMPT ========= #
CITATION_VERIFICATION_PROMPT = """You are a citation verification specialist. Your job is to verify that each citation in the answer is properly supported by the source document.

# TASK
Review the answer and check if each citation [N] is actually supported by the corresponding source document.

# ANSWER TO VERIFY
{answer}

# SOURCE DOCUMENTS
{sources}

# VERIFICATION RULES
1. Each claim with citation [N] must be directly supported by source N
2. The source must explicitly state or clearly imply the claim
3. Paraphrasing is acceptable, but the meaning must match
4. If a citation is unsupported, mark it as INVALID

# OUTPUT FORMAT
Provide a JSON-like response:
{{
  "valid_citations": [1, 2, ...],
  "invalid_citations": [3, 4, ...],
  "issues": ["Citation [3] claims X but source says Y", ...],
  "verification_passed": true/false
}}

If verification_passed is false, the answer needs regeneration or the invalid citations must be removed.
"""

# ========= STATE DEFINITION ========= #
class GenerationState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    strict_mode: bool
    chat_history: List[Dict[str, str]]
    answer: Optional[str]
    citations_verified: bool
    verification_result: Optional[Dict]
    final_answer: str
    see_also: List[Dict[str, str]]

# ========= HELPER FUNCTIONS ========= #
def get_current_academic_year() -> str:
    """Determine current academic year based on current date."""
    now = datetime.now()
    year = now.year
    month = now.month
    
    # Academic year typically starts in September
    if month >= 9:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"

def format_context_for_prompt(context: List[Dict[str, Any]]) -> str:
    """Format retrieved context documents for the prompt."""
    formatted = []
    for i, doc in enumerate(context, 1):
        metadata = doc.get('metadata', {})
        text = doc.get('text', '')
        
        # Extract relevant metadata
        doc_id = metadata.get('doc_id', metadata.get('source', 'Unknown'))
        section = metadata.get('section', '')
        year = metadata.get('academic_year', metadata.get('year', 'Unknown'))
        
        formatted.append(f"[{i}] Document: {doc_id}")
        if section:
            formatted.append(f"    Section: {section}")
        formatted.append(f"    Year: {year}")
        formatted.append(f"    Content: {text}\n")
    
    return "\n".join(formatted)

def extract_citations(text: str) -> List[int]:
    """Extract citation numbers from text."""
    pattern = r'\[(\d+)\]'
    citations = re.findall(pattern, text)
    return [int(c) for c in citations]

def check_document_freshness(context: List[Dict[str, Any]], current_year: str) -> List[Dict]:
    """Identify outdated documents in context."""
    outdated = []
    current_year_int = int(current_year.split('-')[0])
    
    for i, doc in enumerate(context, 1):
        metadata = doc.get('metadata', {})
        doc_year = metadata.get('academic_year', metadata.get('year', ''))
        
        if doc_year:
            try:
                # Extract year from formats like "2021-22" or "2021"
                doc_year_int = int(str(doc_year).split('-')[0])
                age = current_year_int - doc_year_int
                
                if age >= 2:
                    outdated.append({
                        'citation_num': i,
                        'doc_id': metadata.get('doc_id', 'Unknown'),
                        'year': doc_year,
                        'age': age
                    })
            except (ValueError, AttributeError):
                pass
    
    return outdated

def generate_see_also(context: List[Dict[str, Any]], used_citations: List[int]) -> List[Dict[str, str]]:
    """Generate 'See Also' links from unused relevant documents."""
    see_also = []
    
    for i, doc in enumerate(context, 1):
        if i not in used_citations and i <= 5:  # Top 5 unused docs
            metadata = doc.get('metadata', {})
            see_also.append({
                'title': metadata.get('doc_id', metadata.get('source', f'Document {i}')),
                'section': metadata.get('section', ''),
                'citation_num': str(i)
            })
    
    return see_also[:3]  # Max 3 see-also links

# ========= LANGGRAPH NODES ========= #
@langsmith.traceable(name="generate_answer_node")
def generate_answer_node(state: GenerationState) -> GenerationState:
    """Generate initial answer with citations."""
    
    # Initialize LLM
    llm = ChatOllama(
        model="qwen2.5:3b-instruct-q4_K_M",
        temperature=0.1,
        num_predict=1024
    )
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    # )
    
    # Format context
    context_text = format_context_for_prompt(state['context'])
    current_year = get_current_academic_year()
    
    # Check for outdated documents
    outdated_docs = check_document_freshness(state['context'], current_year)
    outdated_warning = ""
    if outdated_docs and state['strict_mode']:
        outdated_warning = "\n\n**IMPORTANT**: Some sources are outdated:\n"
        for doc in outdated_docs:
            outdated_warning += f"- [{doc['citation_num']}] {doc['doc_id']} (from {doc['year']}, {doc['age']} years old)\n"
    
    # Build prompt
    system_msg = SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%B %d, %Y"),
        current_academic_year=current_year
    )
    
    if state['strict_mode']:
        system_msg += "\n\n**STRICT MODE ACTIVE**: Be extra cautious with citations and abstain if any doubt exists."
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", """# CONTEXT DOCUMENTS
{context}

# QUESTION
{question}

# INSTRUCTIONS
Provide a concise, well-cited answer. Remember:
1. Every claim needs a citation [N]
2. Be concise (2-4 sentences typically)
3. Warn about outdated information
4. Abstain if evidence is insufficient

Answer:""")
    ])
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context_text + outdated_warning,
        "question": state['query']
    })
    
    state['answer'] = answer
    return state

@langsmith.traceable(name="verify_citations_node")
def verify_citations_node(state: GenerationState) -> GenerationState:
    """Verify that citations are properly supported."""
    
    if not state['strict_mode']:
        # Skip verification in non-strict mode
        state['citations_verified'] = True
        state['verification_result'] = {"verification_passed": True}
        return state
    
    # Extract citations from answer
    citations = extract_citations(state['answer'])
    
    if not citations:
        # No citations - this is a problem if we have context
        if state['context']:
            state['citations_verified'] = False
            state['verification_result'] = {
                "verification_passed": False,
                "issues": ["Answer contains no citations despite having source documents"]
            }
        else:
            state['citations_verified'] = True
            state['verification_result'] = {"verification_passed": True}
        return state
    
    # For now, do basic validation (can be enhanced with LLM-based verification)
    max_citation = max(citations)
    num_sources = len(state['context'])
    
    if max_citation > num_sources:
        state['citations_verified'] = False
        state['verification_result'] = {
            "verification_passed": False,
            "issues": [f"Citation [{max_citation}] exceeds available sources ({num_sources})"]
        }
    else:
        state['citations_verified'] = True
        state['verification_result'] = {"verification_passed": True}
    
    return state

@langsmith.traceable(name="finalize_answer_node")
def finalize_answer_node(state: GenerationState) -> GenerationState:
    """Finalize answer with see-also links."""
    
    if not state['citations_verified']:
        # Provide abstention message
        state['final_answer'] = """I apologize, but I cannot provide a fully verified answer based on the available documents. 

To get accurate information, I recommend:
- Contacting the relevant NUST department directly
- Visiting the official NUST website
- Checking the latest student handbook

Please let me know if you'd like me to try rephrasing your question or searching for different information."""
        state['see_also'] = []
        return state
    
    # Extract used citations
    used_citations = extract_citations(state['answer'])
    
    # Generate see-also links
    see_also = generate_see_also(state['context'], used_citations)
    
    # Format final answer
    final_answer = state['answer']
    
    if see_also:
        final_answer += "\n\n**See Also:**\n"
        for item in see_also:
            section_text = f" - {item['section']}" if item['section'] else ""
            final_answer += f"- [{item['citation_num']}] {item['title']}{section_text}\n"
    
    state['final_answer'] = final_answer
    state['see_also'] = see_also
    
    return state

# ========= BUILD GRAPH ========= #
def build_generation_graph():
    """Build the LangGraph generation workflow."""
    
    workflow = StateGraph(GenerationState)
    
    # Add nodes
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("verify", verify_citations_node)
    workflow.add_node("finalize", finalize_answer_node)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# ========= MAIN GENERATION FUNCTION ========= #
@langsmith.traceable(name="generate_answer")
def generate_answer(
    query: str,
    context: List[Dict[str, Any]],
    strict_mode: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Generate an answer with citations based on retrieved context.
    
    Args:
        query: User's question
        context: Retrieved documents from hybrid search
        strict_mode: Enable strict citation verification
        chat_history: Previous conversation history
    
    Returns:
        Dictionary with answer, citations, and metadata
    """
    
    # Initialize state
    initial_state = GenerationState(
        query=query,
        context=context,
        strict_mode=strict_mode,
        chat_history=chat_history or [],
        answer=None,
        citations_verified=False,
        verification_result=None,
        final_answer="",
        see_also=[]
    )
    
    # Build and run graph
    graph = build_generation_graph()
    final_state = graph.invoke(initial_state)
    
    # Return result
    return {
        "answer": final_state['final_answer'],
        "citations": extract_citations(final_state.get('answer', '')),
        "sources": final_state['context'],
        "see_also": final_state['see_also'],
        "verification_passed": final_state['citations_verified'],
        "verification_details": final_state.get('verification_result', {})
    }

# ========= TESTING ========= #
if __name__ == "__main__":
    # Example usage
    sample_context = [
        {
            "text": "Students must maintain a minimum CGPA of 2.5 to remain in good academic standing.",
            "metadata": {
                "doc_id": "academic_policy_2024",
                "section": "Academic Standing",
                "academic_year": "2024-25"
            },
            "score": 0.95
        },
        {
            "text": "Attendance is mandatory for all courses. Students must attend at least 75% of classes.",
            "metadata": {
                "doc_id": "attendance_policy_2024",
                "section": "Attendance Requirements",
                "academic_year": "2024-25"
            },
            "score": 0.87
        }
    ]
    
    result = generate_answer(
        query="What is the minimum CGPA required?",
        context=sample_context,
        strict_mode=True
    )
    
    print("="*60)
    print("ANSWER:")
    print("="*60)
    print(result['answer'])
    print("\n" + "="*60)
    print("METADATA:")
    print("="*60)
    print(f"Citations: {result['citations']}")
    print(f"Verification Passed: {result['verification_passed']}")