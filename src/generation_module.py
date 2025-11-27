# generation_module.py

import os
import re
import sqlite3
from typing import List, Dict, Any, Optional, Annotated
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
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

# ========= STATE DEFINITION ========= #
class ChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[List[Dict[str, Any]]]
    strict_mode: bool
    query: Optional[str]
    retrieved: bool

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
    if not context:
        return "No context documents available."
    
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
        if i not in used_citations and i <= 5:
            metadata = doc.get('metadata', {})
            see_also.append({
                'title': metadata.get('doc_id', metadata.get('source', f'Document {i}')),
                'section': metadata.get('section', ''),
                'citation_num': str(i)
            })
    
    return see_also[:3]

# ========= LANGGRAPH NODES ========= #
@langsmith.traceable(name="chat_node")
def chat_node(state: ChatbotState):
    """Main chat node that generates responses with citations."""
    
    # Initialize LLM
    llm = ChatOllama(
        model="qwen2.5:3b-instruct-q4_K_M",
        temperature=0.1,
        num_predict=1024
    )
    
    # Get context and settings
    context = state.get('context', [])
    strict_mode = state.get('strict_mode', False)
    
    # Format context
    context_text = format_context_for_prompt(context)
    current_year = get_current_academic_year()
    
    # Check for outdated documents
    outdated_docs = check_document_freshness(context, current_year) if context else []
    outdated_warning = ""
    if outdated_docs and strict_mode:
        outdated_warning = "\n\n**IMPORTANT**: Some sources are outdated:\n"
        for doc in outdated_docs:
            outdated_warning += f"- [{doc['citation_num']}] {doc['doc_id']} (from {doc['year']}, {doc['age']} years old)\n"
    
    # Build system message
    system_msg_content = SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%B %d, %Y"),
        current_academic_year=current_year
    )
    
    if strict_mode:
        system_msg_content += "\n\n**STRICT MODE ACTIVE**: Be extra cautious with citations and abstain if any doubt exists."
    
    # Add context to system message if available
    if context:
        system_msg_content += f"\n\n# CONTEXT DOCUMENTS\n{context_text}{outdated_warning}"
    else:
        system_msg_content += "\n\n# NO CONTEXT AVAILABLE\nNo relevant documents were found. Politely inform the user and suggest alternative ways to find the information."
    
    # Prepare messages for LLM
    messages = [SystemMessage(content=system_msg_content)] + state['messages']
    
    # Generate response
    response = llm.invoke(messages)
    
    return {
        "messages": [response]
    }

# ========= BUILD GRAPH ========= #
def build_chatbot_graph(checkpointer=None):
    """Build the LangGraph chatbot workflow."""
    
    workflow = StateGraph(ChatbotState)
    
    # Add nodes
    workflow.add_node("chat_node", chat_node)
    
    # Add edges
    workflow.add_edge(START, "chat_node")
    workflow.add_edge("chat_node", END)
    
    # Compile with checkpointer for memory
    return workflow.compile(checkpointer=checkpointer)

# ========= DATABASE SETUP ========= #
# Create SQLite connection and checkpointer
conn = sqlite3.connect(database="nust_copilot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Build the chatbot
chatbot = build_chatbot_graph(checkpointer=checkpointer)

# ========= THREAD MANAGEMENT ========= #
def retrieve_all_threads():
    """Retrieve all conversation thread IDs from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]['thread_id'])
    return list(all_threads)

# ========= MAIN GENERATION FUNCTION ========= #
@langsmith.traceable(name="generate_answer_with_retrieval")
def generate_answer_with_retrieval(
    query: str,
    context: List[Dict[str, Any]],
    strict_mode: bool = False,
    thread_id: str = "default"
) -> Dict[str, Any]:
    """
    Generate an answer with citations and manage conversation state.
    
    Args:
        query: User's question
        context: Retrieved documents from hybrid search
        strict_mode: Enable strict citation verification
        thread_id: Conversation thread identifier
    
    Returns:
        Dictionary with answer, citations, and metadata
    """
    
    # Prepare initial state
    initial_state = ChatbotState(
        messages=[HumanMessage(content=query)],
        context=context,
        strict_mode=strict_mode,
        query=query,
        retrieved=True
    )
    
    # Configure for thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke chatbot
    result = chatbot.invoke(initial_state, config=config)
    
    # Extract answer
    answer = result['messages'][-1].content if result['messages'] else "No response generated."
    
    # Extract citations
    citations = extract_citations(answer)
    
    # Generate see-also
    see_also = generate_see_also(context, citations) if context else []
    
    # Add see-also to answer if available
    if see_also:
        answer += "\n\n**See Also:**\n"
        for item in see_also:
            section_text = f" - {item['section']}" if item['section'] else ""
            answer += f"- [{item['citation_num']}] {item['title']}{section_text}\n"
    
    return {
        "answer": answer,
        "citations": citations,
        "sources": context,
        "see_also": see_also,
        "thread_id": thread_id
    }

# ========= STREAMING SUPPORT ========= #
def stream_chatbot_response(query: str, context: List[Dict[str, Any]], strict_mode: bool, thread_id: str):
    """
    Stream chatbot responses for real-time UI updates.
    
    Args:
        query: User's question
        context: Retrieved documents
        strict_mode: Strict mode flag
        thread_id: Thread identifier
    
    Yields:
        Message chunks as they are generated
    """
    
    # Prepare initial state
    initial_state = ChatbotState(
        messages=[HumanMessage(content=query)],
        context=context,
        strict_mode=strict_mode,
        query=query,
        retrieved=True
    )
    
    # Configure for thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Stream response
    for message_chunk, metadata in chatbot.stream(
        initial_state,
        config=config,
        stream_mode='messages'
    ):
        if isinstance(message_chunk, AIMessage):
            yield message_chunk.content

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
    
    result = generate_answer_with_retrieval(
        query="What is the minimum CGPA required?",
        context=sample_context,
        strict_mode=True,
        thread_id="test_thread"
    )
    
    print("="*60)
    print("ANSWER:")
    print("="*60)
    print(result['answer'])
    print("\n" + "="*60)
    print("METADATA:")
    print("="*60)
    print(f"Citations: {result['citations']}")
    print(f"Thread ID: {result['thread_id']}")