# streamlit_app.py

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict
import uuid
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hybrid_retriever import load_chunks, load_bm25_index, load_vector_indices, hybrid_search
from generation_module import (
    chatbot, 
    retrieve_all_threads, 
    get_current_academic_year,
    ChatbotState,
    generate_see_also,
    extract_citations
)

# ========= PAGE CONFIG ========= #
st.set_page_config(
    page_title="NUST Policies Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= CUSTOM CSS ========= #
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .citation {
        background-color: #e3f2fd;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #1976d2;
        cursor: pointer;
        text-decoration: none;
    }
    
    .citation:hover {
        background-color: #bbdefb;
    }
    
    .source-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
    }
    
    .source-header {
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    
    .source-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .source-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #ffff;
    }
    
    .thread-button {
        width: 100%;
        text-align: left;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .thread-button:hover {
        background-color: #e9ecef;
    }
    
    .thread-button-active {
        background-color: #d4edda;
        border-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ========= UTILITY FUNCTIONS ========= #
def generate_thread_id():
    """Generate a new unique thread ID."""
    return str(uuid.uuid4())

def reset_chat():
    """Start a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['current_sources'] = []

def add_thread(thread_id):
    """Add a new thread to the list if it doesn't exist."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id: str):
    """Load conversation history from a specific thread."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get('messages', [])
        
        # Convert LangChain messages to streamlit format
        temp_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                temp_history.append({"role": "assistant", "content": msg.content})
        
        return temp_history
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []

def get_thread_preview(thread_id: str, max_length: int = 50) -> str:
    """Get a preview of the thread's first message."""
    try:
        messages = load_conversation(thread_id)
        if messages and len(messages) > 0:
            first_msg = messages[0]['content']
            if len(first_msg) > max_length:
                return first_msg[:max_length] + "..."
            return first_msg
        return str(thread_id)[:8]
    except:
        return str(thread_id)[:8]

def display_source_card(source: Dict, index: int):
    """Display a source document card."""
    metadata = source.get('metadata', {})
    text = source.get('text', '')
    
    doc_id = metadata.get('doc_id', metadata.get('source', 'Unknown'))
    section = metadata.get('section', 'N/A')
    year = metadata.get('academic_year', metadata.get('year', 'N/A'))
    source_type = metadata.get('source_type', 'N/A')
    
    with st.expander(f"📄 [{index}] {doc_id}", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Section:** {section}")
        with col2:
            st.markdown(f"**Year:** {year}")
        with col3:
            st.markdown(f"**Type:** {source_type.upper()}")
        
        st.markdown("---")
        st.markdown("**Content:**")
        st.markdown(f"<div class='source-text'>{text[:500]}{'...' if len(text) > 500 else ''}</div>", 
                   unsafe_allow_html=True)

# ========= SESSION STATE INITIALIZATION ========= #
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'indices_loaded' not in st.session_state:
    st.session_state['indices_loaded'] = False

if 'openai_vs' not in st.session_state:
    st.session_state['openai_vs'] = None

if 'bge_vs' not in st.session_state:
    st.session_state['bge_vs'] = None

if 'bm25_retriever' not in st.session_state:
    st.session_state['bm25_retriever'] = None

if 'chunks' not in st.session_state:
    st.session_state['chunks'] = None

if 'current_sources' not in st.session_state:
    st.session_state['current_sources'] = []

if 'strict_mode' not in st.session_state:
    st.session_state['strict_mode'] = False

# Add current thread to list
add_thread(st.session_state['thread_id'])

# ========= LOAD INDICES ========= #
@st.cache_resource
def load_all_indices():
    """Load indices once and cache them."""
    chunks = load_chunks()
    openai_vs, bge_vs = load_vector_indices()
    bm25_retriever = load_bm25_index()
    return chunks, openai_vs, bge_vs, bm25_retriever

# ========= SIDEBAR ========= #
with st.sidebar:
    st.title("🎓 NUST Copilot")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
    
    st.divider()
    
    # Settings Section
    st.subheader("⚙️ Settings")
    
    # Strict Mode Toggle
    strict_mode = st.toggle(
        "🔒 Strict Mode",
        value=st.session_state.get('strict_mode', False),
        help="Enable aggressive citation checking and abstention when evidence is weak"
    )
    st.session_state['strict_mode'] = strict_mode
    
    if strict_mode:
        st.info("**Strict Mode:** Stronger evidence required, more aggressive abstention")
    
    # Retrieval Settings
    num_results = st.slider("Number of sources", 3, 15, 10)
    
    st.divider()
    
    # Chat History Section
    st.subheader("💬 My Chats")
    
    # Display chat threads
    for thread_id in st.session_state['chat_threads']:
        is_active = thread_id == st.session_state['thread_id']
        preview = get_thread_preview(thread_id)
        
        button_label = f"{'🟢 ' if is_active else '⚪ '}{preview}"
        
        if st.button(
            button_label,
            key=f"thread_{thread_id}",
            use_container_width=True,
            type="secondary" if is_active else "tertiary"
        ):
            if thread_id != st.session_state['thread_id']:
                st.session_state['thread_id'] = thread_id
                st.session_state['message_history'] = load_conversation(thread_id)
                st.session_state['current_sources'] = []
                st.rerun()
    
    st.divider()
    
    # Current Year Info
    current_year = get_current_academic_year()
    st.info(f"""
    **Current Academic Year:**  
    {current_year}
    
    Prioritizing most recent policies.
    """)
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **NUST Policies Copilot**
        
        AI assistant for navigating NUST policies.
        
        **Features:**
        - Evidence-based answers
        - Inline citations
        - Source verification
        - Multi-turn conversations
        - Persistent chat history
        
        **Disclaimer:** Verify critical information with official NUST offices.
        """)

# ========= MAIN APP ========= #
st.title("🎓 NUST Policies Copilot")
st.markdown("Ask me anything about NUST policies, procedures, and regulations.")

# Load indices on first run
if not st.session_state['indices_loaded']:
    with st.spinner("🔄 Loading knowledge base..."):
        try:
            chunks, openai_vs, bge_vs, bm25_retriever = load_all_indices()
            st.session_state['chunks'] = chunks
            st.session_state['openai_vs'] = openai_vs
            st.session_state['bge_vs'] = bge_vs
            st.session_state['bm25_retriever'] = bm25_retriever
            st.session_state['indices_loaded'] = True
            st.success("✅ Knowledge base loaded!")
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
            st.stop()

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display sources if available and it's an assistant message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    display_source_card(source, i)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Retrieve relevant documents
    with st.spinner("🔍 Searching knowledge base..."):
        try:
            retrieved_docs = hybrid_search(
                query=user_input,
                openai_vs=st.session_state['openai_vs'],
                bge_vs=st.session_state['bge_vs'],
                bm25_retriever=st.session_state['bm25_retriever'],
                k=num_results
            )
            st.session_state['current_sources'] = retrieved_docs
        except Exception as e:
            st.error(f"❌ Error during retrieval: {e}")
            st.stop()
    
    # Generate streaming response
    with st.chat_message("assistant"):
        # Prepare state for chatbot
        initial_state = ChatbotState(
            messages=[HumanMessage(content=user_input)],
            context=retrieved_docs,
            strict_mode=st.session_state['strict_mode'],
            query=user_input,
            retrieved=True
        )
        
        # Configure thread
        CONFIG = RunnableConfig({
            "configurable": {"thread_id": st.session_state['thread_id']},
            "metadata": {
                "thread_id": st.session_state['thread_id'],
                "strict_mode": st.session_state['strict_mode']
            },
            "run_name": "policy_query"
        })
        
        # Stream response (AI messages only)
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                initial_state,
                config=CONFIG,
                stream_mode='messages'
            ):
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content
        
        # Display streaming response
        ai_message = st.write_stream(ai_only_stream())
        
        # Add see-also section if applicable
        citations = extract_citations(ai_message) # type: ignore
        see_also = generate_see_also(retrieved_docs, citations) if retrieved_docs else []
        
        if see_also:
            see_also_text = "\n\n**See Also:**\n"
            for item in see_also:
                # Use full citation display
                see_also_text += f"- [{item['citation_num']}: {item['citation_display']}]\n"
            st.markdown(see_also_text)
            ai_message += see_also_text
        
        # Display sources
        if retrieved_docs:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(retrieved_docs, 1):
                    display_source_card(source, i)
    
    # Add assistant message to history
    st.session_state['message_history'].append({
        "role": "assistant",
        "content": ai_message,
        "sources": retrieved_docs
    })

# ========= FOOTER ========= #
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p><strong>Disclaimer:</strong> This is an AI assistant and may make mistakes. 
    Always verify important information with official NUST offices and the latest published policies.</p>
    <p>For critical matters, contact the relevant department directly.</p>
    <p style='margin-top: 1rem; font-size: 0.75rem;'>Thread ID: {}</p>
</div>
""".format(st.session_state['thread_id'][:8]), unsafe_allow_html=True)