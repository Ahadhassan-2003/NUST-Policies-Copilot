# streamlit_app.py

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict
import json
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hybrid_retriever import load_chunks, load_vector_indices, load_bm25_index, hybrid_search
from generation_module import generate_answer, get_current_academic_year

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
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* Citation styling */
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
    
    /* Source card */
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
        color: #333;
    }
    
    /* Warning styling */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Info styling */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Chat message */
    .user-message {
        background-color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========= SESSION STATE INITIALIZATION ========= #
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'indices_loaded' not in st.session_state:
    st.session_state.indices_loaded = False

if 'openai_vs' not in st.session_state:
    st.session_state.openai_vs = None

if 'bge_vs' not in st.session_state:
    st.session_state.bge_vs = None

if 'bm25_retriever' not in st.session_state:
    st.session_state.bm25_retriever = None

if 'chunks' not in st.session_state:
    st.session_state.chunks = None

if 'current_sources' not in st.session_state:
    st.session_state.current_sources = []

# ========= HELPER FUNCTIONS ========= #
@st.cache_resource
def load_all_indices():
    """Load indices once and cache them."""
    chunks = load_chunks()
    openai_vs, bge_vs = load_vector_indices()
    bm25_retriever = load_bm25_index()
    return chunks, openai_vs, bge_vs, bm25_retriever

def make_citations_clickable(text: str) -> str:
    """Convert [N] citations to clickable elements."""
    pattern = r'\[(\d+)\]'
    
    def replacement(match):
        num = match.group(1)
        return f'<span class="citation" onclick="scrollToCitation({num})">[{num}]</span>'
    
    return re.sub(pattern, replacement, text)

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
        st.markdown(f"**Content:**")
        st.markdown(f"<div class='source-text'>{text[:500]}{'...' if len(text) > 500 else ''}</div>", 
                   unsafe_allow_html=True)

def display_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Display a chat message with optional sources."""
    if role == "user":
        st.markdown(f"""
        <div class='user-message'>
            <strong>🧑 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='assistant-message'>
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    display_source_card(source, i)

# ========= SIDEBAR ========= #
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Strict Mode Toggle
    strict_mode = st.toggle(
        "🔒 Strict Mode",
        value=False,
        help="Enable aggressive citation checking and abstention when evidence is weak"
    )
    
    if strict_mode:
        st.info("""
        **Strict Mode Active:**
        - Stronger evidence required
        - More aggressive abstention
        - Citation verification enabled
        - Outdated warnings prominent
        """)
    
    st.divider()
    
    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    num_results = st.slider("Number of sources to retrieve", 3, 15, 10)
    
    st.divider()
    
    # Current Year Info
    current_year = get_current_academic_year()
    st.info(f"""
    **Current Academic Year:**  
    {current_year}
    
    **Note:** System prioritizes most recent policies.
    """)
    
    st.divider()
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_sources = []
        st.rerun()
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **NUST Policies Copilot**
        
        An AI assistant for navigating NUST university policies.
        
        **Features:**
        - Evidence-based answers with citations
        - Source verification
        - Temporal awareness
        - Abstention when uncertain
        
        **Disclaimer:** Always verify critical information with official NUST offices.
        """)

# ========= MAIN APP ========= #
st.title("🎓 NUST Policies Copilot")
st.markdown("Ask me anything about NUST policies, procedures, and regulations.")

# Load indices on first run
if not st.session_state.indices_loaded:
    with st.spinner("🔄 Loading knowledge base..."):
        try:
            chunks, openai_vs, bge_vs, bm25_retriever = load_all_indices()
            st.session_state.chunks = chunks
            st.session_state.openai_vs = openai_vs
            st.session_state.bge_vs = bge_vs
            st.session_state.bm25_retriever = bm25_retriever
            st.session_state.indices_loaded = True
            st.success("✅ Knowledge base loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
            st.stop()

# Display chat history
for message in st.session_state.chat_history:
    display_chat_message(
        message['role'],
        message['content'],
        message.get('sources')
    )

# Chat input
query = st.chat_input("Ask a question about NUST policies...")

if query:
    # Add user message to chat
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query
    })
    
    # Display user message
    display_chat_message('user', query)
    
    # Retrieve relevant documents
    with st.spinner("🔍 Searching knowledge base..."):
        try:
            retrieved_docs = hybrid_search(
                query=query,
                openai_vs=st.session_state.openai_vs,
                bge_vs=st.session_state.bge_vs,
                bm25_retriever=st.session_state.bm25_retriever,
                k=num_results
            )
        except Exception as e:
            st.error(f"❌ Error during retrieval: {e}")
            st.stop()
    
    # Generate answer
    with st.spinner("💭 Generating answer..."):
        try:
            result = generate_answer(
                query=query,
                context=retrieved_docs,
                strict_mode=strict_mode,
                chat_history=[
                    {'role': msg['role'], 'content': msg['content']}
                    for msg in st.session_state.chat_history[:-1]
                ]
            )
            
            answer = result['answer']
            sources = result['sources']
            verification_passed = result['verification_passed']
            
            # Add assistant message to chat
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': answer,
                'sources': sources,
                'verification_passed': verification_passed
            })
            
            st.session_state.current_sources = sources
            
            # Display assistant message
            display_chat_message('assistant', answer, sources)
            
            # Display verification warning if needed
            if strict_mode and not verification_passed:
                st.warning("⚠️ Citation verification failed. Answer may be incomplete or abstained.")
            
        except Exception as e:
            st.error(f"❌ Error during generation: {e}")
            st.stop()

# ========= FOOTER ========= #
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p><strong>Disclaimer:</strong> This is an AI assistant and may make mistakes. 
    Always verify important information with official NUST offices and the latest published policies.</p>
    <p>For critical matters, contact the relevant department directly.</p>
</div>
""", unsafe_allow_html=True)