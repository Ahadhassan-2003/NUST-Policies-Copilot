# create_embeddings.py

import json
import pickle
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
from langsmith import traceable
import os

load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_PROJECT"] = "NUST Policies Copilot"

# ========= INPUT / OUTPUT PATHS ========= #
PDF_JSONL = Path("./data/processed/pdf_chunks.jsonl")
HTML_JSONL = Path("./data/processed/html_chunks.jsonl")

INDEX_A_DIR = "./data/vectorstores/chroma_index_openAI"
INDEX_B_DIR = "./data/vectorstores/chroma_index_bge_m3"
BM25_INDEX_PATH = Path("./data/vectorstores/bm25_index.pkl")

# ========= FLATTEN METADATA FOR CHROMADB ========= #
@traceable(name="flatten_metadata")
def flatten_metadata(item):
    """
    ChromaDB only supports simple types (str, int, float, bool) in metadata.
    Convert complex fields like lists to JSON strings.
    Ensures page, section, URL, doc_id are preserved.
    """
    metadata = {}
    
    # Priority fields that should always be extracted
    priority_fields = ['doc_id', 'page', 'section', 'url', 'URL', 'year', 'source']
    
    for key, value in item.items():
        # Skip text content fields
        if key in ["chunk_text", "text"]:
            continue
        
        # Handle priority fields - ensure they're simple types
        if key in priority_fields:
            if value is None:
                continue  # Skip None values
            elif isinstance(value, (int, float, bool)):
                metadata[key] = value
            else:
                # Convert to string for consistency
                metadata[key] = str(value)
        
        # Handle other fields
        elif isinstance(value, (list, dict)):
            # Serialize complex types to JSON string
            metadata[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)):
            # Keep simple types as-is
            metadata[key] = value
        elif value is not None:
            # Convert anything else to string
            metadata[key] = str(value)
    
    # Normalize URL field (handle both 'url' and 'URL')
    if 'URL' in metadata and 'url' not in metadata:
        metadata['url'] = metadata['URL']
    elif 'url' in metadata and 'URL' not in metadata:
        metadata['URL'] = metadata['url']
    
    # Ensure doc_id exists (fallback to source if needed)
    if 'doc_id' not in metadata and 'source' in metadata:
        metadata['doc_id'] = metadata['source']
    
    return metadata

# ========= VALIDATE METADATA ========= #
@traceable(name="validate_chunk_metadata")
def validate_chunk_metadata(chunks):
    """
    Validate that chunks have required metadata fields.
    Print statistics about metadata coverage.
    """
    print("\n" + "="*60)
    print("METADATA VALIDATION")
    print("="*60)
    
    total = len(chunks)
    stats = {
        'has_doc_id': 0,
        'has_page': 0,
        'has_section': 0,
        'has_url': 0,
        'has_year': 0
    }
    
    missing_doc_id = []
    
    for i, chunk in enumerate(chunks):
        metadata = chunk['metadata']
        
        if metadata.get('doc_id'):
            stats['has_doc_id'] += 1
        else:
            missing_doc_id.append(i)
        
        if metadata.get('page') is not None:
            stats['has_page'] += 1
        
        if metadata.get('section'):
            stats['has_section'] += 1
        
        if metadata.get('url') or metadata.get('URL'):
            stats['has_url'] += 1
        
        if metadata.get('year'):
            stats['has_year'] += 1
    
    # Print statistics
    print(f"Total chunks: {total}")
    print(f"\nMetadata Coverage:")
    print(f"  doc_id:  {stats['has_doc_id']:4d} / {total} ({stats['has_doc_id']/total*100:.1f}%)")
    print(f"  page:    {stats['has_page']:4d} / {total} ({stats['has_page']/total*100:.1f}%)")
    print(f"  section: {stats['has_section']:4d} / {total} ({stats['has_section']/total*100:.1f}%)")
    print(f"  url:     {stats['has_url']:4d} / {total} ({stats['has_url']/total*100:.1f}%)")
    print(f"  year:    {stats['has_year']:4d} / {total} ({stats['has_year']/total*100:.1f}%)")
    
    if missing_doc_id:
        print(f"\n⚠ Warning: {len(missing_doc_id)} chunks missing doc_id")
        if len(missing_doc_id) <= 5:
            print(f"  Indices: {missing_doc_id}")
    
    # Sample metadata for verification
    print(f"\nSample Metadata (first 3 chunks):")
    for i in range(min(3, len(chunks))):
        print(f"\n  Chunk {i}:")
        metadata = chunks[i]['metadata']
        print(f"    doc_id:  {metadata.get('doc_id', 'N/A')}")
        print(f"    page:    {metadata.get('page', 'N/A')}")
        print(f"    section: {metadata.get('section', 'N/A')[:50] if metadata.get('section') else 'N/A'}")
        print(f"    url:     {metadata.get('url', metadata.get('URL', 'N/A'))}")
        print(f"    year:    {metadata.get('year', 'N/A')}")
    
    print("="*60 + "\n")
    
    return stats

# ========= LOAD CHUNKS ========= #
@traceable(name="load_chunks")
def load_chunks():
    chunks = []
    
    # Load PDF chunks
    if PDF_JSONL.exists():
        print(f"Loading PDF chunks from {PDF_JSONL}...")
        with PDF_JSONL.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    
                    # Get text content - try both "text" and "chunk_text" fields
                    text_content = item.get("text") or item.get("chunk_text", "")
                    text_content = text_content.strip()
                    
                    if not text_content:
                        continue
                    
                    # Flatten metadata (handle nested structures)
                    metadata = flatten_metadata(item)
                    metadata["source_type"] = "pdf"
                    
                    chunks.append({
                        "text": text_content,
                        "metadata": metadata
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing line {line_num}: {e}")
        
        print(f"  ✓ Loaded {len(chunks)} PDF chunks")
    else:
        print(f"Warning: Missing file {PDF_JSONL}")
    
    # Load HTML chunks
    pdf_count = len(chunks)
    if HTML_JSONL.exists():
        print(f"Loading HTML chunks from {HTML_JSONL}...")
        with HTML_JSONL.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    
                    # Get text content
                    text_content = item.get("text") or item.get("chunk_text", "")
                    text_content = text_content.strip()
                    
                    if not text_content:
                        continue
                    
                    # Flatten metadata (this will handle the links array)
                    metadata = flatten_metadata(item)
                    metadata["source_type"] = "html"
                    
                    chunks.append({
                        "text": text_content,
                        "metadata": metadata
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing line {line_num}: {e}")
        
        print(f"  ✓ Loaded {len(chunks) - pdf_count} HTML chunks")
    else:
        print(f"Warning: Missing file {HTML_JSONL}")
    
    print(f"\nTotal chunks loaded: {len(chunks)}\n")
    
    # Validate metadata
    if chunks:
        validate_chunk_metadata(chunks)
    
    return chunks

# ========= BUILD BM25 INDEX ========= #
@traceable(name="build_bm25_index")
def build_bm25_index(chunks):
    print("="*60)
    print("Creating BM25 Index...")
    print("="*60)
    
    try:
        # Create LangChain Document objects with metadata
        lc_docs = [
            Document(page_content=c["text"], metadata=c["metadata"]) 
            for c in chunks
        ]
        
        print(f"Building BM25 index from {len(lc_docs)} documents...")
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(lc_docs)
        bm25_retriever.k = 50  # Default retrieval count
        
        # Save the retriever
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        
        print(f"\n✓ BM25 Index saved at: {BM25_INDEX_PATH}")
        print(f"  Total documents: {len(lc_docs)}")
        
        # Verify metadata preservation
        print(f"\n  Verifying metadata preservation in BM25...")
        sample_doc = lc_docs[0]
        print(f"    Sample doc_id: {sample_doc.metadata.get('doc_id', 'N/A')}")
        print(f"    Sample page: {sample_doc.metadata.get('page', 'N/A')}")
        print(f"    Sample section: {sample_doc.metadata.get('section', 'N/A')[:50] if sample_doc.metadata.get('section') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating BM25 Index: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========= BUILD INDEX A (OpenAI Embeddings) ========= #
# @traceable(name="build_index_openai")
# def build_index_openai(chunks):
#     print("="*60)
#     print("Creating Chroma Index A (OpenAI Embeddings)...")
#     print("="*60)
    
#     try:
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
#         texts = [c["text"] for c in chunks]
#         metadatas = [c["metadata"] for c in chunks]
        
#         # Create the index in batches to avoid timeout issues
#         batch_size = 100
#         print(f"Processing {len(texts)} chunks in batches of {batch_size}...")
        
#         # Create initial vectorstore with first batch
#         vectorstore = Chroma.from_texts(
#             texts=texts[:batch_size],
#             embedding=embeddings,
#             metadatas=metadatas[:batch_size],
#             persist_directory=INDEX_A_DIR
#         )
        
#         # Add remaining batches
#         for i in range(batch_size, len(texts), batch_size):
#             batch_texts = texts[i:i+batch_size]
#             batch_metadatas = metadatas[i:i+batch_size]
            
#             vectorstore.add_texts(
#                 texts=batch_texts,
#                 metadatas=batch_metadatas
#             )
            
#             print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
#         print(f"\n✓ Index A saved at: {INDEX_A_DIR}")
#         print(f"  Total documents: {len(texts)}")
        
#         # Verify metadata preservation
#         print(f"\n  Verifying metadata preservation in Chroma...")
#         test_results = vectorstore.similarity_search("test", k=1)
#         if test_results:
#             sample_meta = test_results[0].metadata
#             print(f"    Sample doc_id: {sample_meta.get('doc_id', 'N/A')}")
#             print(f"    Sample page: {sample_meta.get('page', 'N/A')}")
#             print(f"    Sample section: {sample_meta.get('section', 'N/A')[:50] if sample_meta.get('section') else 'N/A'}")
        
#         return True
        
#     except Exception as e:
#         print(f"\n✗ Error creating Index A: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # ========= BUILD INDEX B (BAAI/BGE-M3) ========= #
# @traceable(name="build_index_bge")
# def build_index_bge(chunks):
#     print("="*60)
#     print("Creating Chroma Index B (BAAI/bge-m3 Embeddings)...")
#     print("="*60)
    
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        
#         texts = [c["text"] for c in chunks]
#         metadatas = [c["metadata"] for c in chunks]
        
#         # Create the index in batches
#         batch_size = 100
#         print(f"Processing {len(texts)} chunks in batches of {batch_size}...")
        
#         # Create initial vectorstore with first batch
#         vectorstore = Chroma.from_texts(
#             texts=texts[:batch_size],
#             embedding=embeddings,
#             metadatas=metadatas[:batch_size],
#             persist_directory=INDEX_B_DIR
#         )
        
#         # Add remaining batches
#         for i in range(batch_size, len(texts), batch_size):
#             batch_texts = texts[i:i+batch_size]
#             batch_metadatas = metadatas[i:i+batch_size]
            
#             vectorstore.add_texts(
#                 texts=batch_texts,
#                 metadatas=batch_metadatas
#             )
            
#             print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
#         print(f"\n✓ Index B saved at: {INDEX_B_DIR}")
#         print(f"  Total documents: {len(texts)}")
        
#         # Verify metadata preservation
#         print(f"\n  Verifying metadata preservation in BGE...")
#         test_results = vectorstore.similarity_search("test", k=1)
#         if test_results:
#             sample_meta = test_results[0].metadata
#             print(f"    Sample doc_id: {sample_meta.get('doc_id', 'N/A')}")
#             print(f"    Sample page: {sample_meta.get('page', 'N/A')}")
#             print(f"    Sample section: {sample_meta.get('section', 'N/A')[:50] if sample_meta.get('section') else 'N/A'}")
        
#         return True
        
#     except Exception as e:
#         print(f"\n✗ Error creating Index B: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# ========= MAIN RUNNER ========= #
@traceable(name="create_all_indices")
def main():
    print("Starting embedding and index generation process...\n")
    
    chunks = load_chunks()
    
    if len(chunks) == 0:
        print("No chunks found. Exiting.")
        exit(1)
    
    # Build all indexes
    success_bm25 = build_bm25_index(chunks)
    # success_a = build_index_openai(chunks)
    # success_b = build_index_bge(chunks)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"BM25 Index:       {'✓ Success' if success_bm25 else '✗ Failed'}")
    # print(f"Index A (OpenAI): {'✓ Success' if success_a else '✗ Failed'}")
    # print(f"Index B (BGE-M3): {'✓ Success' if success_b else '✗ Failed'}")
    
    # if success_bm25 and success_a and success_b:
    #     print("\n✓ All indices created successfully!")
    #     print("\nMetadata fields preserved:")
    #     print("  • doc_id (document identifier)")
    #     print("  • page (page number)")
    #     print("  • section (section name/heading)")
    #     print("  • url/URL (web address)")
    #     print("  • year (academic year or publication year)")
    #     print("  • source_type (pdf/html)")
    #     print("\nIndices are ready for retrieval with full citation metadata!")
    # else:
    #     print("\n⚠ Some indexes failed to create. Check errors above.")
    #     exit(1)

if __name__ == "__main__":
    main()