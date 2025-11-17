# create_embeddings.py

import json
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


# ========= INPUT / OUTPUT PATHS ========= #

PDF_JSONL = Path("./data/processed/chunks.jsonl")
HTML_JSONL = Path("./data/processed/html_chunks.jsonl")

INDEX_A_DIR = "./data/vectorstores/chroma_index_openAI"
INDEX_B_DIR = "./data/vectorstores/chroma_index_bge_m3"


# ========= FLATTEN METADATA FOR CHROMADB ========= #

def flatten_metadata(item):
    """
    ChromaDB only supports simple types (str, int, float, bool) in metadata.
    Convert complex fields like lists to JSON strings.
    """
    metadata = {}
    
    for key, value in item.items():
        if key == "chunk_text":
            # Skip the text content itself (it goes in the document, not metadata)
            continue
        elif key == "text":
            # Skip the text content itself
            continue
        elif isinstance(value, (list, dict)):
            # Serialize complex types to JSON string
            metadata[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Keep simple types as-is
            metadata[key] = value
        else:
            # Convert anything else to string
            metadata[key] = str(value)
    
    return metadata


# ========= LOAD CHUNKS ========= #

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
    return chunks


# ========= BUILD INDEX A (OpenAI Embeddings) ========= #

def build_index_openai(chunks):
    print("="*60)
    print("Creating Chroma Index A (OpenAI Embeddings)...")
    print("="*60)
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Create the index in batches to avoid timeout issues
        batch_size = 100
        print(f"Processing {len(texts)} chunks in batches of {batch_size}...")
        
        # Create initial vectorstore with first batch
        vectorstore = Chroma.from_texts(
            texts=texts[:batch_size],
            embedding=embeddings,
            metadatas=metadatas[:batch_size],
            persist_directory=INDEX_A_DIR
        )
        
        # Add remaining batches
        for i in range(batch_size, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas
            )
            
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
        print(f"\n✓ Index A saved at: {INDEX_A_DIR}")
        print(f"  Total documents: {len(texts)}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating Index A: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========= BUILD INDEX B (BAAI/BGE-M3) ========= #

def build_index_bge(chunks):
    print("="*60)
    print("Creating Chroma Index B (BAAI/bge-m3 Embeddings)...")
    print("="*60)
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Create the index in batches
        batch_size = 100
        print(f"Processing {len(texts)} chunks in batches of {batch_size}...")
        
        # Create initial vectorstore with first batch
        vectorstore = Chroma.from_texts(
            texts=texts[:batch_size],
            embedding=embeddings,
            metadatas=metadatas[:batch_size],
            persist_directory=INDEX_B_DIR
        )
        
        # Add remaining batches
        for i in range(batch_size, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas
            )
            
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
        print(f"\n✓ Index B saved at: {INDEX_B_DIR}")
        print(f"  Total documents: {len(texts)}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating Index B: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========= MAIN RUNNER ========= #

if __name__ == "__main__":
    print("Starting embedding generation process...\n")
    
    chunks = load_chunks()
    
    if len(chunks) == 0:
        print("No chunks found. Exiting.")
        exit(1)
    
    # Build both indexes
    success_a = build_index_openai(chunks)
    success_b = build_index_bge(chunks)
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Index A (OpenAI): {'✓ Success' if success_a else '✗ Failed'}")
    print(f"Index B (BGE-M3): {'✓ Success' if success_b else '✗ Failed'}")
    
    if success_a and success_b:
        print("\n✓ Embedding generation completed successfully!")
    else:
        print("\n⚠ Some indexes failed to create. Check errors above.")
        exit(1)