# pip install -U langchain langchain-community pypdf python-dotenv langsmith pytesseract pymupdf Pillow tqdm

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# OCR fallback imports
OCR_AVAILABLE = False
pytesseract = None
fitz = None

try:
    import pytesseract as _pytesseract
    import fitz  # PyMuPDF
    from PIL import Image
    import io
    pytesseract = _pytesseract
    OCR_AVAILABLE = True
    print("✓ OCR libraries (Tesseract + PyMuPDF) initialized successfully")
except ImportError as e:
    print(f"Warning: OCR dependencies not available: {e}")
    print("Install with: pip install pytesseract pymupdf Pillow")

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'NUST Policies Copilot - Preprocessing'

# ===================== CONFIG =====================
INPUT_DIR = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 800  # characters (approximates 200-250 tokens)
CHUNK_OVERLAP_PERCENT = 15  # 15% overlap
CHUNK_OVERLAP = int(CHUNK_SIZE * CHUNK_OVERLAP_PERCENT / 100)

MIN_PAGE_CHARS = 30  # threshold for OCR fallback

# ===================== HELPERS =====================

def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@traceable(name="pdf_page_to_image")
def pdf_page_to_image(pdf_path: Path, page_num: int):
    """
    Convert a specific PDF page to PIL Image using PyMuPDF.
    page_num is 0-indexed.
    """
    if not fitz:
        return None
    
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        
        # Render page to pixmap with higher resolution for better OCR
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for 144 DPI (default is 72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return img
    except Exception as e:
        print(f"    Error converting page {page_num} to image: {e}")
        return None

@traceable(name="load_pdf_with_ocr_fallback")
def load_pdf_with_ocr_fallback(pdf_path: Path) -> List[Document]:
    """
    Load PDF using PyPDFLoader. If any page has < MIN_PAGE_CHARS,
    attempt OCR on that page using Tesseract and replace the text.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
    except Exception as e:
        print(f"  Failed to load PDF {pdf_path.name}: {e}")
        return []
    
    if not OCR_AVAILABLE or not documents or pytesseract is None or fitz is None:
        return documents
    
    # Check each page
    for idx, doc in enumerate(documents):
        if len(doc.page_content.strip()) < MIN_PAGE_CHARS:
            try:
                print(f"    Applying OCR to page {idx + 1}...")
                
                # Convert PDF page to image using PyMuPDF
                img = pdf_page_to_image(pdf_path, idx)
                
                if img:
                    # Perform OCR using Tesseract
                    ocr_text = pytesseract.image_to_string(img)
                    
                    if len(ocr_text.strip()) > MIN_PAGE_CHARS:
                        doc.page_content = ocr_text
                        doc.metadata['ocr_applied'] = True
                        doc.metadata['ocr_engine'] = 'tesseract'
                        print(f"    ✓ OCR extracted {len(ocr_text)} characters")
                    else:
                        print(f"    ✗ OCR extracted insufficient text")
                        
            except Exception as e:
                print(f"  OCR failed for page {idx+1} in {pdf_path.name}: {e}")
    
    return documents

@traceable(name="clean_text")
def clean_text(text: str) -> str:
    """
    Clean and normalize text:
    - Remove excessive whitespace
    - Remove common headers/footers patterns
    - Normalize line breaks
    - Preserve structure (bullets, numbering)
    """
    # Remove page numbers pattern (e.g., "Page 1 of 3")
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove repeated header/footer patterns (3+ identical lines)
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = None
    repeat_count = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and len(stripped) < 100:
            repeat_count += 1
            if repeat_count < 2:
                cleaned_lines.append(line)
        else:
            repeat_count = 0
            cleaned_lines.append(line)
        prev_line = stripped
    
    text = '\n'.join(cleaned_lines)
    
    # Normalize multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove trailing/leading whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text.strip()

@traceable(name="normalize_dates")
def normalize_dates(text: str) -> str:
    """
    Normalize date formats to YYYY-MM-DD where possible.
    Also expand academic year patterns like 2025-26 to 2025-2026.
    """
    # Expand academic year patterns: 2025-26 -> 2025-2026
    def expand_academic_year(match):
        year1 = match.group(1)
        year2_suffix = match.group(0)[-2:]
        return f"{year1}-{year1[:2]}{year2_suffix}"
    
    text = re.sub(r'\b(20\d{2})[-–]\d{2}\b', expand_academic_year, text)
    
    # Common date patterns (this is a simple normalization, expand as needed)
    # DD/MM/YYYY or DD-MM-YYYY -> YYYY-MM-DD
    def normalize_date_match(match):
        try:
            date_str = match.group(0)
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str
        except:
            return match.group(0)
    
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', normalize_date_match, text)
    
    return text

@traceable(name="extract_year_from_filename")
def extract_year_from_filename(filename: str) -> Optional[str]:
    """Extract year from filename. Prioritize 4-digit years."""
    # Look for 4-digit year patterns
    matches = re.findall(r'\b(20\d{2})\b', filename)
    if matches:
        return matches[-1]  # Return most recent year if multiple
    
    # Look for 2-digit year and assume 20xx
    matches = re.findall(r'\b(\d{2})[_-](\d{2})\b', filename)
    if matches:
        return f"20{matches[-1][0]}"
    
    return None

@traceable(name="extract_year_from_text")
def extract_year_from_text(text: str) -> Optional[str]:
    """Extract year from document text (first 1000 chars)."""
    sample = text[:1000]
    
    # Look for academic year patterns like "2025-2026" or "Fall 2025"
    matches = re.findall(r'\b(20\d{2})[–-](20\d{2})\b', sample)
    if matches:
        return matches[0][0]
    
    matches = re.findall(r'\b(20\d{2})\b', sample)
    if matches:
        return matches[0]
    
    return None

@traceable(name="extract_section_heading")
def extract_section_heading(text: str, position: int) -> Optional[str]:
    """
    Extract the nearest heading before the given position.
    Looks for:
    - Roman numerals (I., II., III.)
    - Decimal outlines (1., 1.1, 2.4.3)
    - Uppercase title lines
    """
    before_text = text[:position]
    lines = before_text.split('\n')
    
    # Search backwards for heading patterns
    for line in reversed(lines[-50:]):  # Check last 50 lines
        line = line.strip()
        
        if not line or len(line) > 200:  # Skip empty or too long
            continue
        
        # Roman numerals
        if re.match(r'^[IVX]+\.?\s+[A-Z]', line):
            return line
        
        # Decimal outline
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', line):
            return line
        
        # All uppercase (at least 3 words)
        if line.isupper() and len(line.split()) >= 3:
            return line
        
        # Title Case with common policy keywords
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){2,}', line):
            keywords = ['policy', 'admission', 'refund', 'scholarship', 'rules', 'guidelines', 
                       'requirements', 'procedure', 'registration', 'fee', 'hostel']
            if any(kw in line.lower() for kw in keywords):
                return line
    
    return None

@traceable(name="chunk_document")
def chunk_document(doc: Document, doc_id: str, url: str, year: str) -> List[Dict]:
    """
    Chunk a document using RecursiveCharacterTextSplitter.
    Extract metadata for each chunk.
    """
    text = doc.page_content
    page = doc.metadata.get('page', 1)
    
    # Initialize splitter with sentence-aware separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ', ', ' ', ''],
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    
    results = []
    for chunk_text in chunks:
        # Find position in original text to extract section
        position = text.find(chunk_text[:100])  # Use first 100 chars to locate
        section = extract_section_heading(text, position) if position >= 0 else None
        
        chunk_data = {
            "doc_id": doc_id,
            "section": section,
            "page": int(page),
            "url": url,
            "year": year,
            "chunk_text": chunk_text.strip()
        }
        
        # Add OCR metadata if available
        if doc.metadata.get('ocr_applied'):
            chunk_data['ocr_applied'] = True
            chunk_data['ocr_engine'] = doc.metadata.get('ocr_engine', 'unknown')
        
        results.append(chunk_data)
    
    return results

@traceable(name="process_single_pdf")
def process_single_pdf(pdf_path: Path) -> Tuple[List[Dict], bool]:
    """
    Process a single PDF file and return chunks.
    Returns: (chunks, success)
    """
    print(f"\nProcessing: {pdf_path.name}")
    
    try:
        # Generate doc_id
        doc_id = pdf_path.stem.lower().replace(' ', '_')
        url = f"./data/raw/{pdf_path.name}"
        
        # Extract year from filename first
        year = extract_year_from_filename(pdf_path.name)
        
        # Load PDF with OCR fallback
        documents = load_pdf_with_ocr_fallback(pdf_path)
        
        if not documents:
            print(f"  ✗ No content extracted from {pdf_path.name}")
            return [], False
        
        print(f"  ✓ Loaded {len(documents)} pages")
        
        # If year not in filename, try extracting from first page
        if not year and documents:
            year = extract_year_from_text(documents[0].page_content)
        
        year = year or "unknown"
        
        all_chunks = []
        
        for doc in tqdm(documents, desc="  Chunking pages", leave=False):
            # Clean and normalize
            cleaned_text = clean_text(doc.page_content)
            cleaned_text = normalize_dates(cleaned_text)
            
            # Skip empty pages
            if len(cleaned_text.strip()) < 10:
                continue
            
            # Update document content
            doc.page_content = cleaned_text
            
            # Chunk the document
            chunks = chunk_document(doc, doc_id, url, year)
            all_chunks.extend(chunks)
        
        print(f"  ✓ Generated {len(all_chunks)} chunks")
        return all_chunks, True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return [], False

@traceable(name="process_all_pdfs")
def process_all_pdfs(input_dir: Path, output_file: Path):
    """Process all PDFs in input directory and save to JSONL."""
    pdf_files = sorted(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print("="*50)
    
    all_chunks = []
    successful = 0
    failed = 0
    ocr_pages = 0
    
    for pdf_path in pdf_files:
        chunks, success = process_single_pdf(pdf_path)
        all_chunks.extend(chunks)
        
        # Count OCR pages
        ocr_pages += sum(1 for c in chunks if c.get('ocr_applied', False))
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Write to JSONL
    print("\n" + "="*50)
    print(f"Writing {len(all_chunks)} chunks to {output_file}")
    with output_file.open('w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✓ Preprocessing complete! Output: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total chunks generated: {len(all_chunks)}")
    print(f"Chunks with OCR: {ocr_pages}")
    
    if not all_chunks:
        print("\n⚠ Warning: No chunks were generated!")
        return
    
    # Year distribution
    years = {}
    for chunk in all_chunks:
        y = chunk['year']
        years[y] = years.get(y, 0) + 1
    print(f"\nYear distribution:")
    for year, count in sorted(years.items()):
        print(f"  {year}: {count} chunks")
    
    # Document distribution
    docs = {}
    for chunk in all_chunks:
        d = chunk['doc_id']
        docs[d] = docs.get(d, 0) + 1
    print(f"\nTop 10 documents by chunk count:")
    for doc, count in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {doc}: {count} chunks")
    
    # Average chunk size
    avg_chunk_size = sum(len(c['chunk_text']) for c in all_chunks) / len(all_chunks)
    print(f"\nAverage chunk size: {avg_chunk_size:.0f} characters")

# ===================== MAIN =====================

if __name__ == "__main__":
    ensure_directories()
    process_all_pdfs(INPUT_DIR, OUTPUT_FILE)