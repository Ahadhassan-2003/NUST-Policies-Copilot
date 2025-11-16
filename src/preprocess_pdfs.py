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

CHUNK_SIZE = 800
CHUNK_OVERLAP_PERCENT = 15
CHUNK_OVERLAP = int(CHUNK_SIZE * CHUNK_OVERLAP_PERCENT / 100)

MIN_PAGE_CHARS = 30

# ===================== PDF-SPECIFIC SECTION MAPPINGS =====================

PDF_SECTION_MAPPINGS = {
    'NUST-HOSTEL-RULES.pdf': [
        'Introduction',
        'Allotment of Hostel Accommodation',
        'Hostel Dues',
        'Attendance',
        'Fine',
        'Guests',
        'Temporary Hostel Allotment',
        'Conveyance/ Driving',
        'Messing',
        'Meal Timings/Dress Code',
        'Conduct',
        'In/Out Timings',
        'Damage to Property',
        'TV Timings',
        'Penalties',
        'Discipline',
        'Inspections',
        'Medical Care',
        'Washerman Services',
        'Indoor Sports',
        'Temporary Vacation of Hostels',
        'Final Vacation of Hostels',
        'Procedure to Vacate the Hostel',
        "Do's and Dont's",
        'Hostel Administration'
    ],
    'FYP_Guidelines_02_11_2017_v1.4.pdf': [
        'INTRODUCTION',
        'DEGREE PROGRAM LEARNING OUTCOMES (PLOs)',
        'OVERVIEW OF FINAL YEAR PROJECT',
        'FYP MILESTONES AND EVALUATION STAGES',
        'Proposal Defense',
        'Mid-Defense/ Design Expo',
        'Final Defense',
        'Open House',
        'GUIDELINES FOR PROJECT SUPERVISION',
        'Tasks expected from supervisors',
        'Project Development Life Cycle',
        'TEAM LEADERSHIP',
        'STUDENTS RESPONSIBILITY',
        'LATE SUBMISSIONS',
        'PLAGIARISM'
    ],
    'MARCOMS-PROSPECTUS-2025-V.5.0-04032025_compressed.pdf': [
        'ABOUT THE UNIVERSITY',
        'Defining Futures',
        'Why Choose NUST?',
        'Location',
        'Who to Contact',
        'International Affairs',
        'Student Affairs',
        'Accreditations',
        'Membership of Quality Assurance Association/Network',
        'NUST CAMPUSES & INSTITUTIONS',
        'School of Electrical Engineering and Computer Science (SEECS)',
        'NUST Business School (NBS)',
        'School of Social Sciences and Humanities (S3H)',
        'School of Chemical and Materials Engineering (SCME)',
        'School of Civil and Environmental Engineering (SCEE)',
        'School of Mechanical and Manufacturing Engineering (SMME)',
        'School of Art, Design and Architecture (SADA)',
        'School of Interdisciplinary Engineering and Sciences (SINES)',
        'College of Aeronautical Engineering (CAE)',
        'Atta-ur-Rahman School of Applied Biosciences (ASAB)',
        'NUST Center for Advanced Studies in Energy (USPCASE)',
        'NUST Institute of Peace and Conflict Studies (NIPCONS)',
        'Center for Peace and Conflict Studies (CPCS)',
        'NUST Institute of Civil Engineering (NICE)',
        'NUST School of Health Sciences (NSHS)',
        'Military College of Signals (MCS)',
        'College of Electrical and Mechanical Engineering (CEME)',
        'Military College of Engineering (MCE)',
        'College of Aeronautical Engineering (CAE)',
        'National Institute of Transportation (NIT)',
        'NUST Islamabad Campus (NIC)',
        'NUST Balochistan Campus (NBC)',
        'FEE AND FUNDING',
        'National Students',
        'Undergraduate Programmes',
        'Postgraduate Programmes',
        'International Students',
        'NUST Entry Test (NET)',
        'Instructions for NET',
        'Specimen Answer Sheet',
        'APPLYING TO NUST'
    ],
    'UG-Student-Handbook.pdf': [
        'The University',
        'Scheme of Studies',
        'Examinations',
        'Academic Standards',
        'Award of Bachelor Degree and Academic Deficiencies (Applicable to all programmes except those specified separately)',
        'Award of Bachelor of Industrial Design & Architecture Degrees and Academic Deficiencies',
        'Academic Provisions & Flexibilities',
        'Issuance of Bachelor Degrees & Transcripts and Award of Medals & Prizes',
        'Clubs & Societies',
        'NUST Social Media Accounts & IT Services',
        'NUST Code of Conduct',
        'Living on Campus',
        'Annexes'
    ],
    'PG-Student-Handbook.pdf': [
        'The University',
        'Salient Academic Regulations: Postgraduate Programmes',
        'Award of Master Degrees and Academic Standards for Master Students (Less Business & Social Sciences)',
        'Award of Master Degree in Business Administration/Executive Master in Business Administration/Social Sciences',
        'Award of Ph.D. Degree and Academic Deficiencies for Ph.D. Students',
        'Academic Provisions & Flexibilities',
        'Clubs & Societies',
        'Services for International Students',
        'NUST Social Media Accounts & IT Services',
        'NUST Code of Conduct',
        'Living on Campus'
    ],
    'PG-Joining-Instructions-Fall-2025.pdf': [
        'Introduction',
        'Immediate Tasks:',
        'Tasks to be done before arrival at SEECS, NUST.',
        'Registration/Documentation (Date/Time/Venue):',
        'Entry in NUST',
        'Location of SEECS:',
        'Student Handbook',
        'Commencement of Clases Fall, 2025 Semester.',
        'SURETY BOND',
        'Medical Fitness Certificate',
        'CERTIFICATE BY THE DOCTOR'
    ],
    'NUST-Need-Based-Scholarship-Form.pdf': [
        'PROVIDING FALSE INFORMATION',
        'INSTRUCTIONS FOR FILLING OUT THE SCHOLARSHIP APPLICATION FORM:',
        "DO's:",
        'DO NOT:'
    ],
    'Check-list-for-PhD-admissions.pdf': [
        'PhD Duration:',
        'Coursework:',
        'Formation of GEC:',
        'Qualifying Examination:',
        'Approval of Thesis Synopsis.',
        'Research Work:',
        'Monitoring of PhD:',
        'Approval of Thesis by GEC.',
        'Publications.',
        'Rights of Research',
        'Selection of Foreign/Local Evaluators:',
        'Thesis Defence.',
        'Refund of Admission Dues UG/PG/PhD Students'
    ],
    'NUST-Academic-Rules-Regarding-Students.pdf': [
        'ACADEMIC STANDARDS FOR AWARD OF DEGREES',
        'AWARD OF BACHELORS\' DEGREE AND ACADEMIC DEFICIENCIES FOR BACHELOR STUDENTS'
    ],
    'Refund-of-Admission-Dues-Undergraduate-Postgraduate-PhD-1.pdf': {
        1: None,  # Page 1 all null
        2: 'GENERAL INSTRUCTIONS'  # Page 2 has this section
    },
    'Sports-Scholarship-Application-Form-2022-23.pdf': None  # All sections null
}

# ===================== HELPERS =====================

def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def identify_pdf_type(filename: str) -> Optional[str]:
    """Identify PDF type based on exact filename matching."""
    
    # Return the exact filename if it exists in our mappings
    if filename in PDF_SECTION_MAPPINGS:
        return filename
    
    return None

@traceable(name="pdf_page_to_image")
def pdf_page_to_image(pdf_path: Path, page_num: int):
    """Convert a specific PDF page to PIL Image using PyMuPDF."""
    if not fitz:
        return None
    
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        return img
    except Exception as e:
        print(f"    Error converting page {page_num} to image: {e}")
        return None

@traceable(name="load_pdf_with_ocr_fallback")
def load_pdf_with_ocr_fallback(pdf_path: Path) -> List[Document]:
    """Load PDF with OCR fallback for poor quality pages."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
    except Exception as e:
        print(f"  Failed to load PDF {pdf_path.name}: {e}")
        return []
    
    if not OCR_AVAILABLE or not documents or pytesseract is None or fitz is None:
        return documents
    
    for idx, doc in enumerate(documents):
        if len(doc.page_content.strip()) < MIN_PAGE_CHARS:
            try:
                print(f"    Applying OCR to page {idx + 1}...")
                img = pdf_page_to_image(pdf_path, idx)
                
                if img:
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
    """Clean and normalize text."""
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
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
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text.strip()

@traceable(name="normalize_dates")
def normalize_dates(text: str) -> str:
    """Normalize date formats."""
    def expand_academic_year(match):
        year1 = match.group(1)
        year2_suffix = match.group(0)[-2:]
        return f"{year1}-{year1[:2]}{year2_suffix}"
    
    text = re.sub(r'\b(20\d{2})[-–]\d{2}\b', expand_academic_year, text)
    
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
    """Extract year from filename."""
    matches = re.findall(r'\b(20\d{2})\b', filename)
    if matches:
        return matches[-1]
    
    matches = re.findall(r'\b(\d{2})[_-](\d{2})\b', filename)
    if matches:
        return f"20{matches[-1][0]}"
    
    return None

@traceable(name="extract_year_from_text")
def extract_year_from_text(text: str) -> Optional[str]:
    """Extract year from document text."""
    sample = text[:1000]
    matches = re.findall(r'\b(20\d{2})[–-](20\d{2})\b', sample)
    if matches:
        return matches[0][0]
    
    matches = re.findall(r'\b(20\d{2})\b', sample)
    if matches:
        return matches[0]
    
    return None

@traceable(name="find_all_section_positions")
def find_all_section_positions(text: str, section_list: List[str]) -> List[Tuple[int, str]]:
    """
    Find all section headings in text and return their positions.
    Returns list of (position, section_name) tuples sorted by position.
    """
    section_positions = []
    
    for section in section_list:
        # Create case-insensitive pattern
        section_words = section.split()
        pattern_parts = [re.escape(word) for word in section_words]
        pattern_str = r'\s+'.join(pattern_parts)
        
        # Look for the section as a heading
        # Try multiple patterns in order of strictness
        patterns = [
            # Pattern 1: Start of line, section, optional colon/period, whitespace or newline
            r'(?:^|\n)\s*' + pattern_str + r'\s*[:\.]?\s*(?=\n|$)',
            # Pattern 2: For table/structured content - section followed by description
            r'(?:^|\n)\s*' + pattern_str + r'\s*[:\.]?\s+',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                # Store the start position and section name
                section_positions.append((match.start(), section))
                break  # Found with this pattern, don't try others
            
            if section_positions and section_positions[-1][1] == section:
                break  # Found this section, move to next
    
    # Sort by position
    section_positions.sort(key=lambda x: x[0])
    
    return section_positions

@traceable(name="find_section_for_chunk")
def find_section_for_chunk(chunk_start_pos: int, section_positions: List[Tuple[int, str]]) -> Optional[str]:
    """
    Find which section a chunk belongs to based on its position.
    Returns the section that starts before or at the chunk position.
    """
    if not section_positions:
        return None
    
    # Find the last section that starts before or at the chunk position
    current_section = None
    for pos, section in section_positions:
        if pos <= chunk_start_pos:
            current_section = section
        else:
            break
    
    return current_section

@traceable(name="chunk_document_with_mapping")
def chunk_document_with_mapping(doc: Document, doc_id: str, url: str, year: str, 
                                pdf_type: Optional[str], all_pages_text: str) -> List[Dict]:
    """
    Chunk a document with PDF-specific section mapping.
    Handles multiple sections per page by tracking positions.
    """
    text = doc.page_content
    page = doc.metadata.get('page', 1)
    
    # Get section list for this PDF type
    section_list = None
    if pdf_type and pdf_type in PDF_SECTION_MAPPINGS:
        section_list = PDF_SECTION_MAPPINGS[pdf_type]
        
        # Handle special case: refund PDF with page-specific mapping
        if isinstance(section_list, dict):
            default_section = section_list.get(page)
            section_list = None  # Don't use position-based matching
        else:
            default_section = None
    else:
        default_section = None
    
    # Handle special case: sports_scholarship - all null
    if section_list is None and pdf_type == 'Sports-Scholarship-Application-Form-2022-23.pdf':
        default_section = None
    
    # Find all section positions in the page text
    section_positions = []
    if section_list and isinstance(section_list, list):
        section_positions = find_all_section_positions(text, section_list)
    
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ', ', ' ', ''],
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    
    results = []
    for chunk_text in chunks:
        # Find where this chunk starts in the page text
        chunk_start = text.find(chunk_text[:min(100, len(chunk_text))])
        
        # Determine section for this chunk
        if default_section is not None:
            section = default_section
        elif section_positions:
            section = find_section_for_chunk(chunk_start if chunk_start >= 0 else 0, section_positions)
        else:
            section = None
        
        chunk_data = {
            "doc_id": doc_id,
            "section": section,
            "page": int(page),
            "url": url,
            "year": year,
            "chunk_text": chunk_text.strip()
        }
        
        if doc.metadata.get('ocr_applied'):
            chunk_data['ocr_applied'] = True
            chunk_data['ocr_engine'] = doc.metadata.get('ocr_engine', 'unknown')
        
        results.append(chunk_data)
    
    return results

@traceable(name="process_single_pdf")
def process_single_pdf(pdf_path: Path) -> Tuple[List[Dict], bool]:
    """Process a single PDF file and return chunks."""
    print(f"\nProcessing: {pdf_path.name}")
    
    try:
        doc_id = pdf_path.stem.lower().replace(' ', '_')
        url = f"./data/raw/{pdf_path.name}"
        
        # Identify PDF type
        pdf_type = identify_pdf_type(pdf_path.name)
        if pdf_type:
            print(f"  ℹ Identified as: {pdf_type}")
            print(f"  ℹ Available sections: {len(PDF_SECTION_MAPPINGS.get(pdf_type, []) or [])}")
        else:
            print(f"  ⚠ PDF type not recognized, using generic extraction")
        
        year = extract_year_from_filename(pdf_path.name)
        
        documents = load_pdf_with_ocr_fallback(pdf_path)
        
        if not documents:
            print(f"  ✗ No content extracted from {pdf_path.name}")
            return [], False
        
        print(f"  ✓ Loaded {len(documents)} pages")
        
        if not year and documents:
            year = extract_year_from_text(documents[0].page_content)
        
        year = year or "unknown"
        
        # Combine all pages text for context
        all_pages_text = '\n\n'.join(doc.page_content for doc in documents)
        
        all_chunks = []
        sections_found = set()
        
        for doc in tqdm(documents, desc="  Chunking pages", leave=False):
            cleaned_text = clean_text(doc.page_content)
            cleaned_text = normalize_dates(cleaned_text)
            
            if len(cleaned_text.strip()) < 10:
                continue
            
            doc.page_content = cleaned_text
            
            chunks = chunk_document_with_mapping(doc, doc_id, url, year, pdf_type, all_pages_text)
            
            # Track sections found
            for chunk in chunks:
                if chunk['section']:
                    sections_found.add(chunk['section'])
            
            all_chunks.extend(chunks)
        
        print(f"  ✓ Generated {len(all_chunks)} chunks")
        print(f"  ✓ Unique sections found: {len(sections_found)}")
        if sections_found:
            print(f"     {list(sections_found)[:5]}...")  # Show first 5 sections
        
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
        
        ocr_pages += sum(1 for c in chunks if c.get('ocr_applied', False))
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*50)
    print(f"Writing {len(all_chunks)} chunks to {output_file}")
    with output_file.open('w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✓ Preprocessing complete! Output: {output_file}")
    
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
    
    years = {}
    for chunk in all_chunks:
        y = chunk['year']
        years[y] = years.get(y, 0) + 1
    print(f"\nYear distribution:")
    for year, count in sorted(years.items()):
        print(f"  {year}: {count} chunks")
    
    docs = {}
    for chunk in all_chunks:
        d = chunk['doc_id']
        docs[d] = docs.get(d, 0) + 1
    print(f"\nTop 10 documents by chunk count:")
    for doc, count in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {doc}: {count} chunks")
    
    avg_chunk_size = sum(len(c['chunk_text']) for c in all_chunks) / len(all_chunks)
    print(f"\nAverage chunk size: {avg_chunk_size:.0f} characters")

if __name__ == "__main__":
    ensure_directories()
    process_all_pdfs(INPUT_DIR, OUTPUT_FILE)