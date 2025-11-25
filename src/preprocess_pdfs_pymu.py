# pip install -U langchain langchain-community python-dotenv langsmith pytesseract pymupdf Pillow tqdm

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from langsmith import traceable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# PyMuPDF imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("✓ PyMuPDF initialized successfully")
except ImportError as e:
    print(f"Error: PyMuPDF not available: {e}")
    print("Install with: pip install pymupdf")
    PYMUPDF_AVAILABLE = False
    exit(1)

# OCR fallback imports
OCR_AVAILABLE = False
pytesseract = None

try:
    import pytesseract as _pytesseract
    from PIL import Image
    import io
    pytesseract = _pytesseract
    OCR_AVAILABLE = True
    print("✓ OCR libraries (Tesseract) initialized successfully")
except ImportError as e:
    print(f"Warning: OCR dependencies not available: {e}")
    print("Install with: pip install pytesseract Pillow")

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'NUST Policies Copilot'

# ===================== CONFIG =====================
INPUT_DIR = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")
OUTPUT_FILE = OUTPUT_DIR / "pdf_chunks.jsonl"

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
        '1.1 INTRODUCTION',
        ' 1.2 DEGREE PROGRAM LEARNING OUTCOMES (PLOs)',
        '1.3 OVERVIEW OF FINAL YEAR PROJECT',
        '1.4 FYP MILESTONES AND EVALUATION STAGES',
        '1.5 GUIDELINES FOR PROJECT SUPERVISION',
        ' 1.6 TEAM LEADERSHIP',
        ' 1.7 STUDENTS RESPONSIBILITY',
        '1.8 LATE SUBMISSIONS',
        '1.9 PLAGIARISM',
        'FYP Proposal Document Template',
        'FYP Proposal Defense Evaluation Form and Rubrics ',
        'FYP Proposal Defence Evaluation Form ',
        'FYP SRS Document Template',
        'SRS Evaluation Form',
        'Rubrics for Evaluation of FYP Design Expo Poster',
        'FYP Design Expo Poster Evaluation Form',
        'Rubrics for Evaluation of FYP Mid Defence Presentation',
        'FYP Mid Defence Evaluation Form',
        'FYP Final Report Template',
        'Rubrics for Evaluation of FYP Report',
        'FYP Report Evaluation Form',
        'Rubrics for Evaluation of FYP Demonstration',
        'FYP Demonstration Evaluation Form',
        'Rubrics for Evaluation of FYP Defence Oral Presentation',
        'FYP Defence Oral Presentation Evaluation Form',
        'PROJECT EVALUATION FORM (Open House)',
        'References'
        
    ],
    'MARCOMS-PROSPECTUS-2025-V.5.0-04032025_compressed.pdf': [
        'ABOUT THE UNIVERSITY',
        
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
        'Scheme of Studies, Examinations, and Academic Standards for Award of Degrees ',
        'Award of Bachelor Degree and Academic Deficiencies (Applicable to all programmes except those specified separately)',
        'Award of Bachelor of Industrial Design & Architecture Degrees and Academic Deficiencies',
        'Award of Bachelors Degree in Management/Social Sciences and Academic Deficiencies',
        'Issuance of Bachelor Degrees & Transcripts and Award of Medals & Prizes',
        'Academic Provisions & Flexibilities',
        'Clubs & Societies',
        'NUST Social Media Accounts & IT Services',
        'NUST Code of Conduct',
        'Living on Campus',
        'Dress Norms',
        'Dining Ettiquette',
    ],
    'PG-Student-Handbook.pdf': [
        'Chapter 1: The University',
        'Chapter 2: Salient Academic Regulations: Postgraduate Programmes',
        'Chapter 3: Award of Master Degrees and Academic Standards for Master Students (Less Business & Social Sciences)',
        'Chapter 4: Award of Master Degree in Business Administration/Executive Master in Business Administration/Social Sciences',
        'Chapter 5: Award of Ph.D. Degree and Academic Deficiencies for Ph.D. Students',
        'Chapter 6: Academic Provisions & Flexibilities',
        'Chapter 7: Clubs & Societies',
        'Chapter 8: Services for International Students',
        'Chapter 9: NUST Social Media Accounts & IT Services',
        'Chapter 10: NUST Code of Conduct',
        'Chapter 11: Living on Campus',
        'List of Master & Ph.D. Programmes',
        'Re-Checking of Papers',
        'Hostel Allotment Policy'


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
        1: None,
        2: 'GENERAL INSTRUCTIONS'
    },
    'Sports-Scholarship-Application-Form-2022-23.pdf': None
}

# ===================== HELPERS =====================

def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def identify_pdf_type(filename: str) -> Optional[str]:
    """Identify PDF type based on exact filename matching."""
    if filename in PDF_SECTION_MAPPINGS:
        return filename
    return None

@traceable(name="pdf_page_to_image")
def pdf_page_to_image(page):
    """Convert a PyMuPDF page object to PIL Image."""
    if not OCR_AVAILABLE or pytesseract is None:
        return None
    
    try:
        # Render page at 2x resolution for better OCR
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        print(f"    Error converting page to image: {e}")
        return None

@traceable(name="extract_text_from_page")
def extract_text_from_page(page, page_num: int, apply_ocr: bool = False) -> Tuple[str, Dict]:
    """
    Extract text from a PyMuPDF page object.
    Returns (text, metadata).
    """
    metadata = {
        'page': page_num + 1,  # 1-indexed for user-friendliness
        'ocr_applied': False
    }
    
    # Try native text extraction first
    text = page.get_text("text")
    
    # If text is too short and OCR is available, try OCR
    if len(text.strip()) < MIN_PAGE_CHARS and apply_ocr and OCR_AVAILABLE:
        try:
            print(f"    Applying OCR to page {page_num + 1}...")
            img = pdf_page_to_image(page)
            
            if img and pytesseract:
                ocr_text = pytesseract.image_to_string(img)
                if len(ocr_text.strip()) > MIN_PAGE_CHARS:
                    text = ocr_text
                    metadata['ocr_applied'] = True
                    metadata['ocr_engine'] = 'tesseract'
                    print(f"    ✓ OCR extracted {len(ocr_text)} characters")
                else:
                    print(f"    ✗ OCR extracted insufficient text")
        except Exception as e:
            print(f"    OCR failed for page {page_num + 1}: {e}")
    
    return text, metadata

@traceable(name="load_pdf_with_pymupdf")
def load_pdf_with_pymupdf(pdf_path: Path, apply_ocr: bool = True) -> List[Document]:
    """
    Load PDF using PyMuPDF with optional OCR fallback.
    Returns list of Document objects (one per page).
    """
    documents = []
    
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text, metadata = extract_text_from_page(page, page_num, apply_ocr)
            
            # Add source information
            metadata['source'] = str(pdf_path)
            
            # Create Document object
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        doc.close()
        
    except Exception as e:
        print(f"  Failed to load PDF {pdf_path.name}: {e}")
        return []
    
    return documents

@traceable(name="clean_text")
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove repeated lines (common in PDFs)
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
    
    # Normalize whitespace
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

@traceable(name="find_section_boundaries")
def find_section_boundaries(full_text: str, section_list: List[str]) -> List[Tuple[int, str]]:
    """
    Find all section boundaries in the entire document text.
    Returns sorted list of (position, section_name) tuples.
    """
    boundaries = []
    
    for section in section_list:
        # Create case-insensitive pattern matching all words in section
        section_words = section.split()
        
        # Escape special regex characters in each word
        escaped_words = [re.escape(word) for word in section_words]
        
        # Create pattern: flexible whitespace between words, optional punctuation after
        pattern = r'\s+'.join(escaped_words)
        full_pattern = r'(?:^|\n)\s*(' + pattern + r')\s*[:\.]?'
        
        # Find all occurrences
        for match in re.finditer(full_pattern, full_text, re.IGNORECASE | re.MULTILINE):
            boundaries.append((match.start(), section))
    
    # Sort by position
    boundaries.sort(key=lambda x: x[0])
    
    return boundaries

@traceable(name="get_section_for_position")
def get_section_for_position(position: int, boundaries: List[Tuple[int, str]]) -> Optional[str]:
    """
    Get the active section at a given position in the text.
    Returns the most recent section that starts before or at this position.
    """
    if not boundaries:
        return None
    
    current_section = None
    for bound_pos, section_name in boundaries:
        if bound_pos <= position:
            current_section = section_name
        else:
            break
    
    return current_section

@traceable(name="chunk_document_robust")
def chunk_document_robust(doc: Document, doc_id: str, url: str, year: str, 
                          pdf_type: Optional[str], section_boundaries: List[Tuple[int, str]],
                          full_text: str, doc_start_position: int) -> List[Dict]:
    """
    Chunk a document with robust section mapping.
    """
    text = doc.page_content
    page = doc.metadata.get('page', 1)
    
    # Handle special cases
    if pdf_type:
        section_config = PDF_SECTION_MAPPINGS.get(pdf_type)
        
        # Refund PDF: page-specific sections
        if isinstance(section_config, dict):
            default_section = section_config.get(page)
            use_boundaries = False
        # Sports Scholarship: all null
        elif section_config is None:
            default_section = None
            use_boundaries = False
        else:
            default_section = None
            use_boundaries = True
    else:
        default_section = None
        use_boundaries = False
    
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
        # Find where this chunk starts in the full document
        chunk_preview = chunk_text[:min(100, len(chunk_text))]
        
        # Find in page text first
        local_pos = text.find(chunk_preview)
        
        # Calculate absolute position in full document
        if local_pos >= 0:
            absolute_pos = doc_start_position + local_pos
        else:
            absolute_pos = doc_start_position
        
        # Determine section
        if not use_boundaries:
            section = default_section
        else:
            section = get_section_for_position(absolute_pos, section_boundaries)
        
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
            section_config = PDF_SECTION_MAPPINGS.get(pdf_type)
            if isinstance(section_config, list):
                print(f"  ℹ Section list has {len(section_config)} sections")
            elif isinstance(section_config, dict):
                print(f"  ℹ Page-specific sections configured")
            elif section_config is None:
                print(f"  ℹ All sections will be null")
        else:
            print(f"  ⚠ PDF type not recognized")
        
        year = extract_year_from_filename(pdf_path.name)
        
        # Load PDF with PyMuPDF
        documents = load_pdf_with_pymupdf(pdf_path, apply_ocr=OCR_AVAILABLE)
        
        if not documents:
            print(f"  ✗ No content extracted from {pdf_path.name}")
            return [], False
        
        print(f"  ✓ Loaded {len(documents)} pages")
        
        if not year and documents:
            year = extract_year_from_text(documents[0].page_content)
        
        year = year or "unknown"
        
        # Clean all documents first
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.page_content = normalize_dates(doc.page_content)
        
        # Combine all pages into full text
        full_text = '\n\n'.join(doc.page_content for doc in documents)
        
        # Find all section boundaries in the full document
        section_boundaries = []
        if pdf_type:
            section_config = PDF_SECTION_MAPPINGS.get(pdf_type)
            if isinstance(section_config, list):
                section_boundaries = find_section_boundaries(full_text, section_config)
                print(f"  ✓ Found {len(section_boundaries)} section boundaries")
        
        # Track document positions in full text
        current_position = 0
        doc_positions = []
        for doc in documents:
            doc_positions.append(current_position)
            current_position += len(doc.page_content) + 2  # +2 for '\n\n'
        
        # Process each document
        all_chunks = []
        sections_found = set()
        
        for idx, doc in enumerate(tqdm(documents, desc="  Chunking pages", leave=False)):
            if len(doc.page_content.strip()) < 10:
                continue
            
            chunks = chunk_document_robust(
                doc, doc_id, url, year, pdf_type, 
                section_boundaries, full_text, doc_positions[idx]
            )
            
            for chunk in chunks:
                if chunk['section']:
                    sections_found.add(chunk['section'])
            
            all_chunks.extend(chunks)
        
        # Special handling for specific PDFs
        
        # 1. Need-Based-Scholarship-Form: assign all to INSTRUCTIONS section
        if pdf_path.name == 'NUST-Need-Based-Scholarship-Form.pdf':
            print(f"  ℹ Applying special rule: assigning all chunks to 'INSTRUCTIONS FOR FILLING OUT THE SCHOLARSHIP APPLICATION FORM:'")
            for chunk in all_chunks:
                chunk['section'] = 'INSTRUCTIONS FOR FILLING OUT THE SCHOLARSHIP APPLICATION FORM:'
            sections_found = {'INSTRUCTIONS FOR FILLING OUT THE SCHOLARSHIP APPLICATION FORM:'}
        
        # 2. Check-list-for-PhD: if no sections found, assign fallback
        if pdf_path.name == 'Check-list-for-PhD-admissions.pdf':
            if not sections_found or all(chunk['section'] is None for chunk in all_chunks):
                print(f"  ℹ No sections detected, applying fallback: 'Check List – Information Regarding PhD Studies'")
                for chunk in all_chunks:
                    chunk['section'] = 'Check List – Information Regarding PhD Studies'
                sections_found = {'Check List – Information Regarding PhD Studies'}
        
        print(f"  ✓ Generated {len(all_chunks)} chunks")
        print(f"  ✓ Unique sections found: {len(sections_found)}")
        if sections_found:
            print(f"     Sections: {list(sections_found)[:3]}...")
        
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