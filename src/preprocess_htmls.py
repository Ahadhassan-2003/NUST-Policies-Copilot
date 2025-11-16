import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

from langsmith import traceable
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'NUST Policies Copilot'

ROOT_URLS = [
    "https://nust.edu.pk/admissions/scholarships",
    "https://nust.edu.pk/admissions/fee-structure"
]

SEECS_URLS = [
    "https://seecs.nust.edu.pk/program/bachelor-of-science-in-computer-science-2025-and-onwards",
    "https://seecs.nust.edu.pk/contact/"
]

SPECIAL_DEEP_CRAWL = {
    "https://nust.edu.pk/admissions/fee-structure/hostel-accommodation/": [
        "https://campuslife.nust.edu.pk/facility-and-amenity/housing-and-dining/"
    ]
}

OUTPUT_PATH = Path("data/processed/html_chunks.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

doc_id_counter = 0
processed_urls = set()


@traceable(name="fetch_html")
async def fetch_and_parse(url: str) -> Optional[str]:
    try:
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        if docs and len(docs) > 0:
            return docs[0].page_content
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


@traceable(name="extract_main_content")
def extract_main_content(html: str, url: str) -> Dict[str, any]:
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in soup(['script', 'style', 'svg', 'header', 'footer', 'nav']):
        tag.decompose()
    
    h1 = soup.find('h1')
    section_title = h1.get_text(strip=True) if h1 else None
    
    if not section_title:
        breadcrumb_div = soup.find('div', class_='site-bedcrumb')
        if breadcrumb_div:
            spans = breadcrumb_div.find_all('span')
            if spans:
                last_text = spans[-1].get_text(strip=True)
                if last_text:
                    section_title = last_text
    
    if not section_title:
        url_parts = url.rstrip('/').split('/')
        if url_parts:
            section_title = url_parts[-1].replace('-', ' ').title()
    
    if not section_title:
        section_title = "Untitled"
    
    for elem in soup.find_all(class_=re.compile(r'(breadcrumb|quick-nav|sidebar|widget)', re.I)):
        elem.decompose()
    
    main_content = soup.find('div', class_='col-md-8')
    if not main_content:
        main_content = soup.find('div', id='primary')
    if not main_content:
        main_content = soup.find('main')
    if not main_content:
        main_content = soup.find('article')
    if not main_content:
        main_content = soup
    
    links = []
    for a_tag in main_content.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(url, href)
        if full_url.startswith('http'):
            links.append({
                'text': a_tag.get_text(strip=True),
                'url': full_url
            })
    
    if 'seecs.nust.edu.pk/contact' in url:
        text_content = extract_contact_info(main_content)
    elif 'seecs.nust.edu.pk/program' in url:
        text_content = extract_curriculum_info(main_content)
        links = []
    else:
        text_content = main_content.get_text(separator='\n', strip=True)
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    
    return {
        'section': section_title,
        'text': text_content,
        'links': links
    }


@traceable(name="extract_contact_info")
def extract_contact_info(content) -> str:
    output = []
    
    h3_title = content.find('h3')
    if h3_title:
        output.append(h3_title.get_text(strip=True))
        output.append('')
    
    contact_items = content.find_all('div', class_='general-item')
    
    for item in contact_items:
        title_elem = item.find('h5', id='general-title')
        name_elem = item.find('h6', id='general-name')
        phone_elem = item.find('p', id='general-phone')
        email_elem = item.find('p', id='general-email')
        address_elem = item.find('p', id='general-address')
        
        if title_elem:
            output.append(f"Title: {title_elem.get_text(strip=True)}")
        if name_elem:
            output.append(f"Name: {name_elem.get_text(strip=True)}")
        if phone_elem:
            phone_text = phone_elem.get_text(strip=True)
            output.append(phone_text)
        if email_elem:
            email_text = email_elem.get_text(strip=True)
            output.append(email_text)
        if address_elem:
            output.append(f"Address: {address_elem.get_text(strip=True)}")
        
        output.append('')
    
    return '\n'.join(output)


@traceable(name="extract_curriculum_info")
def extract_curriculum_info(content) -> str:
    output = []
    
    program_heading = content.find('h1', class_='program-heading')
    if program_heading:
        output.append(program_heading.get_text(strip=True))
        output.append('')
    
    desc = content.find('div', class_='progame_description')
    if desc:
        desc_text = desc.get_text(strip=True)
        if desc_text:
            output.append(desc_text)
            output.append('')
    
    curriculum_section = content.find('div', id='course_curriculum')
    if curriculum_section:
        h4 = curriculum_section.find('h4')
        if h4:
            output.append(h4.get_text(strip=True))
            output.append('')
        
        tables = curriculum_section.find_all('table', class_='table')
        
        for table in tables:
            thead = table.find('thead')
            if thead:
                header_rows = thead.find_all('tr')
                for row in header_rows:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) == 1 and 'Semester' in cells[0].get_text():
                        output.append(f"\n{cells[0].get_text(strip=True)}")
                        output.append('-' * 50)
                    elif len(cells) > 1:
                        headers = [cell.get_text(strip=True) for cell in cells]
                        output.append(' | '.join(headers))
                        output.append('-' * 80)
            
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    cell_texts = []
                    
                    for idx, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        
                        content_link = cell.find('a', href=True)
                        if content_link and content_link.get('href'):
                            href = content_link.get('href')
                            if href.startswith('http'):
                                cell_text = f"{cell_text} [Content Link: {href}]"
                        
                        cell_texts.append(cell_text)
                    
                    output.append(' | '.join(cell_texts))
            
            output.append('')
    
    return '\n'.join(output)


@traceable(name="extract_year")
def extract_year(text: str) -> Optional[str]:
    patterns = [
        r'\b(20\d{2}[-–—]\d{2})\b',
        r'\b(20\d{2}[-–—]20\d{2})\b',
        r'\b(20\d{2})\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None


@traceable(name="chunk_text")
def chunk_text(docs: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=int(800 * 0.15),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    
    for doc in docs:
        text = doc['text']
        splits = splitter.split_text(text)
        
        for i, split in enumerate(splits):
            chunk = {
                'doc_id': doc['doc_id'],
                'section': doc['section'],
                'page': 1,
                'URL': doc['URL'],
                'year': doc['year'],
                'text': split,
                'links': doc.get('links', []) if i == 0 else []
            }
            chunks.append(chunk)
    
    return chunks


@traceable(name="save_jsonl")
def save_jsonl(chunks: List[Dict], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Saved {len(chunks)} chunks to {path}")


@traceable(name="extract_links_from_content")
def extract_links_from_content(links: List[Dict], base_url: str) -> List[str]:
    extracted = []
    base_domain = urlparse(base_url).netloc
    
    for link in links:
        url = link['url']
        parsed = urlparse(url)
        
        if parsed.netloc == base_domain or 'nust.edu.pk' in parsed.netloc:
            if url not in processed_urls:
                extracted.append(url)
    
    return list(set(extracted))


@traceable(name="process_page")
async def process_page(url: str, depth: int = 0, is_seecs: bool = False) -> List[Dict]:
    global doc_id_counter, processed_urls
    
    if url in processed_urls:
        return []
    
    processed_urls.add(url)
    print(f"Processing: {url} (depth={depth}, seecs={is_seecs})")
    
    html = await fetch_and_parse(url)
    if not html:
        print(f"Failed to fetch {url}")
        return []
    
    content = extract_main_content(html, url)
    
    if not content['text']:
        print(f"No content found at {url}")
        return []
    
    doc_id_counter += 1
    
    doc = {
        'doc_id': doc_id_counter,
        'section': content['section'],
        'URL': url,
        'year': extract_year(content['text']),
        'text': content['text'],
        'links': content['links']
    }
    
    docs = [doc]
    
    if not is_seecs and depth == 0:
        child_links = extract_links_from_content(content['links'], url)
        for child_url in child_links:
            await asyncio.sleep(0.5)
            child_docs = await process_page(child_url, depth=1, is_seecs=False)
            docs.extend(child_docs)
    
    if url in SPECIAL_DEEP_CRAWL:
        special_urls = SPECIAL_DEEP_CRAWL[url]
        for special_url in special_urls:
            if special_url not in processed_urls:
                await asyncio.sleep(0.5)
                special_docs = await process_page(special_url, depth=2, is_seecs=False)
                docs.extend(special_docs)
    
    return docs


@traceable(name="html_preprocessing_pipeline")
async def main():
    print("Starting HTML preprocessing pipeline...")
    print(f"Root URLs: {ROOT_URLS}")
    print(f"SEECS URLs: {SEECS_URLS}")
    
    all_docs = []
    
    for root_url in ROOT_URLS:
        print(f"\nProcessing root: {root_url}")
        docs = await process_page(root_url, depth=0, is_seecs=False)
        all_docs.extend(docs)
        await asyncio.sleep(1)
    
    for seecs_url in SEECS_URLS:
        print(f"\nProcessing SEECS page (0 depth): {seecs_url}")
        docs = await process_page(seecs_url, depth=0, is_seecs=True)
        all_docs.extend(docs)
        await asyncio.sleep(1)
    
    print(f"\nTotal documents extracted: {len(all_docs)}")
    
    print("\nChunking documents...")
    chunks = chunk_text(all_docs)
    
    print(f"Total chunks created: {len(chunks)}")
    
    save_jsonl(chunks, OUTPUT_PATH)
    
    print("\nPreprocessing complete!")
    return chunks


if __name__ == "__main__":
    asyncio.run(main())