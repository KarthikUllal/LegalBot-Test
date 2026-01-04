# backend/app/ingestion.py
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
from typing import List, Dict
import re

try:
    from .config import settings
    from .utils import clean_text
except ImportError:
    from config import settings
    from utils import clean_text

# Strong pattern for section numbers
SECTION_PATTERN = re.compile(r"\b(?:Section|SECTION|Sec\.?)\s+([0-9]+[A-Za-z]?)", re.IGNORECASE)

# Law detection patterns
LAW_PATTERNS = {
    "BNS": re.compile(r"\b(?:Bharatiya\s+Nyaya\s+Sanhita|BNS)\b", re.IGNORECASE),
    "IPC": re.compile(r"\b(?:Indian\s+Penal\s+Code|IPC)\b", re.IGNORECASE),
    "BNSS": re.compile(r"\b(?:Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita|BNSS)\b", re.IGNORECASE),
    "BSA": re.compile(r"\b(?:Bharatiya\s+Sakshya\s+Adhiniyam|BSA)\b", re.IGNORECASE),
    "Consumer Protection": re.compile(r"\b(?:Consumer\s+Protection\s+Act|Consumer\s+Rights)\b", re.IGNORECASE),
    "Domestic Violence": re.compile(r"\b(?:Protection\s+of\s+Women\s+from\s+Domestic\s+Violence|Domestic\s+Violence\s+Act)\b", re.IGNORECASE),
}

# Important legal keywords to boost retrieval
KEYWORD_LIST = [
    "cheating", "fraud", "murder", "theft", "robbery", "assault", "rape", "dowry",
    "harassment", "defective", "unfair trade", "consumer complaint", "rights",
    "punishment", "imprisonment", "fine", "procedure", "complaint", "FIR",
    "evidence", "witness", "trial", "bail", "arrest"
]

def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return clean_text("\n\n".join(pages))

def fetch_html_text(url: str) -> str:
    resp = requests.get(url, timeout=15, headers={
        "User-Agent": "LegalResearchBot/1.0"
    })
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))

def split_into_chunks(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cs = chunk_size or settings.CHUNK_SIZE
    co = overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=[
            "\n\nSection ", "\n\nSECTION ",
            "\n\nChapter ", "\n\nCHAPTER ",
            "\n\nArticle ", "\n\nARTICLE ",
            "\n\nPart ", "\n\nPART ",
            "\n\n", "\n", ". ", " ", ""
        ],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def detect_law_from_text(text: str) -> str:
    """Detect primary law from chunk text"""
    for law_name, pattern in LAW_PATTERNS.items():
        if pattern.search(text):
            return law_name
    return "Unknown"

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract top legal keywords from text"""
    text_lower = text.lower()
    found = [kw for kw in KEYWORD_LIST if kw in text_lower]
    return found[:max_keywords]

def enrich_metadata_with_section(chunks: List[str], base_metadata: Dict) -> List[Dict]:
    """Rich metadata — SAFE for ChromaDB (no lists)"""
    enriched = []
    base_act = base_metadata.get("act", "Unknown")

    for i, chunk in enumerate(chunks):
        # Extract section number
        section_match = SECTION_PATTERN.search(chunk)
        section_num = section_match.group(1) if section_match else None

        # Detect law from text
        detected_law = detect_law_from_text(chunk)
        final_act = detected_law if detected_law != "Unknown" else base_act

        # Extract keywords → join into string (Chroma-safe)
        keywords_list = extract_keywords(chunk)
        keywords_str = ", ".join(keywords_list) if keywords_list else "None"

        meta = {
            **base_metadata,
            "chunk_index": i,
            "act": final_act,
            "section": section_num or "None",
            "has_section": bool(section_num),  # bool is allowed
            "keywords": keywords_str,         # ← String, not list
            "detected_law": detected_law,
            "text_preview": chunk[:300].replace("\n", " "),
        }
        enriched.append(meta)

    return enriched

    return enriched