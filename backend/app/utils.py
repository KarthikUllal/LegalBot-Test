# backend/app/utils.py
import re
import pymupdf

DISCLAIMER = "This application provides general legal information only and is NOT a substitute for professional legal advice."  # to clean extracted text from each pages


def clean_text(s: str) -> str:
    if not s:
        return ""

    # Fix hyphenated line breaks (very common in PDFs)
    s = re.sub(r"-\s*\n", "", s)

    # Normalize multiple spaces
    s = re.sub(r"[ \t]+", " ", s)

    # Preserve paragraph/section breaks – crucial for legal documents
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)  # 3+ newlines → 2
    s = re.sub(r"\n\s*\n", "\n\n", s)        # single newline with spaces → paragraph break

    # Optional: remove page numbers (e.g., lines with just digits)
    lines = s.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.isdigit() and len(stripped) > 0:
            cleaned_lines.append(line)
    
    return "\n\n".join(cleaned_lines).strip()

def extract_clean_text_from_pdf(pdf_path: str) -> str:
    """
    Extract and clean text from PDF while handling PDF-specific issues.
    """
    import pdfplumber

    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # PDF-aware text extraction
            text = page.extract_text()
            if text:
                text_chunks.append(text)

    full_text = "\n".join(text_chunks)
    # Now you might need PDF-specific cleaning, not generic clean_text()
    return clean_pdf_text(full_text)


def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDFs - handles PDF-specific issues.
    """
    # Fix common PDF extraction problems:
    # 1. Remove page numbers/headers
    # 2. Fix hyphenated words across line breaks
    # 3. Handle multi-column text ordering
    # 4. Remove PDF artifacts
    text = re.sub(r"-\n", "", text)  # Fix line-break hyphens
    text = re.sub(r"\n+", "\n", text)  # Reduce multiple newlines
    return text.strip()
