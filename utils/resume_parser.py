"""
resume_parser.py
Handles PDF/text resume extraction and structured NLP parsing.
"""

import re
import pdfplumber
import spacy
import nltk
from pathlib import Path

# Download NLTK data on first use
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file) -> str:
    """Extract raw text from an uploaded PDF file object or file path."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")
    return text.strip()


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from a Streamlit UploadedFile object (PDF or TXT)."""
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def extract_emails(text: str) -> list[str]:
    """Extract email addresses using regex."""
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)


def extract_phone_numbers(text: str) -> list[str]:
    """Extract phone numbers using regex."""
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    return re.findall(pattern, text)


def extract_urls(text: str) -> list[str]:
    """Extract LinkedIn, GitHub, portfolio URLs."""
    return re.findall(r"https?://[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+", text)


def extract_named_entities(text: str) -> dict:
    """Use spaCy NER to extract named entities (persons, orgs, locations, dates)."""
    doc = nlp(text[:50000])  # limit to avoid memory issues
    entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    # Deduplicate
    return {k: list(set(v)) for k, v in entities.items()}


def estimate_years_of_experience(text: str) -> int:
    """Roughly estimate years of experience from date patterns in text."""
    year_pattern = re.findall(r"\b(19|20)\d{2}\b", text)
    if len(year_pattern) >= 2:
        years = [int(y) for y in year_pattern]
        span = max(years) - min(years)
        return min(span, 30)  # cap at 30 years
    return 0


def count_sections(text: str) -> dict:
    """Detect common resume section headers and return their presence."""
    sections = {
        "education": bool(re.search(r"\b(education|academic|degree|university|college)\b", text, re.I)),
        "experience": bool(re.search(r"\b(experience|employment|work history|career)\b", text, re.I)),
        "skills": bool(re.search(r"\b(skills|technologies|tools|competencies)\b", text, re.I)),
        "projects": bool(re.search(r"\b(projects|portfolio|case studies)\b", text, re.I)),
        "certifications": bool(re.search(r"\b(certifications|certificates|credentials|courses)\b", text, re.I)),
        "achievements": bool(re.search(r"\b(achievements|awards|honors|publications)\b", text, re.I)),
    }
    return sections


def parse_resume(text: str) -> dict:
    """
    Run the full traditional NLP parsing pipeline on resume text.
    Returns a structured dict of extracted metadata.
    """
    cleaned = clean_text(text)
    return {
        "raw_text": cleaned,
        "emails": extract_emails(cleaned),
        "phones": extract_phone_numbers(cleaned),
        "urls": extract_urls(cleaned),
        "entities": extract_named_entities(cleaned),
        "estimated_experience_years": estimate_years_of_experience(cleaned),
        "sections_detected": count_sections(cleaned),
    }
