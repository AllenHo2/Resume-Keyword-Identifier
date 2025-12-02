import re
import PyPDF2
from typing import Union

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def clean_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize text by lowercasing and optionally removing punctuation.
    
    Args:
        text: Input text to clean
        remove_punctuation: Whether to remove punctuation marks
    
    Returns:
        Cleaned text string
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S*\d+\S*", "", text)
    
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    else:
        text = re.sub(r"[^\w\s\-']", " ", text)
    
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_document(file_path: str, remove_punctuation: bool = True) -> str:
    """
    Process a document (PDF or text file) and return cleaned text.
    
    Args:
        file_path: Path to the document
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        Cleaned text string
    """
    if file_path.lower().endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    
    return clean_text(raw_text, remove_punctuation)
