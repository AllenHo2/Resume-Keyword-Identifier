import re
import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"\S*\d+\S*", "", text)  # Remove words containing numbers (e.g., "Python3", "2023")
    text = re.sub(r"[^\w\s\-']", " ", text)  # Keep letters, spaces, hyphens, apostrophes
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # Iterate through all the pages and extract text
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize cleaned text into words."""
    return word_tokenize(text)

def process_pdf(pdf_path: str) -> List[str]:
    """Extract, clean, and tokenize text from a PDF."""
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text(cleaned_text)
    return tokens

def process_all_pdfs(directory: str) -> Dict[str, List[str]]:
    """Process all PDFs in a directory."""
    results = {}
    pdf_dir = Path(directory)
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        tokens = process_pdf(str(pdf_file))
        results[pdf_file.name] = tokens
    
    return results

# Example usage
if __name__ == "__main__":
    # Process a single PDF
    # tokens = process_pdf("path/to/resume.pdf")
    # print(f"Total tokens: {len(tokens)}")
    
    # Process all PDFs in a directory
    all_tokens = process_all_pdfs("../data/resumes")
    for filename, tokens in all_tokens.items():
        print(f"{filename}: {len(tokens)} tokens")