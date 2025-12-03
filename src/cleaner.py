import re
import PyPDF2

# Extract text from a PDF file.
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# Clean and preprocess text by lowercasing and removing unwanted elements.
def clean_text(text: str, remove_punctuation: bool = True) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S*\d+\S*", "", text)
    
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    else:
        text = re.sub(r"[^\w\s\-']", " ", text)
    
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Process a document (PDF or text file) and return cleaned text.
def process_document(file_path: str, remove_punctuation: bool = True) -> str:
    if file_path.lower().endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    
    return clean_text(raw_text, remove_punctuation)
