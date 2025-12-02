# Resume Keyword Extractor

A complete NLP pipeline for extracting and ranking keywords from resumes and job descriptions using TF-IDF scoring. This tool helps identify the most important terms in documents and analyze alignment between resumes and job postings.

## Features

- **PDF and Text Processing**: Extracts text from PDF files using PyPDF2
- **Text Normalization**: Lowercasing, punctuation removal, and cleaning
- **Tokenization**: NLTK-based word tokenization with stopword removal
- **Stemming/Lemmatization**: PorterStemmer or WordNetLemmatizer for term normalization
- **TF-IDF Scoring**: Scikit-learn TfidfVectorizer for keyword importance ranking
- **Keyword Extraction**: Identifies top N keywords ranked by TF-IDF scores
- **Document Comparison**: Analyzes keyword overlap between resumes and job descriptions
- **Validation Testing**: Automated tests to verify extractor functionality

## Pipeline Overview

The keyword extraction pipeline consists of the following steps:

1. **Document Loading**: Read PDF or text files
2. **Text Cleaning**: Normalize text (lowercase, remove URLs, numbers, punctuation)
3. **Tokenization**: Split text into words using NLTK
4. **Stopword Removal**: Filter out common words (the, is, at, etc.)
5. **Stemming**: Reduce words to root form (running → run, studies → studi)
6. **TF-IDF Computation**: Calculate term importance scores
7. **Keyword Ranking**: Sort and extract top N keywords by score
8. **Overlap Analysis**: Compare keywords between documents

## Project Structure

```
resume-keyword-extractor/
│
├── assets/
│   ├── resumes/          # Place resume files here (PDF or TXT)
│   │   └── resume.pdf
│   └── jobs/             # Place job description files here (TXT)
│       └── job1.txt
│
├── src/
│   ├── cleaner.py        # PDF parsing and text normalization
│   ├── tokenizer.py      # NLTK tokenization and stopword removal
│   ├── stemmer.py        # Stemming and lemmatization
│   ├── tfidf_vectorizer.py  # TF-IDF computation
│   ├── keyword_extractor.py # Keyword extraction and ranking
│   ├── validator.py      # Validation tests
│   └── utils.py          # File I/O utilities
│
├── main.py               # Main execution script
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

Install all dependencies:

```bash
pip install nltk scikit-learn PyPDF2 numpy
```

### NLTK Data Downloads

The program will automatically download required NLTK data on first run, but you can manually download them:

```python
import nltk
nltk.download('punkt')           # Tokenizer
nltk.download('stopwords')       # Stopword list
nltk.download('wordnet')         # Lemmatizer
nltk.download('omw-1.4')         # Open Multilingual WordNet
```

Or run in terminal:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### 1. Prepare Your Documents

Place your files in the appropriate directories:
- **Resume**: `assets/resumes/` (PDF or TXT format)
- **Job Description**: `assets/jobs/` (TXT format)

### 2. Run the Extractor

Execute the main script:

```bash
python main.py
```

### 3. Interpret the Output

The program will output:

#### Ranked Keywords
Lists of top keywords with TF-IDF scores for both resume and job description:

```
TOP KEYWORDS - RESUME:
1. python               (score: 0.4523)
2. machine              (score: 0.3891)
3. learn                (score: 0.3654)
4. data                 (score: 0.3421)
...
```

#### Keyword Overlap Analysis
Common keywords between resume and job description:

```
OVERLAPPING KEYWORDS: 8
   python, machine, learn, data, develop, project, model, scikit

MATCH SCORE: 28.57%
   ~ Moderate alignment between resume and job description
```

**Match Score Interpretation:**
- **≥30%**: Strong alignment - resume closely matches job requirements
- **15-29%**: Moderate alignment - some overlap, may need optimization
- **<15%**: Low alignment - consider tailoring resume to job description

#### Validation Tests
Automated tests verify the extractor is working correctly:

```
Test 1: Extractor Returns Keywords - PASS
Test 2: Scores Are Non-Zero - PASS
Test 3: Keyword-Heavy Document Ranking - PASS
Test 4: Multiple Documents Differentiation - PASS
```

## Understanding TF-IDF Scores

**TF-IDF (Term Frequency-Inverse Document Frequency)** measures keyword importance:

- **High Score**: Word appears frequently in document but rarely across all documents
- **Low Score**: Word is either rare in document or very common across all documents

The scores are normalized between 0 and 1, with higher scores indicating more distinctive and important keywords.

## Validation Testing

The validator module (`validator.py`) includes four key tests:

### Test 1: Extractor Returns Keywords
Verifies that the pipeline successfully extracts keywords from sample text.

### Test 2: Scores Are Non-Zero
Ensures all extracted keywords have valid, non-zero TF-IDF scores.

### Test 3: Keyword-Heavy Document Ranking
Tests that repeated important terms are ranked higher (e.g., "python" appearing multiple times).

### Test 4: Multiple Documents Differentiation
Confirms that documents with different content produce distinct keyword sets.

All tests run automatically when executing `main.py`. If any test fails, check your NLTK installation and data downloads.

## Customization

### Adjust Number of Keywords

Modify the `top_n` parameter in `main.py`:

```python
resume_keywords = extract_top_keywords(tfidf_scores[0], top_n=20)  # Extract 20 keywords
```

### Use Lemmatization Instead of Stemming

Change the `use_lemmatizer` parameter:

```python
resume_processed = process_and_extract_keywords(resume_path, use_lemmatizer=True)
```

### Adjust TF-IDF Parameters

Modify parameters in `tfidf_vectorizer.py`:

```python
tfidf_scores = get_all_tfidf_scores(documents, max_features=200)  # More features
```

## Example Output

```
================================================================================
RESUME KEYWORD EXTRACTOR
================================================================================

Resume: resume.pdf
Job Description: job1.txt

--------------------------------------------------------------------------------
PROCESSING DOCUMENTS
--------------------------------------------------------------------------------

Processing: resume.pdf
Processing: job1.txt

--------------------------------------------------------------------------------
COMPUTING TF-IDF SCORES
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
EXTRACTING TOP KEYWORDS
--------------------------------------------------------------------------------

📄 TOP KEYWORDS - RESUME (resume.pdf):
--------------------------------------------------------------------------------
 1. python               (score: 0.3845)
 2. machine              (score: 0.3421)
 3. learn                (score: 0.3198)
 4. data                 (score: 0.2987)
 5. model                (score: 0.2765)
 ...

💼 TOP KEYWORDS - JOB DESCRIPTION (job1.txt):
--------------------------------------------------------------------------------
 1. machine              (score: 0.4123)
 2. learn                (score: 0.3891)
 3. experi               (score: 0.3654)
 4. develop              (score: 0.3421)
 5. python               (score: 0.3198)
 ...

--------------------------------------------------------------------------------
KEYWORD OVERLAP ANALYSIS
--------------------------------------------------------------------------------

🔗 OVERLAPPING KEYWORDS: 7
   python, machine, learn, data, model, develop, engin

📊 MATCH SCORE: 25.93%
   ~ Moderate alignment between resume and job description

================================================================================
RUNNING VALIDATION TESTS
================================================================================

=== Test 1: Extractor Returns Keywords ===
✓ PASS: Extractor returned keywords
  Keywords found: 5

=== Test 2: Scores Are Non-Zero ===
✓ PASS: All scores are non-zero
  Score range: 0.3162 to 0.6325

=== Test 3: Keyword-Heavy Document Ranking ===
✓ PASS: Keyword-heavy terms ranked highly
  Top keywords in heavy doc: ['python', 'develop', 'experi', 'expert', 'skill']

=== Test 4: Multiple Documents Differentiation ===
✓ PASS: Different documents produce different keywords
  Doc1 top keywords: ['machin', 'learn', 'artifici']
  Doc2 top keywords: ['javascript', 'web', 'develop']
  Overlap: 1 keywords

============================================================
VALIDATION RESULTS: 4/4 tests passed
============================================================

================================================================================
PROCESS COMPLETE
================================================================================
```

## Troubleshooting

### ImportError: No module named 'nltk'
Install NLTK: `pip install nltk`

### LookupError: Resource punkt not found
Run NLTK downloads (see Installation section)

### PyPDF2 errors reading PDF
Ensure PDF is not corrupted or encrypted. Try converting to text file first.

### No keywords extracted
Check that your documents contain sufficient text. Very short documents may not produce meaningful keywords.

### All keywords have same score
This may occur with very short documents or single-document analysis. TF-IDF works best with multiple documents.

## License

This project is provided for educational purposes.

## Author

NLP Class Project - Resume Keyword Identifier