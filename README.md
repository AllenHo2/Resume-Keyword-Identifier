# Resume Keyword Extractor

A complete NLP pipeline for extracting and ranking keywords from resumes and job descriptions using TF-IDF scoring. This tool helps identify the most important terms in documents and analyze alignment between resumes and job postings.

## Features

- **PDF and Text Processing**: Extracts text from PDF files using PyPDF2
- **Abbreviation Handling**: Automatically normalizes B.S., BS, Bachelors → bachelor's degree
- **Text Normalization**: Lowercasing, punctuation removal, and cleaning
- **Smart Tokenization**: NLTK-based word tokenization with stopword removal
- **POS Filtering**: Removes irrelevant verbs, adjectives, adverbs, and time-related words (months)
- **CS Keyword Focus**: Extracts only Computer Science and technology-related terms
- **Smart Stemming**: Groups similar words (developer, developing, developed) but returns readable full forms
- **TF-IDF Scoring**: Scikit-learn TfidfVectorizer for keyword importance ranking
- **N-gram Support**: Captures multi-word phrases (unigrams, bigrams, trigrams)
- **Cosine Similarity**: Measures resume-job alignment with detailed breakdown
- **Full Word Output**: Returns complete words (python, machine learning) not stems (machin, learn)
- **Smart Overlap Detection**: Handles plurals and variations when comparing documents

## Pipeline Overview

The multi-resume keyword analysis pipeline consists of the following steps:

1. **Multiple Resume Loading**: Read all resumes from assets/resumes/ (PDF or text files)
2. **Abbreviation Normalization**: Convert common abbreviations (B.S., BS → bachelor's degree)
3. **Text Cleaning**: Normalize text (lowercase, remove URLs, numbers, punctuation)
4. **Tokenization**: Split text into words using NLTK
5. **Stopword Removal**: Filter out common words (the, is, at, etc.)
6. **POS Filtering**: Remove irrelevant verbs, adjectives, adverbs, and months
7. **CS Keyword Extraction**: Extract only Computer Science and technical terms
8. **TF-IDF Computation with N-grams and Stemming**: Calculate term importance scores across ALL resumes
   - Stemming: Groups similar words (developer, developing, developed) but returns full forms
   - Unigrams: single words (python, docker, aws)
   - Bigrams: 2-word phrases (machine learning, data science)
   - Trigrams: 3-word phrases (natural language processing, deep neural networks)
9. **Frequency Analysis**: Identify most common keywords across all resumes
10. **Cosine Similarity Analysis**: Measure how similar each resume is to the job description (0-100%)
11. **Job Matching**: Check which common keywords appear in the job description
12. **Recommendations**: Suggest high-priority keywords to include in your resume

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
│   ├── similarity.py     # Cosine similarity computation
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
- **Resumes**: `assets/resumes/` (PDF or TXT format) - can include multiple resume files
- **Job Description**: `assets/jobs/` (TXT format) - one job description to analyze against

### 2. Run the Extractor

Execute the main script:

```bash
python main.py
```

### 3. Interpret the Output

The program will output:

#### Step 1-3: Resume Processing
Extracts keywords from all resumes using TF-IDF:

```
📁 Found 4 resume(s)
Processing: Resume1.pdf
Processing: Resume2.pdf
Processing: Resume3.pdf
Processing: Resume4.pdf
```

#### Step 4: Keyword Frequency Analysis
Shows the most common CS keywords across all resumes:

```
� TOP COMPUTER SCIENCE KEYWORDS ACROSS ALL RESUMES:
Rank   Keyword                        Frequency    Avg Score   
--------------------------------------------------------------------------------
1      python                         4/4 resumes  0.4523
2      machine learning               4/4 resumes  0.3891
3      docker                         3/4 resumes  0.3654
4      tensorflow                     3/4 resumes  0.3421
5      aws                            3/4 resumes  0.3198
...
```

#### Step 6: Match Analysis
Shows which common keywords appear in the job description:

```
✅ TOP KEYWORDS FOUND IN JOB DESCRIPTION: 18/30
1. python                        (4/4 resumes)    ✓
2. machine learning              (4/4 resumes)    ✓
3. docker                        (3/4 resumes)    ✓
...

❌ TOP KEYWORDS NOT FOUND IN JOB DESCRIPTION: 12/30
1. react                         (2/4 resumes)    ✗
2. mongodb                       (2/4 resumes)    ✗
...

📊 KEYWORD MATCH RATE: 60.0% (18/30 keywords)
   ✓ Excellent match! Common resume keywords align well with job requirements.
```

#### Step 6: Cosine Similarity Analysis
Measures how similar each resume is to the job description:

```
STEP 6: COMPUTING COSINE SIMILARITY
Measuring how similar each resume is to the job description...

Resume                                   Similarity    Match Quality
--------------------------------------------------------------------------------
john_doe_resume.pdf                       78.34%      Good match - strong alignment
jane_smith_resume.pdf                     82.15%      Excellent match - very high alignment
alex_jones_resume.pdf                     65.42%      Good match - strong alignment
sam_wilson_resume.pdf                     71.89%      Good match - strong alignment

📊 TOP CONTRIBUTING KEYWORDS TO SIMILARITY:

john_doe_resume.pdf:
  Rank   Keyword                   Resume    Job       Contribution
  ----------------------------------------------------------------------
  1      machine learning          0.4523    0.4812    0.2176
  2      python                    0.4201    0.4356    0.1830
  3      deep learning             0.3845    0.3978    0.1530
  4      tensorflow                0.3512    0.3201    0.1124
  5      neural networks           0.3201    0.3089    0.0989
  ...
```

**Similarity Score Interpretation:**
- **80-100%**: Excellent match - very high alignment with job requirements
- **60-79%**: Good match - strong alignment, competitive candidate
- **40-59%**: Moderate match - some alignment but gaps exist
- **20-39%**: Weak match - limited alignment, significant gaps
- **0-19%**: Poor match - minimal alignment, major mismatch

#### Step 7: Job Matching
Shows which keywords appear in the job description:

```
STEP 7: MATCHING TOP KEYWORDS WITH JOB DESCRIPTION
```

(Previous Step 6 content)

#### Step 8: Recommendations
Provides actionable keyword suggestions:

```
💡 RECOMMENDED KEYWORDS TO INCLUDE IN YOUR RESUME:
These keywords appear in the job description and are commonly used across resumes:

1. python                        🔥 High Priority (in 4/4 resumes, 100%)
2. machine learning              🔥 High Priority (in 4/4 resumes, 100%)
3. docker                        🔥 High Priority (in 3/4 resumes, 75%)
4. aws                           🔥 High Priority (in 3/4 resumes, 75%)
5. kubernetes                    ⭐ Important (in 2/4 resumes, 50%)
...

🎯 KEYWORDS FROM JOB DESCRIPTION YOU MIGHT BE MISSING:
These keywords appear in the job but are rare or absent in the analyzed resumes:

1. agile
2. scrum
3. ci/cd
4. jenkins
...
```

**Match Rate Interpretation:**
- **≥60%**: Excellent match - common keywords align very well with the job
- **40-59%**: Good match - many relevant keywords are present
- **20-39%**: Moderate match - consider including more job-specific keywords
- **<20%**: Low match - review job requirements carefully

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

## Key Improvements

### 1. CS-Focused Keywords
The system now focuses exclusively on Computer Science and technical terms, filtering out:
- Generic verbs (worked, managed, developed) unless they're technical terms
- Adjectives (excellent, great, strong)
- Months and dates (January, February, 2023)
- Common adverbs (very, really, extremely)

### 2. Full Word Output
Keywords are returned in their complete form:
- ✓ "python", "machine learning", "tensorflow"
- ✗ NOT "machin", "learn", "tensorflow" (stemmed forms)

### 3. Abbreviation Handling
Common degree abbreviations are automatically normalized:
- B.S., BS, b.s. → bachelor's degree
- M.S., MS → master's degree
- Ph.D., PhD → doctorate

### 4. N-gram Support for Multi-word Phrases
The system uses n-grams (1-3 words) to capture important technical phrases:
- **Unigrams** (1 word): python, docker, kubernetes
- **Bigrams** (2 words): machine learning, data structures, neural networks
- **Trigrams** (3 words): natural language processing, convolutional neural networks

This ensures phrases like "machine learning" are treated as single concepts rather than separate words "machine" and "learning".

### 5. Cosine Similarity Analysis
Measures the overall alignment between resumes and job descriptions:
- **Similarity Score**: 0-100% indicating how similar documents are
- **Length-Independent**: Normalizes for document size
- **Multi-Dimensional**: Considers all keywords simultaneously
- **Contribution Breakdown**: Shows which keywords drive similarity
- **Actionable Insights**: Identifies strengths and gaps

See [COSINE_SIMILARITY_GUIDE.md](COSINE_SIMILARITY_GUIDE.md) for detailed explanation.

### 6. Smart Overlap Detection
When comparing documents, the system intelligently handles:
- Plural variations (algorithm/algorithms)
- Case differences (Python/python)
- Multi-word terms (machine learning, data science)

## Validation Testing

The validator module (`validator.py`) includes four key tests:

### Test 1: Extractor Returns Keywords
Verifies that the pipeline successfully extracts keywords from sample text.

### Test 2: Scores Are Non-Zero
Ensures all extracted keywords have valid, non-zero TF-IDF scores.

### Test 3: Keyword-Heavy Document and CS Filtering
Tests that repeated important terms are ranked higher and that non-CS terms are filtered out (e.g., "python" appearing multiple times should rank high, while "excellent" should be filtered).

### Test 4: Multiple Documents Differentiation
Confirms that documents with different CS content produce distinct keyword sets (e.g., ML keywords vs web development keywords).

All tests run automatically when executing `main.py`. If any test fails, check your NLTK installation and data downloads.

## Customization

### Adjust Number of Keywords

Modify the `top_n` parameter in `main.py`:

```python
resume_keywords = extract_top_keywords(tfidf_scores[0], top_n=20)  # Extract 20 keywords
```

### Include Non-CS Keywords

To extract all keywords (not just CS-related), modify `tokenizer.py` or set `cs_only=False`:

```python
tokens = tokenize_and_remove_stopwords(cleaned, cs_only=False, filter_pos=True)
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
PROCESSING RESUME
--------------------------------------------------------------------------------

Processing: resume.pdf

--------------------------------------------------------------------------------
COMPUTING TF-IDF SCORES FOR RESUME
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
EXTRACTING RESUME KEYWORDS
--------------------------------------------------------------------------------

📄 TOP KEYWORDS FROM RESUME (resume.pdf):
--------------------------------------------------------------------------------
 1. python                        (score: 0.3845)
 2. machine learning              (score: 0.3421)
 3. tensorflow                    (score: 0.3198)
 4. docker                        (score: 0.2987)
 5. kubernetes                    (score: 0.2765)
 ...

💼 LOADING JOB DESCRIPTION (job1.txt):
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
MATCHING RESUME KEYWORDS WITH JOB DESCRIPTION
--------------------------------------------------------------------------------

✅ RESUME KEYWORDS FOUND IN JOB DESCRIPTION: 12/20
--------------------------------------------------------------------------------
 1. python                        (score: 0.3845) ✓
 2. machine learning              (score: 0.3421) ✓
 3. tensorflow                    (score: 0.3198) ✓
 4. docker                        (score: 0.2987) ✓
 5. aws                           (score: 0.2654) ✓
 ...

❌ RESUME KEYWORDS NOT FOUND IN JOB DESCRIPTION: 8/20
--------------------------------------------------------------------------------
 1. react                         (score: 0.2543) ✗
 2. mongodb                       (score: 0.2387) ✗
 3. redis                         (score: 0.2198) ✗
 ...

--------------------------------------------------------------------------------
MATCH ANALYSIS
--------------------------------------------------------------------------------

� KEYWORD MATCH RATE: 60.0% (12/20 keywords)
   ✓ Excellent match! Your resume skills align well with the job requirements.

================================================================================
RUNNING VALIDATION TESTS
================================================================================

=== Test 1: Extractor Returns CS Keywords ===
✓ PASS: Extractor returned CS keywords
  Keywords found: 6
  Sample keywords: ['python', 'machine learning', 'artificial intelligence']

=== Test 2: Scores Are Non-Zero and Full Words ===
✓ PASS: All scores are non-zero and words are full forms
  Score range: 0.2887 to 0.5774
  Sample keywords: ['python', 'java', 'javascript']

=== Test 3: Keyword-Heavy Document and CS Filtering ===
✓ PASS: Keyword-heavy CS terms ranked highly
  Top keywords in heavy doc: ['python', 'tensorflow', 'machine learning', 'kubernetes', 'docker']
  Doc1 CS keywords: ['software', 'engineer']

=== Test 4: Multiple Documents Differentiation ===
✓ PASS: Different documents produce different CS keywords
  Doc1 top keywords: ['machine learning', 'artificial intelligence', 'tensorflow']
  Doc2 top keywords: ['javascript', 'react', 'angular']
  Overlap: 0 keywords

============================================================
VALIDATION RESULTS: 4/4 tests passed
============================================================

================================================================================
PROCESS COMPLETE
================================================================================
```

## Understanding Cosine Similarity

Cosine similarity measures how similar two documents are by comparing their TF-IDF vectors. The score ranges from 0 to 1 (or 0% to 100%):

- **80-100%**: Excellent match - very high alignment
- **60-79%**: Good match - strong alignment
- **40-59%**: Moderate match - some alignment but gaps
- **20-39%**: Weak match - limited alignment
- **0-19%**: Poor match - minimal alignment

### Why Cosine Similarity Matters

1. **Length-Independent**: Normalizes for document size
2. **Multi-Dimensional**: Considers ALL keywords simultaneously
3. **Captures Overlap**: Identifies shared technical terms and phrases
4. **Actionable Insights**: Shows which keywords contribute most

### Contribution Breakdown

For each resume, the system shows which keywords contribute most to the similarity score:

```
Rank   Keyword                   Resume    Job       Contribution
1      machine learning          0.4523    0.4812    0.2176
2      python                    0.4201    0.4356    0.1830
```

- **Resume**: TF-IDF score in your resume
- **Job**: TF-IDF score in job description
- **Contribution**: Product of both scores (how much it helps similarity)

High contribution = keyword appears prominently in both documents (your strength!)

For detailed explanation, mathematical foundation, and examples, see [COSINE_SIMILARITY_GUIDE.md](COSINE_SIMILARITY_GUIDE.md).

## Additional Documentation

- **[N-GRAM_GUIDE.md](NGRAM_GUIDE.md)**: Complete guide to n-gram support and multi-word phrase extraction
- **[COSINE_SIMILARITY_GUIDE.md](COSINE_SIMILARITY_GUIDE.md)**: In-depth explanation of cosine similarity analysis
- **[MULTI_RESUME_GUIDE.md](MULTI_RESUME_GUIDE.md)**: Technical documentation for multi-resume analysis
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Quick reference for implementation details
- **[CHANGES.md](CHANGES.md)**: Version history and changelog

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

### Low cosine similarity scores
This is normal if your resume and the job description use different technical vocabularies. Review the contribution breakdown to identify gaps and add relevant keywords to your resume.

## License

This project is provided for educational purposes.

## Author

NLP Class Project - Resume Keyword Identifier