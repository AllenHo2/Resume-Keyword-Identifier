from typing import List, Tuple, Dict
from src.cleaner import clean_text
from src.tokenizer import tokenize_and_remove_stopwords
from src.stemmer import stem_tokens
from src.tfidf_vectorizer import get_all_tfidf_scores
from src.keyword_extractor import extract_top_keywords

def test_extractor_returns_keywords():
    """Test 1: Verify that the extractor returns keywords."""
    print("\n=== Test 1: Extractor Returns Keywords ===")
    
    test_doc = "python programming machine learning data science artificial intelligence"
    cleaned = clean_text(test_doc)
    tokens = tokenize_and_remove_stopwords(cleaned)
    stemmed = stem_tokens(tokens)
    processed = ' '.join(stemmed)
    
    tfidf_scores = get_all_tfidf_scores([processed])
    keywords = extract_top_keywords(tfidf_scores[0], top_n=5)
    
    if len(keywords) > 0:
        print("✓ PASS: Extractor returned keywords")
        print(f"  Keywords found: {len(keywords)}")
        return True
    else:
        print("✗ FAIL: Extractor returned no keywords")
        return False

def test_scores_are_nonzero():
    """Test 2: Verify that TF-IDF scores are non-zero."""
    print("\n=== Test 2: Scores Are Non-Zero ===")
    
    test_doc = "software engineer with experience in python java javascript development testing"
    cleaned = clean_text(test_doc)
    tokens = tokenize_and_remove_stopwords(cleaned)
    stemmed = stem_tokens(tokens)
    processed = ' '.join(stemmed)
    
    tfidf_scores = get_all_tfidf_scores([processed])
    keywords = extract_top_keywords(tfidf_scores[0], top_n=5)
    
    all_nonzero = all(score > 0 for _, score in keywords)
    
    if all_nonzero and len(keywords) > 0:
        print("✓ PASS: All scores are non-zero")
        print(f"  Score range: {min(s for _, s in keywords):.4f} to {max(s for _, s in keywords):.4f}")
        return True
    else:
        print("✗ FAIL: Some scores are zero or no keywords found")
        return False

def test_keyword_heavy_document_ranking():
    """Test 3: Verify that keyword-heavy documents affect ranking appropriately."""
    print("\n=== Test 3: Keyword-Heavy Document Ranking ===")
    
    # Document 1: balanced content
    doc1 = "software engineer with experience in various programming languages"
    
    # Document 2: keyword-heavy with 'python'
    doc2 = "python python python developer expert python programming python skills python experience"
    
    # Process both documents
    docs = [doc1, doc2]
    processed_docs = []
    
    for doc in docs:
        cleaned = clean_text(doc)
        tokens = tokenize_and_remove_stopwords(cleaned)
        stemmed = stem_tokens(tokens)
        processed_docs.append(' '.join(stemmed))
    
    # Get TF-IDF scores
    tfidf_scores = get_all_tfidf_scores(processed_docs)
    
    # Extract keywords from doc2
    keywords_doc2 = extract_top_keywords(tfidf_scores[1], top_n=5)
    
    # Check if 'python' (or its stem) appears in top keywords of doc2
    top_words = [kw for kw, _ in keywords_doc2]
    python_variants = ['python', 'python']
    
    has_python = any(pv in top_words for pv in python_variants)
    
    if has_python:
        print("✓ PASS: Keyword-heavy terms ranked highly")
        print(f"  Top keywords in heavy doc: {top_words}")
        return True
    else:
        print("✗ FAIL: Keyword-heavy terms not ranked appropriately")
        print(f"  Top keywords: {top_words}")
        return False

def test_multiple_documents_differentiation():
    """Test 4: Verify that different documents produce different keyword rankings."""
    print("\n=== Test 4: Multiple Documents Differentiation ===")
    
    doc1 = "machine learning artificial intelligence deep learning neural networks"
    doc2 = "web development frontend backend javascript html css react"
    
    processed_docs = []
    for doc in [doc1, doc2]:
        cleaned = clean_text(doc)
        tokens = tokenize_and_remove_stopwords(cleaned)
        stemmed = stem_tokens(tokens)
        processed_docs.append(' '.join(stemmed))
    
    tfidf_scores = get_all_tfidf_scores(processed_docs)
    keywords1 = extract_top_keywords(tfidf_scores[0], top_n=5)
    keywords2 = extract_top_keywords(tfidf_scores[1], top_n=5)
    
    words1 = set(kw for kw, _ in keywords1)
    words2 = set(kw for kw, _ in keywords2)
    
    overlap = words1.intersection(words2)
    
    if len(overlap) < len(words1) and len(overlap) < len(words2):
        print("✓ PASS: Different documents produce different keywords")
        print(f"  Doc1 top keywords: {list(words1)[:3]}")
        print(f"  Doc2 top keywords: {list(words2)[:3]}")
        print(f"  Overlap: {len(overlap)} keywords")
        return True
    else:
        print("✗ FAIL: Documents not sufficiently differentiated")
        return False

def run_all_validation_tests():
    """Run all validation tests and report results."""
    print("\n" + "="*60)
    print("RUNNING VALIDATION TESTS")
    print("="*60)
    
    tests = [
        test_extractor_returns_keywords,
        test_scores_are_nonzero,
        test_keyword_heavy_document_ranking,
        test_multiple_documents_differentiation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ ERROR in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"VALIDATION RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    return all(results)
