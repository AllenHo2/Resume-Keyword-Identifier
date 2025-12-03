"""
Simple test script to verify cosine similarity implementation.
"""

from src.similarity import (
    compute_pairwise_similarity,
    compute_resume_job_similarities,
    interpret_similarity_score
)

def test_cosine_similarity():
    """Test cosine similarity functionality."""
    
    print("="*80)
    print("COSINE SIMILARITY TESTS")
    print("="*80)
    
    # Test 1: Identical documents
    print("\n=== Test 1: Identical Documents ===")
    doc1 = "python machine learning tensorflow deep learning"
    doc2 = "python machine learning tensorflow deep learning"
    similarity = compute_pairwise_similarity(doc1, doc2)
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
    print(f"Expected: ~1.0 (identical)")
    print(f"Result: {'✓ PASS' if similarity > 0.99 else '✗ FAIL'}")
    
    # Test 2: Very similar documents
    print("\n=== Test 2: Very Similar Documents ===")
    doc1 = "python machine learning tensorflow deep learning neural networks"
    doc2 = "python machine learning tensorflow deep learning computer vision"
    similarity = compute_pairwise_similarity(doc1, doc2)
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
    print(f"Expected: 0.7-0.9 (high similarity)")
    print(f"Interpretation: {interpret_similarity_score(similarity)}")
    print(f"Result: {'✓ PASS' if 0.7 <= similarity <= 0.9 else '✗ FAIL'}")
    
    # Test 3: Somewhat similar documents
    print("\n=== Test 3: Somewhat Similar Documents ===")
    doc1 = "python machine learning data science tensorflow"
    doc2 = "javascript react nodejs web development"
    similarity = compute_pairwise_similarity(doc1, doc2)
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
    print(f"Expected: 0.0-0.3 (low similarity)")
    print(f"Interpretation: {interpret_similarity_score(similarity)}")
    print(f"Result: {'✓ PASS' if similarity < 0.3 else '✗ FAIL'}")
    
    # Test 4: Multiple resumes vs job
    print("\n=== Test 4: Multiple Resumes vs Job Description ===")
    resume1 = "python machine learning tensorflow deep learning neural networks data science"
    resume2 = "python flask django rest api web development backend"
    resume3 = "javascript react angular vue frontend web development"
    job = "python machine learning tensorflow neural networks data science artificial intelligence"
    
    resumes = [resume1, resume2, resume3]
    similarities = compute_resume_job_similarities(resumes, job)
    
    print(f"Job Description: {job}\n")
    for idx, (resume, similarity) in enumerate(zip(resumes, similarities), 1):
        print(f"Resume {idx}: {resume}")
        print(f"Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
        print(f"Interpretation: {interpret_similarity_score(similarity)}")
        print()
    
    print(f"Expected: Resume 1 should have highest similarity (most overlap with job)")
    print(f"Result: {'✓ PASS' if similarities[0] > similarities[1] and similarities[0] > similarities[2] else '✗ FAIL'}")
    
    # Test 5: N-gram support
    print("\n=== Test 5: N-gram Support ===")
    doc1 = "machine learning deep learning natural language processing"
    doc2 = "machine learning deep learning computer vision"
    
    # Without n-grams (unigrams only)
    sim_unigrams = compute_pairwise_similarity(doc1, doc2, ngram_range=(1, 1))
    
    # With n-grams
    sim_with_ngrams = compute_pairwise_similarity(doc1, doc2, ngram_range=(1, 2))
    
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Similarity (unigrams only): {sim_unigrams:.4f} ({sim_unigrams*100:.2f}%)")
    print(f"Similarity (with bigrams):  {sim_with_ngrams:.4f} ({sim_with_ngrams*100:.2f}%)")
    print(f"Expected: Both should show high similarity (shared phrases)")
    print(f"Result: {'✓ PASS' if sim_with_ngrams > 0.6 else '✗ FAIL'}")
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_cosine_similarity()
