import os
from src.cleaner import process_document
from src.tokenizer import tokenize_and_remove_stopwords
from src.stemmer import stem_tokens
from src.tfidf_vectorizer import get_all_tfidf_scores
from src.keyword_extractor import extract_top_keywords, find_common_keywords, calculate_keyword_match_score
from src.validator import run_all_validation_tests
from src.utils import file_exists, list_files_in_directory

def process_and_extract_keywords(file_path: str, top_n: int = 15, use_lemmatizer: bool = False):
    """
    Complete pipeline: load → clean → tokenize → stem → TF-IDF → extract keywords.
    
    Args:
        file_path: Path to document (PDF or text)
        top_n: Number of top keywords to extract
        use_lemmatizer: Whether to use lemmatization instead of stemming
    
    Returns:
        Tuple of (keywords, processed_text)
    """
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    # Step 1: Clean
    cleaned_text = process_document(file_path, remove_punctuation=True)
    
    # Step 2: Tokenize and remove stopwords
    tokens = tokenize_and_remove_stopwords(cleaned_text)
    
    # Step 3: Stem/Lemmatize
    stemmed_tokens = stem_tokens(tokens, use_lemmatizer=use_lemmatizer)
    
    # Step 4: Prepare for TF-IDF
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

def main():
    print("="*80)
    print("RESUME KEYWORD EXTRACTOR")
    print("="*80)
    
    # Define file paths
    resume_dir = "assets/resumes"
    job_dir = "assets/jobs"
    
    # Find resume and job files
    resume_files = list_files_in_directory(resume_dir, '.pdf')
    if not resume_files:
        resume_files = list_files_in_directory(resume_dir, '.txt')
    
    job_files = list_files_in_directory(job_dir, '.txt')
    
    if not resume_files:
        print(f"\n⚠ No resume files found in {resume_dir}")
        print("Please add a resume file (PDF or TXT) to the assets/resumes directory.")
        return
    
    if not job_files:
        print(f"\n⚠ No job description files found in {job_dir}")
        print("Please add a job description file (TXT) to the assets/jobs directory.")
        return
    
    # Use first resume and first job description
    resume_path = resume_files[0]
    job_path = job_files[0]
    
    print(f"\nResume: {os.path.basename(resume_path)}")
    print(f"Job Description: {os.path.basename(job_path)}")
    
    # Process both documents
    print("\n" + "-"*80)
    print("PROCESSING DOCUMENTS")
    print("-"*80)
    
    resume_processed = process_and_extract_keywords(resume_path)
    job_processed = process_and_extract_keywords(job_path)
    
    # Compute TF-IDF for both documents
    print("\n" + "-"*80)
    print("COMPUTING TF-IDF SCORES")
    print("-"*80)
    
    documents = [resume_processed, job_processed]
    tfidf_scores = get_all_tfidf_scores(documents, max_features=100)
    
    # Extract top keywords
    print("\n" + "-"*80)
    print("EXTRACTING TOP KEYWORDS")
    print("-"*80)
    
    resume_keywords = extract_top_keywords(tfidf_scores[0], top_n=15)
    job_keywords = extract_top_keywords(tfidf_scores[1], top_n=15)
    
    # Display resume keywords
    print(f"\n📄 TOP KEYWORDS - RESUME ({os.path.basename(resume_path)}):")
    print("-"*80)
    for idx, (keyword, score) in enumerate(resume_keywords, 1):
        print(f"{idx:2d}. {keyword:20s} (score: {score:.4f})")
    
    # Display job keywords
    print(f"\n💼 TOP KEYWORDS - JOB DESCRIPTION ({os.path.basename(job_path)}):")
    print("-"*80)
    for idx, (keyword, score) in enumerate(job_keywords, 1):
        print(f"{idx:2d}. {keyword:20s} (score: {score:.4f})")
    
    # Find overlapping keywords
    print("\n" + "-"*80)
    print("KEYWORD OVERLAP ANALYSIS")
    print("-"*80)
    
    common_keywords = find_common_keywords(resume_keywords, job_keywords)
    match_score = calculate_keyword_match_score(resume_keywords, job_keywords)
    
    print(f"\n🔗 OVERLAPPING KEYWORDS: {len(common_keywords)}")
    if common_keywords:
        print("   " + ", ".join(common_keywords))
    else:
        print("   No overlapping keywords found in top 15.")
    
    print(f"\n📊 MATCH SCORE: {match_score:.2f}%")
    
    if match_score >= 30:
        print("   ✓ Strong alignment between resume and job description")
    elif match_score >= 15:
        print("   ~ Moderate alignment between resume and job description")
    else:
        print("   ⚠ Low alignment between resume and job description")
    
    # Run validation tests
    print("\n" + "="*80)
    print("RUNNING VALIDATION TESTS")
    print("="*80)
    
    run_all_validation_tests()
    
    print("\n" + "="*80)
    print("PROCESS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
