import os
import re
from src.cleaner import process_document
from src.tokenizer import tokenize_and_remove_stopwords
from src.tfidf_vectorizer import get_all_tfidf_scores
from src.keyword_extractor import extract_top_keywords
from src.similarity import compute_similarity_with_breakdown, interpret_similarity_score
from src.utils import list_files_in_directory, load_text_file

# Process a document and extract keywords
def process_and_extract_keywords(file_path: str, top_n: int = 15, use_lemmatizer: bool = False):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    cleaned_text = process_document(file_path, remove_punctuation=True)
    tokens = tokenize_and_remove_stopwords(cleaned_text, cs_only=True, filter_pos=True)
    processed_text = ' '.join(tokens)
    
    return processed_text

# Find which resume keywords appear in the job description
def find_resume_keywords_in_job(resume_keywords, job_text_raw):
    job_text_lower = job_text_raw.lower()
    
    found = []
    missing = []
    
    for keyword, score in resume_keywords:
        keyword_lower = keyword.lower()
        
        # Check if keyword appears in job text (as whole word or phrase)
        # Use word boundaries for single words, direct match for phrases
        if ' ' in keyword_lower:
            # Multi-word phrase - direct match
            if keyword_lower in job_text_lower:
                found.append((keyword, score))
            else:
                missing.append((keyword, score))
        else:
            # Single word - use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, job_text_lower):
                found.append((keyword, score))
            else:
                missing.append((keyword, score))
    
    return found, missing

# Analyze keyword frequency across multiple resumes
def analyze_keyword_frequency(all_keywords_by_resume):
    from collections import defaultdict
    
    keyword_stats = defaultdict(lambda: {'count': 0, 'scores': []})
    
    # Aggregate counts and scores
    for resume_keywords in all_keywords_by_resume:
        for keyword, score in resume_keywords:
            keyword_stats[keyword]['count'] += 1
            keyword_stats[keyword]['scores'].append(score)
    
    # Calculate average scores
    keyword_frequency = {}
    for keyword, stats in keyword_stats.items():
        avg_score = sum(stats['scores']) / len(stats['scores'])
        keyword_frequency[keyword] = {
            'frequency': stats['count'],
            'avg_score': avg_score
        }
    
    return keyword_frequency

def main():
    print("="*80)
    print("MULTI-RESUME KEYWORD ANALYZER")
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
    
    # Interactive job selection if multiple jobs available
    if len(job_files) > 1:
        print(f"\n📋 Found {len(job_files)} job description(s):")
        for idx, job_file in enumerate(job_files, 1):
            print(f"  {idx}. {os.path.basename(job_file)}")
        
        while True:
            try:
                choice = input(f"\nSelect job description (1-{len(job_files)}): ").strip()
                job_idx = int(choice) - 1
                if 0 <= job_idx < len(job_files):
                    job_path = job_files[job_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(job_files)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                return
    else:
        job_path = job_files[0]
    
    print(f"\n📁 Found {len(resume_files)} resume(s)")
    print(f"📄 Job Description: {os.path.basename(job_path)}")
    
    # Process all resumes
    print("\n" + "="*80)
    print("STEP 1: PROCESSING ALL RESUMES")
    print("="*80)
    
    all_processed_resumes = []
    resume_names = []
    
    for resume_path in resume_files:
        processed = process_and_extract_keywords(resume_path)
        all_processed_resumes.append(processed)
        resume_names.append(os.path.basename(resume_path))
    
    # Use n-grams to capture multi-word phrases like "machine learning", "data structures"
    # Use stemming to group similar words but return original forms
    tfidf_scores_all = get_all_tfidf_scores(all_processed_resumes, max_features=150, ngram_range=(1, 3), use_stemming=True)
    
    # Extract keywords from each resume
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING KEYWORDS FROM EACH RESUME")
    print("="*80)
    
    all_keywords_by_resume = []
    for idx, scores in enumerate(tfidf_scores_all):
        keywords = extract_top_keywords(scores, top_n=30)
        all_keywords_by_resume.append(keywords)
        print(f"\n{resume_names[idx]}: {len(keywords)} keywords extracted")
    
    # Analyze keyword frequency across resumes
    print("\n" + "="*80)
    print("STEP 3: ANALYZING KEYWORD FREQUENCY ACROSS RESUMES")
    print("="*80)
    
    keyword_frequency = analyze_keyword_frequency(all_keywords_by_resume)
    
    # Sort by frequency, then by average score
    sorted_keywords = sorted(
        keyword_frequency.items(),
        key=lambda x: (x[1]['frequency'], x[1]['avg_score']),
        reverse=True
    )
    
    print(f"\n🔍 TOP COMPUTER SCIENCE KEYWORDS ACROSS ALL RESUMES:")
    print("-"*80)
    print(f"{'Rank':<6} {'Keyword':<30} {'Frequency':<12} {'Avg Score':<12}")
    print("-"*80)
    for idx, (keyword, stats) in enumerate(sorted_keywords[:25], 1):
        freq_display = f"{stats['frequency']}/{len(resume_files)} resumes"
        print(f"{idx:<6} {keyword:<30} {freq_display:<12} {stats['avg_score']:.4f}")
    
    # Load job description
    print(f"\n" + "="*80)
    print("STEP 4: LOADING JOB DESCRIPTION")
    print("="*80)
    job_text_raw = load_text_file(job_path)
    print(f"Loaded: {os.path.basename(job_path)}")
    
    # Process job description for similarity analysis
    job_cleaned = process_document(job_path, remove_punctuation=True)
    job_tokens = tokenize_and_remove_stopwords(job_cleaned, cs_only=True, filter_pos=True)
    job_processed = ' '.join(job_tokens)
    
    # Compute cosine similarity
    print(f"\n" + "="*80)
    print("STEP 5: COMPUTING COSINE SIMILARITY")
    print("="*80)
    print("Measuring how similar each resume is to the job description...\n")
    
    similarity_results = compute_similarity_with_breakdown(
        all_processed_resumes, 
        job_processed, 
        ngram_range=(1, 3),
        max_features=150,
        top_n=15
    )
    
    print(f"{'Resume':<40} {'Similarity':<12} {'Match Quality'}")
    print("-"*80)
    for idx, result in enumerate(similarity_results):
        similarity_score = result['similarity']
        interpretation = interpret_similarity_score(similarity_score)
        similarity_pct = similarity_score * 100
        print(f"{resume_names[idx]:<40} {similarity_pct:>5.2f}%      {interpretation}")
    
    # Show top contributing keywords for each resume
    print(f"\n📊 TOP CONTRIBUTING KEYWORDS TO SIMILARITY:")
    print("-"*80)
    for idx, result in enumerate(similarity_results):
        print(f"\n{resume_names[idx]}:")
        print(f"  {'Rank':<6} {'Keyword':<25} {'Resume':<10} {'Job':<10} {'Contribution'}")
        print(f"  {'-'*70}")
        for rank, (keyword, resume_score, job_score, contribution) in enumerate(result['breakdown'], 1):
            print(f"  {rank:<6} {keyword:<25} {resume_score:>6.4f}    {job_score:>6.4f}    {contribution:>6.4f}")
    
    # Match top keywords with job description
    print("\n" + "="*80)
    print("STEP 6: MATCHING TOP KEYWORDS WITH JOB DESCRIPTION")
    print("="*80)
    
    # Get top 30 keywords overall
    top_keywords = [(kw, stats['avg_score']) for kw, stats in sorted_keywords[:30]]
    found_keywords, missing_keywords = find_resume_keywords_in_job(top_keywords, job_text_raw)
    
    print(f"\n✅ TOP KEYWORDS FOUND IN JOB DESCRIPTION: {len(found_keywords)}/{len(top_keywords)}")
    print("-"*80)
    if found_keywords:
        for idx, (keyword, score) in enumerate(found_keywords, 1):
            freq = keyword_frequency[keyword]['frequency']
            freq_display = f"({freq}/{len(resume_files)} resumes)"
            print(f"{idx:2d}. {keyword:<30} {freq_display:<15} ✓")
    else:
        print("   No keywords found in job description.")
    
    print(f"\n❌ TOP KEYWORDS NOT FOUND IN JOB DESCRIPTION: {len(missing_keywords)}/{len(top_keywords)}")
    print("-"*80)
    if missing_keywords:
        for idx, (keyword, score) in enumerate(missing_keywords, 1):
            freq = keyword_frequency[keyword]['frequency']
            freq_display = f"({freq}/{len(resume_files)} resumes)"
            print(f"{idx:2d}. {keyword:<30} {freq_display:<15} ✗")
    
    # Recommendations
    print("\n" + "="*80)
    print("STEP 7: KEYWORD RECOMMENDATIONS")
    print("="*80)
    
    # Find keywords that appear in job but are common across resumes
    print("\n💡 RECOMMENDED KEYWORDS TO INCLUDE IN YOUR RESUME:")
    print("-"*80)
    print("These keywords appear in the job description and are commonly used across resumes:\n")
    
    if found_keywords:
        recommendations = []
        for keyword, score in found_keywords[:15]:
            freq = keyword_frequency[keyword]['frequency']
            recommendations.append((keyword, freq, score))
        
        # Sort by frequency (most common first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        for idx, (keyword, freq, score) in enumerate(recommendations, 1):
            freq_pct = (freq / len(resume_files)) * 100
            priority = "High Priority" if freq >= len(resume_files) * 0.5 else "Important"
            print(f"{idx:2d}. {keyword:<30} {priority} (in {freq}/{len(resume_files)} resumes, {freq_pct:.0f}%)")
    else:
        print("   No matching keywords found to recommend.")
    
    # Additional suggestions from job that aren't common in resumes
    print("\n🎯 KEYWORDS FROM JOB DESCRIPTION YOU MIGHT BE MISSING:")
    print("-"*80)
    print("These keywords also appeared in the job but are rare or absent in the analyzed resumes:\n")
    
    # Use already processed job tokens
    job_cs_keywords = set(job_tokens)
    
    # Find keywords in job but not common in resumes
    common_resume_keywords = set(kw for kw, stats in keyword_frequency.items() if stats['frequency'] >= 2)
    missing_from_resumes = job_cs_keywords - common_resume_keywords
    
    if missing_from_resumes:
        missing_list = sorted(list(missing_from_resumes))[:15]
        for idx, keyword in enumerate(missing_list, 1):
            print(f"{idx:2d}. {keyword}")
    else:
        print("   Your resumes already cover most job requirements!")

if __name__ == "__main__":
    main()
