from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict

# Compute cosine similarity matrix for a list of documents
def compute_cosine_similarity(documents: List[str], ngram_range: Tuple[int, int] = (1, 3), 
                              max_features: int = None) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        token_pattern=r'\b\w+\b'
    )
    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix

# Compute resume-job similarities
def compute_resume_job_similarities(resume_texts: List[str], job_text: str,
                                   ngram_range: Tuple[int, int] = (1, 3),
                                   max_features: int = None) -> List[float]:
    # Combine all documents (resumes + job)
    all_documents = resume_texts + [job_text]
    
    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity(all_documents, ngram_range, max_features)
    
    # Extract similarities between each resume and the job (last document)
    job_index = len(resume_texts)
    similarities = []
    
    for resume_idx in range(len(resume_texts)):
        similarity = similarity_matrix[resume_idx, job_index]
        similarities.append(similarity)
    
    return similarities

# Get similarity breakdown for a specific resume
def get_similarity_breakdown(doc1_vector, doc2_vector, feature_names: List[str], 
                             top_n: int = 20) -> List[Tuple[str, float, float, float]]:
    doc1_array = doc1_vector.toarray()[0] if hasattr(doc1_vector, 'toarray') else doc1_vector
    doc2_array = doc2_vector.toarray()[0] if hasattr(doc2_vector, 'toarray') else doc2_vector
    
    contributions = []
    for idx, feature in enumerate(feature_names):
        score1 = doc1_array[idx]
        score2 = doc2_array[idx]
        
        # Contribution is the product of both scores (both must be non-zero to contribute)
        contribution = score1 * score2
        
        if contribution > 0:
            contributions.append((feature, score1, score2, contribution))
    
    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[3], reverse=True)
    
    return contributions[:top_n]

# Compute similarity with breakdown
def compute_similarity_with_breakdown(resume_texts: List[str], job_text: str,
                                     ngram_range: Tuple[int, int] = (1, 3),
                                     max_features: int = None,
                                     top_n: int = 20) -> List[Dict]:
    # Create vectorizer and compute TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        token_pattern=r'\b\w+\b'
    )

    # Compute TF-IDF matrix
    all_documents = resume_texts + [job_text]
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    feature_names = vectorizer.get_feature_names_out()

    # Get job vector
    job_index = len(resume_texts)
    job_vector = tfidf_matrix[job_index]
    
    results = []
    for resume_idx in range(len(resume_texts)):
        resume_vector = tfidf_matrix[resume_idx]
        
        # Compute similarity
        similarity = cosine_similarity(resume_vector, job_vector)[0, 0]
        
        # Get breakdown
        breakdown = get_similarity_breakdown(resume_vector, job_vector, feature_names, top_n)
        
        results.append({
            'similarity': similarity,
            'breakdown': breakdown
        })
    
    return results

# Interpret similarity score
def interpret_similarity_score(score: float) -> str:
    if score >= 0.8:
        return "Excellent match - very high alignment"
    elif score >= 0.6:
        return "Good match - strong alignment"
    elif score >= 0.4:
        return "Moderate match - some alignment"
    elif score >= 0.2:
        return "Weak match - limited alignment"
    else:
        return "Poor match - minimal alignment"
