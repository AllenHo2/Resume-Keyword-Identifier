from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Tuple, Dict

def compute_tfidf(documents: List[str], max_features: int = None, 
                  min_df: int = 1, max_df: float = 1.0) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
    """
    Compute TF-IDF matrix for a list of documents using scikit-learn.
    
    Args:
        documents: List of document strings (preprocessed text)
        max_features: Maximum number of features to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency (as fraction)
    
    Returns:
        Tuple of (tfidf_matrix, feature_names, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names, vectorizer

def get_tfidf_scores(tfidf_matrix: np.ndarray, feature_names: List[str], 
                     doc_index: int = 0) -> Dict[str, float]:
    """
    Extract TF-IDF scores for a specific document as a dictionary.
    
    Args:
        tfidf_matrix: TF-IDF matrix from compute_tfidf
        feature_names: List of feature names (vocabulary)
        doc_index: Index of the document to extract scores for
    
    Returns:
        Dictionary mapping terms to their TF-IDF scores
    """
    feature_array = tfidf_matrix.toarray()[doc_index]
    tfidf_scores = {}
    
    for idx, score in enumerate(feature_array):
        if score > 0:
            tfidf_scores[feature_names[idx]] = score
    
    return tfidf_scores

def get_all_tfidf_scores(documents: List[str], max_features: int = None) -> List[Dict[str, float]]:
    """
    Compute TF-IDF scores for all documents and return as list of dictionaries.
    
    Args:
        documents: List of document strings
        max_features: Maximum number of features
    
    Returns:
        List of dictionaries, one per document, mapping terms to scores
    """
    tfidf_matrix, feature_names, _ = compute_tfidf(documents, max_features)
    
    results = []
    for doc_idx in range(len(documents)):
        scores = get_tfidf_scores(tfidf_matrix, feature_names, doc_idx)
        results.append(scores)
    
    return results
