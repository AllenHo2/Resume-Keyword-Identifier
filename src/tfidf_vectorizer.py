from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Tuple, Dict
from src.stemmer import stem_with_mapping

# Compute TF-IDF matrix for a collection of documents.
def compute_tfidf(documents: List[str], max_features: int = None, 
                  min_df: int = 1, max_df: float = 1.0, ngram_range: Tuple[int, int] = (1, 3),
                  use_stemming: bool = True) -> Tuple[np.ndarray, List[str], TfidfVectorizer, Dict[str, str]]:
    stem_mapping = {}
    processed_docs = documents
    
    if use_stemming:
        # Stem all documents to group similar words
        stemmed_docs = []
        
        for doc in documents:
            tokens = doc.split()
            stemmed_tokens, doc_mapping = stem_with_mapping(tokens, use_lemmatizer=False)
            stemmed_docs.append(' '.join(stemmed_tokens))
            stem_mapping.update(doc_mapping)
        
        processed_docs = stemmed_docs
    
    # Create TF-IDF vectorizer with n-gram support
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'\b\w+\b',
        ngram_range=ngram_range
    )
    
    # Fit and transform on stemmed documents (if stemming enabled)
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Restore original forms for feature names
    if use_stemming and stem_mapping:
        original_feature_names = []
        for feature in feature_names:
            # Check if it's a multi-word phrase (n-gram)
            words = feature.split()
            if len(words) > 1:
                # Multi-word phrase: restore each word
                restored_words = [stem_mapping.get(w, w) for w in words]
                original_feature_names.append(' '.join(restored_words))
            else:
                # Single word: restore from mapping
                original_feature_names.append(stem_mapping.get(feature, feature))
        
        return tfidf_matrix, original_feature_names, vectorizer, stem_mapping
    
    return tfidf_matrix, feature_names, vectorizer, stem_mapping

# Extract TF-IDF scores for a specific document
def get_tfidf_scores(tfidf_matrix: np.ndarray, feature_names: List[str], 
                     doc_index: int = 0) -> Dict[str, float]:
    feature_array = tfidf_matrix.toarray()[doc_index]
    tfidf_scores = {}
    
    for idx, score in enumerate(feature_array):
        if score > 0:
            tfidf_scores[feature_names[idx]] = score
    
    return tfidf_scores

# Get TF-IDF scores for all documents
def get_all_tfidf_scores(documents: List[str], max_features: int = None, 
                        min_df: int = 1, max_df: float = 1.0,
                        ngram_range: Tuple[int, int] = (1, 3),
                        use_stemming: bool = True) -> List[Dict[str, float]]:
    tfidf_matrix, feature_names, _, _ = compute_tfidf(
        documents, 
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range, 
        use_stemming=use_stemming
    )
    
    results = []
    # Extract scores for each document
    for doc_idx in range(len(documents)):
        scores = get_tfidf_scores(tfidf_matrix, feature_names, doc_idx)
        results.append(scores)
    
    return results
