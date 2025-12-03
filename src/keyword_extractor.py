from typing import List, Tuple, Dict

# Extract top N keywords based on TF-IDF scores
def extract_top_keywords(tfidf_scores: Dict[str, float], top_n: int = 10) -> List[Tuple[str, float]]:
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]

# Extract keywords from multiple documents
def extract_keywords_from_multiple_docs(all_tfidf_scores: List[Dict[str, float]], 
                                        top_n: int = 10) -> List[List[Tuple[str, float]]]:
    return [extract_top_keywords(scores, top_n) for scores in all_tfidf_scores]

# Normalize keyword for comparison (e.g., singularize, lowercase)
def normalize_keyword_for_comparison(keyword: str) -> str:
    # Convert to lowercase and strip
    normalized = keyword.lower().strip()
    
    # Handle plural forms and common variations
    if normalized.endswith('s') and len(normalized) > 3:
        singular = normalized[:-1]
        # Check if it's likely a plural (not "class", "pass", etc.)
        if not normalized.endswith(('ss', 'us', 'is')):
            return singular
    
    return normalized