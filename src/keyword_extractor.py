from typing import List, Tuple, Dict

def extract_top_keywords(tfidf_scores: Dict[str, float], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract top N keywords ranked by TF-IDF score.
    
    Args:
        tfidf_scores: Dictionary mapping keywords to TF-IDF scores
        top_n: Number of top keywords to extract
    
    Returns:
        List of (keyword, score) tuples sorted by score in descending order
    """
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]

def extract_keywords_from_multiple_docs(all_tfidf_scores: List[Dict[str, float]], 
                                        top_n: int = 10) -> List[List[Tuple[str, float]]]:
    """
    Extract top keywords from multiple documents.
    
    Args:
        all_tfidf_scores: List of dictionaries with TF-IDF scores per document
        top_n: Number of top keywords per document
    
    Returns:
        List of lists containing (keyword, score) tuples for each document
    """
    return [extract_top_keywords(scores, top_n) for scores in all_tfidf_scores]

def find_common_keywords(keywords_list1: List[Tuple[str, float]], 
                         keywords_list2: List[Tuple[str, float]]) -> List[str]:
    """
    Find overlapping keywords between two keyword lists.
    
    Args:
        keywords_list1: First list of (keyword, score) tuples
        keywords_list2: Second list of (keyword, score) tuples
    
    Returns:
        List of common keywords
    """
    set1 = set(keyword for keyword, _ in keywords_list1)
    set2 = set(keyword for keyword, _ in keywords_list2)
    return sorted(list(set1.intersection(set2)))

def calculate_keyword_match_score(keywords_list1: List[Tuple[str, float]], 
                                  keywords_list2: List[Tuple[str, float]]) -> float:
    """
    Calculate a match score based on overlapping keywords.
    
    Args:
        keywords_list1: First list of (keyword, score) tuples
        keywords_list2: Second list of (keyword, score) tuples
    
    Returns:
        Match score as percentage of overlap
    """
    common = find_common_keywords(keywords_list1, keywords_list2)
    total_unique = len(set(k for k, _ in keywords_list1) | set(k for k, _ in keywords_list2))
    
    if total_unique == 0:
        return 0.0
    
    return (len(common) / total_unique) * 100
