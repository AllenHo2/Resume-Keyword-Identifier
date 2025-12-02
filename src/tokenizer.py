import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def tokenize(text: str) -> List[str]:
    """
    Tokenize text using NLTK word_tokenize.
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens
    """
    return word_tokenize(text)

def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """
    Remove stopwords from token list using NLTK stopwords.
    
    Args:
        tokens: List of tokens
        language: Language for stopwords (default: 'english')
    
    Returns:
        List of tokens with stopwords removed
    """
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token not in stop_words and len(token) > 1]

def tokenize_and_remove_stopwords(text: str, language: str = 'english') -> List[str]:
    """
    Tokenize text and remove stopwords in one step.
    
    Args:
        text: Input text string
        language: Language for stopwords
    
    Returns:
        List of filtered tokens
    """
    tokens = tokenize(text)
    return remove_stopwords(tokens, language)
