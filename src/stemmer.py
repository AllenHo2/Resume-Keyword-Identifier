import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def stem_tokens(tokens: List[str], use_lemmatizer: bool = False) -> List[str]:
    """
    Stem or lemmatize tokens using NLTK.
    
    Args:
        tokens: List of tokens to process
        use_lemmatizer: If True, use WordNetLemmatizer; otherwise use PorterStemmer
    
    Returns:
        List of stemmed/lemmatized tokens
    """
    if use_lemmatizer:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    else:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

def stem_text(text: str, use_lemmatizer: bool = False) -> str:
    """
    Stem or lemmatize a text string.
    
    Args:
        text: Space-separated token string
        use_lemmatizer: If True, use WordNetLemmatizer; otherwise use PorterStemmer
    
    Returns:
        Space-separated string of stemmed/lemmatized tokens
    """
    tokens = text.split()
    stemmed_tokens = stem_tokens(tokens, use_lemmatizer)
    return ' '.join(stemmed_tokens)
