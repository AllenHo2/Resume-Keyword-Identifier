import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Dict, Tuple
from collections import Counter

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Stem tokens and create mapping from stemmed forms to original words
def stem_tokens(tokens: List[str], use_lemmatizer: bool = False) -> Tuple[List[str], Dict[str, List[str]]]:
    if use_lemmatizer:
        stemmer = WordNetLemmatizer()
        stemmed = [stemmer.lemmatize(token) for token in tokens]
    else:
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(token) for token in tokens]
    
    # Create mapping from stem to original words
    stem_to_original = {}
    for original, stem in zip(tokens, stemmed):
        if stem not in stem_to_original:
            stem_to_original[stem] = []
        stem_to_original[stem].append(original)
    
    return stemmed, stem_to_original

# Choose representative original word for each stem
def get_representative_word(stem: str, original_words: List[str]) -> str:
    if not original_words:
        return stem
    
    # Count frequency of each original form
    word_counts = Counter(original_words)
    
    # Get most common
    most_common = word_counts.most_common()
    max_count = most_common[0][1]
    
    # If multiple words have same frequency, choose shortest
    top_words = [word for word, count in most_common if count == max_count]
    return min(top_words, key=len)

# Create mapping from stemmed tokens to representative original words
def create_stem_mapping(tokens: List[str], use_lemmatizer: bool = False) -> Dict[str, str]:
    stemmed, stem_to_originals = stem_tokens(tokens, use_lemmatizer)
    
    stem_mapping = {}
    for stem, originals in stem_to_originals.items():
        stem_mapping[stem] = get_representative_word(stem, originals)
    
    return stem_mapping

# Stem tokens and return stemmed list along with mapping
def stem_with_mapping(tokens: List[str], use_lemmatizer: bool = False) -> Tuple[List[str], Dict[str, str]]:
    stemmed, stem_to_originals = stem_tokens(tokens, use_lemmatizer)
    
    stem_mapping = {}
    for stem, originals in stem_to_originals.items():
        stem_mapping[stem] = get_representative_word(stem, originals)
    
    return stemmed, stem_mapping
