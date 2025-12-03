import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from typing import List
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Computer Science and Technical Terms Dictionary
CS_KEYWORDS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'cpp', 'csharp', 'c#', 'ruby', 'go', 'golang',
    'rust', 'swift', 'kotlin', 'scala', 'php', 'perl', 'r', 'matlab', 'sql', 'html', 'css',
    'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'django', 'flask', 'spring', 'fastapi',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 'scipy',
    'machine learning', 'deep learning', 'neural networks', 'artificial intelligence', 'ai', 'ml',
    'computer vision', 'nlp', 'natural language processing', 'data science', 'big data',
    'algorithms', 'data structures', 'algorithm', 'database', 'databases', 'mongodb', 'postgresql',
    'mysql', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'nosql',
    'aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker', 'container', 'containerization',
    'microservices', 'api', 'rest', 'graphql', 'grpc', 'websocket', 'http', 'https',
    'git', 'github', 'gitlab', 'bitbucket', 'ci/cd', 'jenkins', 'travis', 'circleci',
    'linux', 'unix', 'bash', 'shell', 'windows', 'macos',
    'testing', 'unit testing', 'integration testing', 'tdd', 'bdd', 'junit', 'pytest', 'jest',
    'agile', 'scrum', 'kanban', 'jira', 'confluence',
    'frontend', 'backend', 'fullstack', 'full-stack', 'devops', 'sre',
    'optimization', 'performance', 'scalability', 'security', 'encryption',
    'software engineering', 'software development', 'programming', 'coding', 'development',
    'web development', 'mobile development', 'ios', 'android',
    'data analysis', 'data visualization', 'analytics', 'statistics', 'statistical',
    'distributed systems', 'parallel computing', 'concurrency', 'multithreading',
    'object-oriented', 'oop', 'functional programming', 'design patterns',
    'version control', 'debugging', 'profiling', 'monitoring', 'logging',
    'networking', 'tcp', 'udp', 'dns', 'load balancing', 'cdn',
    'hadoop', 'spark', 'kafka', 'airflow', 'etl', 'data pipeline', 'pipeline',
    'blockchain', 'ethereum', 'solidity', 'cryptocurrency',
    'quantum computing', 'embedded systems', 'iot', 'robotics',
    'compiler', 'interpreter', 'virtual machine', 'jvm',
    'data modeling', 'schema', 'orm', 'mvc', 'mvvm',
    'authentication', 'authorization', 'oauth', 'jwt', 'ssl', 'tls',
    'webscraping', 'crawler', 'scraping', 'automation',
    'tensorflow', 'opencv', 'matplotlib', 'seaborn', 'plotly',
    'jupyter', 'notebook', 'colab', 'anaconda',
    'regression', 'classification', 'clustering', 'recommendation',
    'supervised', 'unsupervised', 'reinforcement learning',
    'model', 'models', 'training', 'inference', 'deployment',
    'feature engineering', 'preprocessing', 'pipeline',
    'cross-validation', 'hyperparameter', 'tuning', 'optimization',
    'convolutional', 'recurrent', 'transformer', 'attention',
    'lstm', 'gru', 'rnn', 'cnn', 'gan', 'vae', 'bert', 'gpt',
    'computer science', 'cs', 'engineering', 'engineer', 'developer',
    'architect', 'architecture', 'technical', 'technology', 'tech',
    'software', 'hardware', 'framework', 'frameworks', 'library', 'libraries',
    'sdk', 'api', 'cli', 'gui', 'ui', 'ux', 'interface', 'computer science'
}

# Common degree abbreviations
DEGREE_ABBREVIATIONS = {
    'b.s.': "bachelor's degree",
    'bs': "bachelor's degree",
    'b.s': "bachelor's degree",
    'ba': "bachelor's degree",
    'b.a.': "bachelor's degree",
    'b.a': "bachelor's degree",
    'bachelors': "bachelor's degree",
    'bachelor': "bachelor's degree",
    'm.s.': "master's degree",
    'ms': "master's degree",
    'm.s': "master's degree",
    'ma': "master's degree",
    'm.a.': "master's degree",
    'm.a': "master's degree",
    'masters': "master's degree",
    'master': "master's degree",
    'mba': "master's degree",
    'm.b.a.': "master's degree",
    'phd': 'doctorate',
    'ph.d.': 'doctorate',
    'ph.d': 'doctorate',
}

# Months and irrelevant time-related words
MONTHS = {
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'
}

# Normalize degree abbreviations in text
def normalize_abbreviations(text: str) -> str:
    text_lower = text.lower()
    for abbr, full_form in DEGREE_ABBREVIATIONS.items():
        # Match abbreviation with word boundaries
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text_lower = re.sub(pattern, full_form, text_lower)
    return text_lower

# Tokenize text into words
def tokenize(text: str) -> List[str]:
    # Normalize abbreviations first
    text = normalize_abbreviations(text)
    return word_tokenize(text)

# Determine if a word is relevant based on its POS tag
def is_relevant_word(word: str, pos: str) -> bool:
    word_lower = word.lower()
    
    # Filter out months
    if word_lower in MONTHS:
        return False
    
    # Filter out most verbs (VB*) and adjectives (JJ*) unless they're technical terms
    if pos.startswith('VB') or pos.startswith('JJ'):
        # Keep if it's a CS keyword
        if word_lower not in CS_KEYWORDS and not any(word_lower in kw for kw in CS_KEYWORDS if ' ' not in kw):
            return False
    
    # Filter out adverbs
    if pos.startswith('RB'):
        return False
    
    return True

# Extract CS keywords from tokens
def extract_cs_keywords(tokens: List[str]) -> List[str]:
    cs_tokens = []
    
    # Check for multi-word CS terms first
    text = ' '.join(tokens).lower()
    multi_word_terms = [kw for kw in CS_KEYWORDS if ' ' in kw]
    
    found_multi_word = set()
    for term in multi_word_terms:
        if term in text:
            found_multi_word.add(term)
    
    # Add multi-word terms
    cs_tokens.extend(found_multi_word)
    
    # Check single-word tokens
    for token in tokens:
        token_lower = token.lower()
        # Check if token is a CS keyword or part of one
        if token_lower in CS_KEYWORDS:
            cs_tokens.append(token_lower)
        # Also check for partial matches (e.g., "scikit" in "scikit-learn")
        elif any(token_lower in kw for kw in CS_KEYWORDS if ' ' not in kw and len(token_lower) > 3):
            cs_tokens.append(token_lower)
    
    return cs_tokens

# Remove stopwords from tokens with optional POS filtering
def remove_stopwords(tokens: List[str], language: str = 'english', filter_pos: bool = True) -> List[str]:
    stop_words = set(stopwords.words(language))
    
    # Get POS tags if filtering
    if filter_pos:
        tagged = pos_tag(tokens)
        filtered = []
        for word, pos in tagged:
            word_lower = word.lower()
            if (word_lower not in stop_words and 
                len(word) > 1 and 
                is_relevant_word(word, pos)):
                filtered.append(word_lower)
        return filtered
    else:
        return [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 1]

def tokenize_and_remove_stopwords(text: str, language: str = 'english', 
                                  cs_only: bool = True, filter_pos: bool = True) -> List[str]:
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens, language, filter_pos)
    
    cs_tokens = extract_cs_keywords(filtered)
    return cs_tokens
    

