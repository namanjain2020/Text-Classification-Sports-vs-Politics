import re 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(ENGLISH_STOP_WORDS)

def cleaned_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    processed_data = []
    
    for token in tokens:
        if token in STOPWORDS:
            continue
        
        lemma = lemmatizer.lemmatize(token)
        processed_data.append(lemma)
    
    return " ".join(processed_data)
    
def preprocess_corpus(texts):
    return [cleaned_text(t) for t in texts]