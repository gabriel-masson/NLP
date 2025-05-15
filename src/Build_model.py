from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import spacy
import pt_core_news_md
from nltk.stem import RSLPStemmer
import nltk
import random
from sklearn.ensemble import RandomForestClassifier


nltk.download('rslp')

spacyPT = pt_core_news_md.load()
vectorizer = CountVectorizer()


class Build_model:
    def __init__(self):
        with open('intents.json', 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.data = self.preprocess_data(self.dataset)
        self.model = None

    def preprocess_data(data):
        intents = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                intents.append((pattern, intent['tag']))
        return pd.DataFrame(intents, columns=['patterns', 'tags'])

    def preprocess_text(text):
        # Tokenize the text
        doc = spacyPT(text)
        # Remove stop words and punctuation
        tokens = [
            token.text for token in doc if not token.is_stop and not token.is_punct
        ]
        # stem the tokens
        stemmer = RSLPStemmer()
        stem = [
            stemmer.stem(token) for token in tokens
        ]
        # vectorizer = CountVectorizer()
        return vectorizer.fit_transform(stem).toarray()

    def save_model(self, path):
        # Implement the model saving logic here
        pass

    def load_model(self, path):
        # Implement the model loading logic here
        pass
