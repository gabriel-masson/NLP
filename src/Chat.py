from joblib import dump, load
import json
import random

import json
import random
from joblib import load


class Chat:
    def __init__(self):
        self.model = load('modelo.joblib')
        with open('intents.json', 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Acessa os componentes do modelo
        self.spacy = self.model['spacyPT']
        self.stemmer = self.model['stemmer']
        self.vectorizer = self.model['vectorizer']
        self.classifier = self.model['model']

    def preprocess_text(self, text):
        doc = self.spacy(text)
        tokens = [
            token.text for token in doc if not token.is_stop and not token.is_punct]
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return " ".join(stemmed)

# generate a response based on the user's input
    def answer(self, ask):
        text = self.preprocess_text(ask)
        vec = self.vectorizer.transform([text])
        tag_predict = self.classifier.predict(vec)[0]

        for intent in self.data['intents']:
            if intent['tag'] == tag_predict:
                return random.choice(intent['responses'])
