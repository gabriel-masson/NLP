from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import spacy
import pt_core_news_md
from nltk.stem import RSLPStemmer
import nltk
import random
from sklearn.naive_bayes import MultinomialNB


nltk.download('rslp')

spacyPT = pt_core_news_md.load()


with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# Pré-processamento

stemmer = RSLPStemmer()
vectorizer = CountVectorizer()


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
    stem = [
        stemmer.stem(token) for token in tokens
    ]
    return " ".join(stem)  # Retorna uma string com os tokens processados


df = preprocess_data(data)
# print(df.head()
processed_texts = df['patterns'].apply(preprocess_text)
X = vectorizer.fit_transform(processed_texts)

y = df['tags']
# print(X)

model = MultinomialNB()
model.fit(X, y)


def responder(pergunta_usuario, vectorizer, model, data):
    pergunta_usuario = pergunta_usuario.lower()
    pergunta_processada = preprocess_text(pergunta_usuario)
    entrada = vectorizer.transform([pergunta_processada])
    tag_prevista = model.predict(entrada)[0]

    for intent in data['intents']:
        if intent['tag'] == tag_prevista:
            return random.choice(intent['responses'])


while True:
    pergunta = input("Você: ")
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        break
    resposta = responder(pergunta, vectorizer, model, data)
    print("Bot:", resposta)
