from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import spacy
import pt_core_news_md
from nltk.stem import RSLPStemmer
import nltk
import random
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load


nltk.download('rslp')

spacyPT = pt_core_news_md.load()


df = pd.read_csv('dominio.csv', encoding='utf-8', sep=',')
df = df[['responses', 'dominio']]
df = df.dropna()  # Remove linhas com valores ausentes

# Pré-processamento

stemmer = RSLPStemmer()
vectorizer = CountVectorizer()


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


# print(df.head()
processed_texts = df['responses'].apply(preprocess_text)
X = vectorizer.fit_transform(processed_texts)

y = df['dominio']


model = MultinomialNB()
model.fit(X, y)
modelo_completo = {
    'model': model,
    'vectorizer': vectorizer,
    'stemmer': stemmer,
    'spacyPT': spacyPT
}

# Salvar apenas objetos serializáveis
dump(modelo_completo, 'modelo_dominio.joblib')
