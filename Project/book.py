#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load the trained model and the TF-IDF vectorizer
model = pickle.load(open('bookgenremodel.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfdifvector.pkl', 'rb'))

# Ensure necessary NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Preprocessing functions
def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text.lower()

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def lematizing(sentence):
    lemma = WordNetLemmatizer()
    return ' '.join([lemma.lemmatize(word) for word in sentence.split()])

def stemming(sentence):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in sentence.split()])

def preprocess_text(text):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)
    return text

def predict_genre(summary):
    genre_mapping = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction', 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    processed_text = preprocess_text(summary)
    text_vector = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    predicted_genre = genre_mapping[prediction[0]]
    return predicted_genre

# Streamlit app UI
st.title('Book Genre Classification')
st.write("This app predicts the genre of a book based on its summary.")

# User input
summary = st.text_area("Enter the book summary:", "")

# Prediction
if st.button("Predict Genre"):
    if summary:
        predicted_genre = predict_genre(summary)
        st.write(f"**Predicted Genre:** {predicted_genre}")
    else:
        st.write("Please enter a book summary for prediction.")

