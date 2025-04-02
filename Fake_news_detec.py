import pandas as pd
import re
import os
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('wordnet')

model_path = "fake_news_model.h5"
max_length = 500
tokenizer = Tokenizer(num_words=5000)

def clean_text(text):
    """Clean the input text by removing stopwords, lemmatizing, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)



def load_model_or_train():
    """Load a pre-trained model or train a new one if it doesn't exist."""
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        return load_model(model_path)
    
    print("Training a new model...")

    fake = pd.read_csv("Fake_news_Detection/Fake.csv")
    true = pd.read_csv("Fake_news_Detection/True.csv")

    min_samples = min(len(fake), len(true))
    fake = fake.sample(min_samples)
    true = true.sample(min_samples)

    fake['label'] = 0
    true['label'] = 1

    data = pd.concat([fake, true], axis=0).reset_index(drop=True)
    data['text'] = data['text'].apply(clean_text)

    X = data['text']
    y = data['label']

    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post')

    X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1)
    model.save(model_path)  
    print("Model trained and saved successfully!")

    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    print("Validation Results:")
    print(classification_report(y_val, y_pred))

   
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return model

def predict_news(text, model, threshold=0.5):
    """Predict whether the news is real or fake."""
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    print(f"Prediction probability: {prediction}")
    return "This news is REAL" if prediction > threshold else "This news is FAKE"

def get_text_from_url(url):
    """Fetch and extract text content from a URL."""
    try:
        response = requests.get(url, timeout=10)  
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

def process_input(input_data, model):
    """Process the input data (URL, file, or text) and predict news authenticity."""
    if input_data.startswith('http'):
        print("Checking the URL...")
        text = get_text_from_url(input_data)
        if text:
            print(predict_news(text, model))
        else:
            print("Unable to extract content from the URL.")
    elif os.path.isfile(input_data):
        print("Reading from file...")
        try:
            with open(input_data, 'r', encoding='utf-8') as file:
                text = file.read()
                print(predict_news(text, model))
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("Checking the text...")
        print(predict_news(input_data, model))

def start_program():
    """Start the program and handle user inputs."""
    print("Fake News Detection Program")
    while True:
        print("\nEnter a news article (text, file path, or URL) or type 'exit' to stop:")
        user_input = input().strip()
        if user_input.lower() == 'exit':
            print("Exiting program...")
            break
        elif not user_input:
            print("Input cannot be empty. Please try again.")
        else:
            process_input(user_input, model)


model = load_model_or_train()

print(f"Model is starting")
start_program()