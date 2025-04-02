# Import Required Libraries
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from hazm import Normalizer, word_tokenize, Stemmer, stopwords_list
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Check for GPU
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the Dataset
train_data = pd.read_csv('train.csv') 
test_data = pd.read_csv('test.csv')

# Data Exploration
train_data.info()
test_data.info()
train_data['recommendation_status'].value_counts()

# Handle Missing Values and Encode Labels
train_data['recommendation_status'] = train_data['recommendation_status'].fillna("no_idea")
label_map = {"no_idea": 2, "recommended": 1, "not_recommended": 0}
train_data['recommendation_status'] = train_data['recommendation_status'].map(label_map)

# Verify preprocessing
train_data["recommendation_status"].unique()
train_data["recommendation_status"].value_counts()

# Text Preprocessing
stopwords = set(stopwords_list())
normalizer = Normalizer()
stemmer = Stemmer()

punctuations = r'[!()-\[\]{};:\'",؟<>./?@#$%^&*_~]'
numbers_regex = r'[۰-۹\d]+'
white_space = r'\s+'

def preprocess_text(text):
    text = normalizer.normalize(str(text))
    text = re.sub(r'[۰-۹\d]+', '', text)
    text = re.sub(r'[!()-\[\]{};:\'",؟<>./?@#$%^&*_~]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return [stemmer.stem(token) for token in tokens if token not in stopwords and token.strip()]

train_data['preprocess'] = train_data['body'].apply(preprocess_text)

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['preprocess'])
sequences = tokenizer.texts_to_sequences(train_data['preprocess'])
max_len = max(map(len, sequences))
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = train_data['recommendation_status'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model with GPU
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Prediction Function
def predict_recommendation(comment):
    preprocessed_comment = preprocess_text(comment)
    seq = tokenizer.texts_to_sequences([preprocessed_comment])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded_seq)
    return {v: k for k, v in label_map.items()}[np.argmax(prediction)]

# Test Prediction
print(predict_recommendation("نمیدونم"))


def predict_sentiments_for_file(input_file, output_file, summary_file, model_accuracy=None):
    try:
        comments_df = pd.read_csv(input_file, header=None, names=['comment'])
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    results = []
    error_count = 0
    
    for comment in tqdm(comments_df['comment'], desc="Predicting sentiments"):
        try:
            sentiment = predict_recommendation(comment)
            results.append({'comment': comment, 'sentiment': sentiment})
        except Exception as e:
            print(f"Error predicting sentiment for '{comment}'. : {e}")
            results.append({'comment': comment, 'sentiment': 'error'})
            error_count += 1
    
    results_df = pd.DataFrame(results)
    
    sentiment_counts = results_df['sentiment'].value_counts()
    total_comments = len(results_df)
    
    summary_data = {
        'Sentiment': [
            'number of comments',
            'recommended',
            'not recommended', 
            'no idea',
            'number of errors',
            'model accuracy (%)'
        ],
        'Number': [
            total_comments,
            sentiment_counts.get('recommended', 0),
            sentiment_counts.get('not_recommended', 0),
            sentiment_counts.get('no_idea', 0),
            error_count,
            '-' 
        ],
        'Percentage': [
            100,
            round(sentiment_counts.get('recommended', 0) / total_comments * 100, 2),
            round(sentiment_counts.get('not_recommended', 0) / total_comments * 100, 2),
            round(sentiment_counts.get('no_idea', 0) / total_comments * 100, 2),
            round(error_count / total_comments * 100, 2),
            round(model_accuracy * 100, 2) if model_accuracy is not None else '-'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    try:
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"results saved in '{output_file}' .")
        
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"result summary saved in '{summary_file}'.")
        
        print("\nresults summary:")
        print(summary_df.to_string(index=False))
        
    except Exception as e:
        print(f"Error saving results: {e}")

input_csv = 'comments.csv'
output_csv = 'sentiment_results.csv'
summary_csv = 'sentiment_summary.csv'


predict_sentiments_for_file(input_csv, output_csv, summary_csv, model_accuracy=accuracy)

#Github : RezaGooner
