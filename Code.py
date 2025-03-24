# Import Required Libraries
import pandas as pd
from hazm import Normalizer, word_tokenize, Stemmer, stopwords_list
import re
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Dataset
train_data = pd.read_csv('train.csv') 
test_data = pd.read_csv('test.csv')

# Data Exploration
train_data.info()
test_data.info()
train_data['recommendation_status'].value_counts()

# Handle Missing Values and Encode Labels
train_data["recommendation_status"] = train_data["recommendation_status"].fillna("no_idea")

valid_statuses = {"no_idea", "recommended", "not_recommended"}
train_data["recommendation_status"] = train_data["recommendation_status"].apply(
    lambda x: x if x in valid_statuses else "no_idea"
)

train_data["recommendation_status"] = train_data["recommendation_status"].map({
    "no_idea": 2,
    "recommended": 1,
    "not_recommended": 0
})

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
    text = re.sub(numbers_regex, '', text)
    text = re.sub(punctuations, ' ', text)
    text = re.sub(white_space, ' ', text).strip()
    
    tokens = word_tokenize(text)
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stopwords and token.strip()
    ]
    
    return processed_tokens

# Test preprocessing function
exmpale = "من متولد سال ۱۳۷۷ هستم"
preprocess_text(exmpale)

# Apply preprocessing to all data
dataes = train_data['body']

def process_chunks(series, chunk_size=1000):
    chunks = [series[i:i + chunk_size] for i in range(0, len(series), chunk_size)]
    processed_data = []
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        processed_chunk = chunk.apply(preprocess_text)
        processed_data.extend(processed_chunk)
    
    return pd.Series(processed_data)

data_processed = process_chunks(dataes)
train_data["preprocess"] = data_processed
train_data.head()

# Word2Vec Embedding
model = Word2Vec(sentences=train_data["preprocess"], vector_size=100, window=5, min_count=1, workers=4)

# Test Word2Vec
model.wv.most_similar("دوست")

# Sentence Vectorization
def sentence_vector(sentence):
    vectors = []
    for word in sentence:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            vectors.append(np.zeros(100))
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

sentence_vectors = train_data['preprocess'].apply(sentence_vector)
sentence_vectors

# Prepare Data for Model
X = np.array(sentence_vectors.to_list())
y = train_data["recommendation_status"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluate Model
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Prediction Function
def predict_recommendation(comment):
    preprocessed_comment = preprocess_text(comment)
    sentence_vector_comment = sentence_vector(preprocessed_comment)
    X_comment = np.array([sentence_vector_comment])
    prediction = logistic_model.predict(X_comment)
    if prediction[0] == 2:
        return "no_idea"
    elif prediction[0] == 1:
        return "recommended"
    else:
        return "not_recommended"

# Test Prediction
new_comment = 'نمیدونم'
predict_recommendation(new_comment)

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
