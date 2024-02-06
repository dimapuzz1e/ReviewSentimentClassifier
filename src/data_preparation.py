import os
import re
import nltk
import json
import pickle
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')

def load_configuration(config_path='settings.json'):
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration settings.
    """
    with open(config_path, 'r') as file:
        settings = json.load(file)
    logging.info(f"Settings loaded from {config_path}")
    return settings

def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Directory path.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

def preprocess_review(review_text):
    """
    Clean and preprocess review text.

    Args:
        review_text (str): Raw review text.

    Returns:
        str: Cleaned and processed review text.
    """
    review_text = re.sub(r'http\S+', '', review_text)
    review_text = re.sub('[^a-zA-z]', ' ', review_text)
    review_text = review_text.lower()
    tokens = word_tokenize(review_text)
    stop_words_set = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_set and word not in ['film', 'movie', 'br', 'one']]
    return ' '.join(stem_tokens)

def prepare_data(file_path, settings, vectorizer=None, is_train_data=True):
    """
    Preprocess and vectorize the data.

    Args:
        file_path (str): Path to the data file.
        settings (dict): Configuration settings.
        vectorizer (TfidfVectorizer): TfidfVectorizer object for text vectorization.
        is_train_data (bool): Indicates whether the data is for training.

    Returns:
        pd.DataFrame: Processed and vectorized data.
    """
    logging.info(f"Preparing data from {file_path}.")
    data = pd.read_csv(file_path)
    data['CleanReview_Stem'] = data['review'].apply(preprocess_review)
    
    # If this is train data, fit the vectorizer
    if is_train_data:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf_stem = vectorizer.fit_transform(data['CleanReview_Stem'])
        # Save the vectorizer for later use in inference
        vectorizer_path = os.path.join(settings['config']['outputs_directory'], 'vectors')
        create_directory(vectorizer_path)
        with open(os.path.join(vectorizer_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Replace sentiment with 0 and 1
        data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    else:
        # Load the vectorizer used in training for consistent feature spaces
        with open(os.path.join(settings['config']['outputs_directory'], 'vectors', 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        X_tfidf_stem = vectorizer.transform(data['CleanReview_Stem'])

    logging.info("TF-IDF vectorization complete.")

    processed_data_path = os.path.join(settings['config']['data_directory'], 'processed')
    create_directory(processed_data_path)
    
    # Convert sparse matrix to DataFrame to save in CSV
    tfidf_df = pd.DataFrame(X_tfidf_stem.toarray(), columns=vectorizer.get_feature_names_out())
    
    if is_train_data:
        # Include the target variable "sentiment"
        tfidf_df['sentiment'] = data['sentiment']
        
        suffix = 'train'
    else:
        if 'sentiment' in data.columns:
            tfidf_df['sentiment'] = data['sentiment']
        suffix = 'inference'
    
    output_file = os.path.join(processed_data_path, f'processed_{suffix}.csv')
    tfidf_df.to_csv(output_file, index=False)
    logging.info(f"Processed data saved to {output_file}")
    return tfidf_df

# Load configuration settings
settings = load_configuration()

# Prepare data for training
if settings['data_processing']['prepare_train_data']:
    train_file_path = os.path.join(settings['config']['data_directory'], 'raw', settings['data_processing']['train_file_to_preparation'])
    prepare_data(train_file_path, settings, is_train_data=True)

# Prepare data for inference
if settings['data_processing']['prepare_inference_data']:
    inference_file_path = os.path.join(settings['config']['data_directory'], 'raw', settings['data_processing']['inference_file_to_preparation'])
    prepare_data(inference_file_path, settings, is_train_data=False)
