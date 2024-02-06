import os
import json
import pickle
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings(settings_path='settings.json'):
    """
    Load configuration settings from a JSON file.

    Args:
        settings_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration settings.
    """
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    logging.info(f"Settings loaded from {settings_path}")
    return settings

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def train_logistic_regression(X_train, y_train, regularization_parameter):
    """
    Train a Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.
        regularization_parameter (float): Regularization parameter.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(C=regularization_parameter, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model.

    Args:
        model: Trained classification model.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): True labels for testing.

    Returns:
        tuple: Evaluation metrics (accuracy, recall, precision, f1 score, classification report).
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Recall: {recall}")
    logging.info(f"Precision: {precision}")
    logging.info(f"F1 Score: {f1}")
    logging.info("\nClassification report:\n" + classification_rep)

    return accuracy, recall, precision, f1, classification_rep

def save_trained_model(model, model_path):
    """
    Save the trained model to a file.

    Args:
        model: Trained model.
        model_path (str): Path to save the model.
    """
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {model_path}")

def save_evaluation_metrics(accuracy, recall, precision, f1, classification_rep, metrics_path):
    """
    Save the evaluation metrics to a file.

    Args:
        accuracy (float): Model accuracy.
        recall (float): Recall metric.
        precision (float): Precision metric.
        f1 (float): F1 score metric.
        classification_rep (str): Classification report.
        metrics_path (str): Path to save the metrics.
    """
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write("\nClassification report:\n" + classification_rep)
    logging.info(f"Metrics saved to {metrics_path}")

def create_directory(path):
    """
    Create a directory if it does not exist.

    Args:
        path (str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")
    else:
        logging.info(f"Directory already exists: {path}")

def main():
    # Load configuration settings
    settings = load_settings()

    # Load and preprocess data
    processed_data_path = os.path.join(settings['config']['data_directory'], 'processed')
    train_data_path = os.path.join(processed_data_path, f"processed_{settings['data_processing']['train_file_to_preparation'].replace('.csv', '')}.csv")
    data = load_data(train_data_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('sentiment', axis=1), data['sentiment'], test_size=settings['training']['test_size'], random_state=settings['config']['seed'])

    # Train the Logistic Regression model
    logging.info("Training the model...")
    best_model = train_logistic_regression(X_train, y_train, settings['training']['regularization_parameter'])

    # Evaluate the model
    logging.info("Evaluating the model...")
    accuracy, recall, precision, f1, classification_rep = evaluate_classification_model(best_model, X_test, y_test)

    # Save the trained model
    model_name = settings['training']['model_name'].replace('.pkl', '') + '.pkl'
    model_path = os.path.join(settings['config']['outputs_directory'], 'models', model_name)
    save_trained_model(best_model, model_path)

    # Save evaluation metrics
    metrics_dir = os.path.join(settings['config']['outputs_directory'], 'predictions')
    create_directory(metrics_dir)
    metrics_name = f"metrics_{model_name.replace('.pkl', '')}.txt"
    metrics_path = os.path.join(metrics_dir, metrics_name)
    save_evaluation_metrics(accuracy, recall, precision, f1, classification_rep, metrics_path)

if __name__ == "__main__":
    main()
