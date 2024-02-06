import os
import json
import pickle
import logging
import pandas as pd

# Configuring logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings(settings_path='settings.json'):
    """
    Load settings from a JSON file.

    Parameters:
    - settings_path (str): Path to the JSON settings file.

    Returns:
    - settings (dict): Loaded settings from the file.
    """
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    logging.info(f"Settings loaded from {settings_path}")
    return settings

def load_saved_model(model_path):
    """
    Load a pre-trained model from a file.

    Parameters:
    - model_path (str): Path to the file containing the saved model.

    Returns:
    - model: Loaded pre-trained model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Model loaded from {model_path}")
    return model

def create_directory_if_not_exists(path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def perform_inference(model, input_data, outputs_dir):
    """
    Perform inference using the model and save the results.

    Parameters:
    - model: Model for performing inference.
    - input_data (pd.DataFrame): Data to be passed to the model for predictions.
    - outputs_dir (str): Directory to save the results.

    """
    # Drop the 'sentiment' column if present in the input data
    if 'sentiment' in input_data.columns:
        input_data = input_data.drop('sentiment', axis=1)

    # Get predictions from the model
    predictions = model.predict(input_data)

    # Create a directory to save predictions if it doesn't exist
    predictions_dir = os.path.join(outputs_dir, 'predictions')
    create_directory_if_not_exists(predictions_dir)

    # Path to the file with inference results
    results_path = os.path.join(predictions_dir, 'inference_results.csv')

    # Create a DataFrame with predictions and save to CSV
    results_df = pd.DataFrame({'prediction': predictions})
    results_df.to_csv(results_path, index=False)
    logging.info(f"Inference results saved to {results_path}")

def main():
    # Load settings
    settings = load_settings()

    # Load input data for inference
    inference_data_path = os.path.join(settings['config']['data_directory'], 'processed', settings['inference']['input_file_name'])
    inference_data = pd.read_csv(inference_data_path)

    # Load the pre-trained model
    model_path = os.path.join(settings['config']['outputs_directory'], settings['config']['models_directory'], settings['inference']['model_name'])
    pre_trained_model = load_saved_model(model_path)

    # Perform inference and save the results
    perform_inference(pre_trained_model, inference_data, settings['config']['outputs_directory'])

if __name__ == "__main__":
    main()
