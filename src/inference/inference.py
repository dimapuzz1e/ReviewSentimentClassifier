import os
import pickle
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings(settings_path='settings.json'):
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    logging.info(f"Settings loaded from {settings_path}")
    return settings

def load_saved_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Model loaded from {model_path}")
    return model

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def perform_inference(model, input_data, outputs_dir):
    if 'sentiment' in input_data.columns:
        input_data = input_data.drop('sentiment', axis=1)

    predictions = model.predict(input_data)

    predictions_dir = os.path.join(outputs_dir, 'predictions')
    create_directory_if_not_exists(predictions_dir)

    results_path = os.path.join(predictions_dir, 'inference_results.csv')
    results_df = pd.DataFrame({'prediction': predictions})
    results_df.to_csv(results_path, index=False)
    logging.info(f"Inference results saved to {results_path}")

def main():
    settings = load_settings()

    inference_data_path = os.path.join(settings['config']['data_directory'], 'processed', settings['inference']['input_file_name'])
    inference_data = pd.read_csv(inference_data_path)

    model_path = os.path.join(settings['config']['outputs_directory'], settings['config']['models_directory'], settings['inference']['model_name'])
    pre_trained_model = load_saved_model(model_path)

    perform_inference(pre_trained_model, inference_data, settings['config']['outputs_directory'])

if __name__ == "__main__":
    main()