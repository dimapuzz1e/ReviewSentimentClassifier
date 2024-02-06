import os
import json
import shutil
import zipfile
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_configuration(config_path='settings.json'):
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration settings.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def create_data_directory(directory_path):
    """
    Create a directory for storing data if it does not exist.

    Args:
        directory_path (str): Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)
    logging.info(f"Created directory: {directory_path}")

def fetch_and_extract_data(url, extraction_path, expected_csv_name, save_as):
    """
    Download a ZIP file from the given URL, extract its contents,
    and save the specified CSV file.

    Args:
        url (str): URL of the ZIP file.
        extraction_path (str): Directory to extract the ZIP contents.
        expected_csv_name (str): Name of the CSV file expected in the ZIP.
        save_as (str): Name to save the extracted CSV file as.
    """
    temp_zip_path = os.path.join(extraction_path, 'temp.zip')

    try:
        # Download ZIP file
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(temp_zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)

        # Extract ZIP contents
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)

        # Remove existing CSV file, if it exists
        csv_save_path = os.path.join(extraction_path, save_as)
        if os.path.exists(csv_save_path):
            os.remove(csv_save_path)

        # Move CSV files and delete directories
        for root, dirs, files in os.walk(extraction_path, topdown=False):
            for name in files:
                if name == expected_csv_name:
                    shutil.move(os.path.join(root, name), csv_save_path)
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

        logging.info(f"Downloaded and unzipped data from {url}. Saved CSV as {csv_save_path}")

    except Exception as e:
        logging.error(f"Error downloading and unzipping data from {url}: {str(e)}")

# Load configuration settings
config_settings = load_configuration()

# Paths for downloading data from configuration settings
data_directory = config_settings['config']['data_directory']
raw_data_directory = os.path.join(data_directory, 'raw')
create_data_directory(raw_data_directory)

# URLs for training and inference data from configuration settings
train_data_url = config_settings['data_processing']['train_data_url']
inference_data_url = config_settings['data_processing']['inference_data_url']

# Download and save training data
fetch_and_extract_data(train_data_url, raw_data_directory, 'train.csv', config_settings['data_processing']['train_file_to_preparation'])

# Download and save data for inference
fetch_and_extract_data(inference_data_url, raw_data_directory, 'test.csv', config_settings['data_processing']['inference_file_to_preparation'])
