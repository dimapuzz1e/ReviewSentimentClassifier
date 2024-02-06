# Data Science Project: Movie Review Sentiment Classification

# DS Part

## Project Overview

This data science project focuses on binary sentiment classification of movie reviews, aiming to categorize reviews as either positive or negative. Utilizing a dataset comprising 50,000 reviews, a comprehensive approach incorporating diverse data science and machine learning techniques was employed. This README outlines key steps in the project, including exploratory data analysis (EDA), feature engineering, model selection, performance evaluation, and potential business applications.

## Insights from Exploratory Data Analysis (EDA)

- **Balanced Dataset:** The dataset exhibits a well-balanced distribution of positive and negative reviews, ensuring unbiased model training.
- **Length Analysis:** Review lengths vary, but most fall within a moderate range, eliminating the need for extensive text trimming.
- **Word Frequency Analysis:** Unique patterns in frequently used words were identified for positive and negative reviews, indicating the prevalence of sentiment-driven language.

## Feature Engineering Description

- **Text Preprocessing:** Applied cleaning steps such as URL removal, non-alphabetic character removal, lowercasing, and stop word elimination to focus on relevant text.
- **Tokenization:** Reviews were broken down into individual words for subsequent analysis.
- **Stemming and Lemmatization Comparison:** After evaluation, stemming was chosen for its efficiency in reducing words to their base forms, aligning with analytical goals.
- **Vectorization with TF-IDF and Stemming:** TF-IDF Vectorizer combined with stemming effectively captures word importance in documents, striking a balance between word frequency and relevance.

## Considerations in Model Selection

- **Model Exploration:** Tested various models (Logistic Regression, SVM, Random Forest, and Naive Bayes), evaluating them based on accuracy, precision, recall, and F1 score.
- **Optimal Model - Logistic Regression with TF-IDF + Stemming:** Logistic Regression stood out for superior performance, especially when combined with TF-IDF vectorization and stemming. This combination excelled in handling high-dimensional data, balancing accuracy and computational efficiency.

### Decision Criteria

Considering the detailed analysis of performance metrics, model interpretability, computational efficiency, and suitability for sentiment analysis, Logistic Regression emerges as the preferred choice for movie review sentiment classification. Its ability to handle imbalanced data, provide interpretable outputs, and efficient resource utilization align well with the nuanced requirements of the task at hand.

## Potential Business Applications and Value

- **Customer Insight Enhancement:** Automated sentiment analysis of customer reviews for deeper satisfaction and preference understanding.
- **Product Review Analysis:** Streamlined processing of product reviews for informed product development and marketing strategies.
- **Market Research:** Leveraging sentiment analysis for social media and market trend analysis, offering insights into public opinion and consumer behavior.
- **Content Moderation:** Automated moderation of user-generated content on platforms by identifying negative sentiments.

## Conclusion

This sentiment classification project showcases sophisticated data science methodologies, specifically highlighting the successful implementation of TF-IDF with stemming and the Logistic Regression model. The outcomes of the project carry substantial implications for businesses aiming to enhance customer experience and conduct market analysis. The Logistic Regression model, with its ability to handle non-linear relationships and capture more complex patterns in the data, opens new avenues for insightful applications across diverse domains.

# ML Part

## Repository Structure

```
/ReviewClassification/

|--data/ #This folder is ignored

|   |--raw/ #This folder stores raw data

|   |--processed/ #This folder stores the prepared data

|--notebooks/ #A notebook with detailed EDA is stored in this folder

|--src/ #Source code directory.

|   |--train/ #Docker and train script for training are stored in this folder

|   |   |--train.py #Script for training the model.

|   |   |--Dockerfile #Docker configuration for training.

|   |--inference/ #Docker and train script for inference are stored in this folder

|   |   |--inference.py #Script for model inference.

|   |   |--Dockerfile #Docker configuration for inference.

|   |--data_load.py #Script for downloading data

|   |--data_preparation.py #Script for data preparation

|--outputs/ #This folder is ignored

|   |--models/ #This folder stores trained models

|   |   |--model_1.pkl #Example trained model.

|   |--predictions/ #This folder stores model predictions and their metrics

|   |   |--predictions.csv #CSV file containing model predictions.

|   |   |--metrics.txt #Text file with metrics related to the model's performance.

|   |--vectors/ #This folder stores the vectorizer for the same data preparation

|--README.MD #Documentation file providing information about the project.

|--requirements.txt #File specifying the dependencies required to run the project.

|--settings.json #Configuration file for project settings.
```

## Project Build Instructions

### Cloning the Repository

To clone the repository from GitHub, execute the following command:

```bash
git clone https://github.com/dimapuzz1e/ReviewSentimentClassifier
```

### Data Load

Navigate to the `ReviewSentimentClassifier` folder:

```bash
cd ReviewSentimentClassifier
```

If you possess the requisite files for model training and inference, create a `data` folder:

```bash
mkdir Data
```

Then, create a `raw` folder within it:

```bash
cd Data
mkdir raw
```

Move your files into the `raw` folder.

If you lack the necessary files, use the `data_load.py` script:

```bash
python3 src/data_load.py
```

This script loads files into the `data/raw` directory using the link from the `settings.json` file, naming them `train.csv` and `inference.csv` respectively.

### Data Preparation

To prepare the data, execute the `data_preparation.py` script:

```bash
python3 src/data_preparation.py
```

In the `settings.json` file, specify the files to be prepared by setting True or False, and specify their names. By default, these are the `train.csv` and `inference.csv` files loaded using the previous script.

### Training

Customize training parameters in the `settings.json` file, such as model name, data file for training, test_size, and the hyperparameter C.

#### Docker Training

1. Build the training Docker image:

```bash
docker build -f src/train/Dockerfile -t training-image .
```

2. Run the container to train the model:

```bash
docker run -v ${PWD}/outputs:/app/outputs training-image
```

#### Local Training

Alternatively, run the `train.py` script locally:

```bash
python3 src/train/train.py
```

### Inference

After training, the model can be used for predictions on new data in the inference stage. Customize inference parameters in the `settings.json` file, such as model name and data file for inference.

#### Docker Inference

1. Build the inference Docker image:

```bash
docker build -t inference-image -f src/inference/Dockerfile .
```

2. Run the inference Docker container:

```bash
docker run -v ${PWD}/outputs:/app/outputs inference-image
```

The inference results will be stored in the designated folder on your local machine.

#### Local Inference

Alternatively, run the inference script locally:

```bash
python src/inference/inference.py
```
