# Sentiment-Analysis
Sentiment Analysis with BERT &amp; Flask API This project uses BERT for sequence classification (sentiment analysis) on a labeled text dataset and serves predictions through a Flask API. The model is fine-tuned using TensorFlow and Hugging Face's Transformers, and deployed as a web service to classify input text as Positive or Negative.

## Table of Contents
1. Introduction
2. Features
3. Requirements
4. Installation
5. Usage
6. API Endpoints
7. License

## 1. Introduction
Sentiment analysis is a natural language processing task that aims to determine the sentiment expressed in a piece of text. This project uses a fine-tuned BERT model to achieve high accuracy in classifying sentiment.

## 2. Features
- Fine-tuned BERT model for effective sentiment classification.
- Simple Flask web application for serving predictions.
- Easy tokenization and preprocessing of input text.

## 3. Requirements
- Python 3.7 or higher
- TensorFlow 2.x
- Hugging Face Transformers
- Flask
- Pandas
- NumPy
- Scikit-learn

## 4. Installation
To set up the project, follow these steps:

1. Clone the repository:
git clone https://github.com/ramamoorthy07/Sentiment-Analysis.git


2. Install the required packages:
pip install -r requirements.txt


## 5. Usage
1. Ensure you have your dataset `MovieReviews.csv` in the project directory.
2. Run the training script to fine-tune the model: python fine_tune_model.py
3. Start the Flask application: python chatbot.py
4. The application will run on `http://127.0.0.1:5000/`. You can access it through your web browser.

## 6. API Endpoints
### POST /predict
This endpoint accepts JSON input with the text to be analyzed.
**Request body:**
```json
{
  "text": "Your message here"
}
```
Response:
```
{
    "sentiment": "Positive"
}
```
