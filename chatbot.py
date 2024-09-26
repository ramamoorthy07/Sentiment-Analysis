from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from flask_cors import CORS  # Import CORS

# Load the fine-tuned model and tokenizer outside of the request to optimize performance
model = TFBertForSequenceClassification.from_pretrained('model/sentiment_model')
tokenizer = BertTokenizer.from_pretrained('model/sentiment_model')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    data = request.get_json()
    
    # Check if the required key is present
    if 'text' not in data:
        return jsonify({'error': 'Text input is required.'}), 400

    text = data['text']

    # Tokenize input
    try:
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    except Exception as e:
        return jsonify({'error': 'Tokenization error: ' + str(e)}), 400

    # Get model predictions
    try:
        predictions = model(inputs)
        logits = predictions.logits.numpy()
        predicted_class = np.argmax(logits, axis=1)[0]
    except Exception as e:
        return jsonify({'error': 'Model prediction failed: ' + str(e)}), 500
    
    # Convert prediction to sentiment
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
