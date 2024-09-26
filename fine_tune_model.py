import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('MovieReviews.csv', sep='\t', header=None, names=['label', 'text'])

# Inspect the data structure
print("Initial dataset:")
print(data.head())
print(data.columns)

# Preprocess data
data.columns = ['label', 'text']  # Adjust if necessary

# Map labels to 0 and 1
data['label'] = data['label'].map({'Positive': 1, 'Negative': 0})

# Check for missing or NaN labels
print(f"Missing labels: {data['label'].isnull().sum()}")  # Should be 0 after cleaning
if data['label'].isnull().sum() > 0:
    data = data.dropna(subset=['label'])

# Check label distribution
X = data['text'].values
y = data['label'].values

# Check label distribution
print(f"Label distribution: {np.unique(y, return_counts=True)}")  # Should show counts of both 0 and 1

# Split the dataset
if len(X) > 0 and len(y) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train-test split successful!")
    print(f"X_train size: {len(X_train)}, X_test size: {len(X_test)}")
else:
    print("Error: Not enough samples in the dataset to split!")
    exit()  # Stop execution if there are no samples to split

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize and encode the data
print("Tokenizing training data...")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors='tf')

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# Fine-tune the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Save the model and tokenizer
model.save_pretrained('model/sentiment_model')
tokenizer.save_pretrained('model/sentiment_model')

print("Model and tokenizer saved successfully!")
