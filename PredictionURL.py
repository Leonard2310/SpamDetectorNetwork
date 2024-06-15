import subprocess
import sys

# Function to install a package using pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uninstall current TensorFlow and install the specific version
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y"])
install_package("tensorflow==2.15.0")
install_package("langdetect")
install_package("googletrans==4.0.0-rc1")
install_package("transformers")

import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
from googletrans import Translator
from transformers import DistilBertTokenizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Define the model directory
MODEL_DIR = '/Users/l.catello/Library/Mobile Documents/com~apple~CloudDocs/Magistrale Ingegneria Informatica/Cognitive and Computing System/SpamDetection/Model/'

# Initialize Flask app
app = Flask(__name__)

# Load the TensorFlow SavedModel
load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
model = tf.saved_model.load(MODEL_DIR, options=load_options)

# Function to translate text to English
def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang == 'en':
            return text
    except LangDetectException as e:
        print(f"Error detecting language: {e}")
        return text

    translator = Translator()
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    try:
        for chunk in chunks:
            translated_chunk = translator.translate(chunk, src='auto', dest='en').text
            translated_chunks.append(translated_chunk)
        translated_text = ' '.join(translated_chunks)
        return translated_text
    except Exception as e:
        print(f"Translation failed for text: {text}. Error: {e}")
        return text

# Function to get WordNet POS tag
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Text preprocessing function
def preprocess_text(text):
    translated_text = translate_to_english(text)
    clean_text = re.sub(r'[^\w\s]', '', translated_text)
    clean_text = re.sub(r'\b(in|the|all|for|and|on)\b', '_connector_', clean_text)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer.tokenize(clean_text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_tokens]

    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    processed_text = preprocess_text(text)

    # Perform prediction using the loaded model
    infer = model.signatures["serving_default"]
    prediction = infer(tf.constant([processed_text]))["output"].numpy()[0]
    predicted_class = np.argmax(prediction)

    response = {'prediction': int(predicted_class)}
    return jsonify(response)

# Run the Flask server
if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)