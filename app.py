import numpy as np
import json
import random
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
import nltk

# Load the model and preprocessing data
model = load_model('models/chatbot_improvemodel_functional.h5')
tags = np.load('preprocessing/tags.npy', allow_pickle=True)
all_words = np.load('preprocessing/all_words.npy', allow_pickle=True)

stemmer = PorterStemmer()
nltk.download('punkt')

app = Flask(__name__)

def preprocess_input(sentence):
    # Tokenize and stem input sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words if word.isalnum()]

    # Create a bag of words for the sentence
    bag = np.array([1 if w in sentence_words else 0 for w in all_words])
    return np.array([bag])

def get_response(prediction):
    tag_index = np.argmax(prediction)
    tag = tags[tag_index]

    with open('dataset/intents.json') as file:
        intents = json.load(file)

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data['message']

    # Preprocess the input
    bag_of_words = preprocess_input(message)

    # Get model prediction
    prediction = model.predict(bag_of_words)

    # Get response based on prediction
    response = get_response(prediction)

    return jsonify({'response': response})
@app.route("/")
def home():
    message = "ChatBot Home page. Welcome to home of Intelligence"
    return message
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
