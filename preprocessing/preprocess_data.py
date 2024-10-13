import nltk
import json
import csv
import numpy as np
import random
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Initialize the NLTK tokenizer and stemmer

nltk.data.path.append('/home/iamnderitum/nltk_data')
nltk.download('punkt_tab')
stemmer = PorterStemmer()

"""# Load the intents JSON dataset
with open('dataset/intents.json') as file:
    intents = json.load(file)"""

# Load the intents CSV dataset
intents = []
with open("dataset/intents.csv", mode="r") as file:
    reader = csv.DictReader(file)
    # intents = []
    for row in reader:
        intent = {
            "tag": row["tag"],
            "patterns": row["patterns"].split(","),
            "responses": row["responses"].split(",")
        }
        intents.append(intent)

# Initialize lists to store words, labels, and the data
all_words = []
tags = []
xy = []

# Tokenize and stem words, add patterns and tags
for intent in intents:
    tag = intent['tag']
    tags.append(tag)

    # Check if patterns are a string or list and handle accordingly
    patterns = intent["patterns"]
    if isinstance(patterns, str):
        patterns = patterns.split(",")

    elif isinstance(patterns, list):
        # If patterns are in a list, we join  them into  a single string with a delimeter
        patterns = [pattern for sublist in patterns for pattern in sublist] # # Flatten the list if needed

    for pattern in patterns:
        # Tokenize each word
        words = nltk.word_tokenize(pattern)
        all_words.extend(words)

        # Add the tokenized pattern and its tag
        xy.append((words, tag))

# Stem and lower each word
all_words = [stemmer.stem(w.lower()) for w in all_words if w.isalnum()]
all_words = sorted(set(all_words))

# Sort the tags
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # Stem and tokenize words
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence if w.isalnum()]

    # Convert words to numerical values
    bag = [1 if w in pattern_words else 0 for w in all_words]

    X_train.append(bag)

    # Get the index of the tag
    label = tags.index(tag)
    y_train.append(label)

# Convert to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Save preprocessed data
np.save('preprocessing/X_train.npy', X_train)
np.save('preprocessing/y_train.npy', y_train)
np.save('preprocessing/tags.npy', tags)
np.save('preprocessing/all_words.npy', all_words)

# Display Useful Information
print("Saved X_train.npy, y_train.py, tags.npy, all_words.npy")
print("Preprocessing complete and data saved.")
print(f"Number of samples in X_train: {X_train.shape[0]}")
print(f"Number of samples in y_train: {y_train.shape[0]}")
print(f"Unique tags: {tags}")
print(f"Number of unique tags: {len(tags)}")
print(f"Sample of X_train data (first 5 samples): \n{X_train[:5]}")
print(f"Sample of y_train data (first 5 samples): \n{y_train[:5]}")
print(f"Unique words in the vocabulary: {len(all_words)}")
print(f"Sample of all words: {all_words[:10]}")  # Display first 10 words for brevity