import nltk
import json
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

# Load the intents JSON dataset
with open('dataset/intents.json') as file:
    intents = json.load(file)

# Initialize lists to store words, labels, and the data
all_words = []
tags = []
xy = []

# Tokenize and stem words, add patterns and tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
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