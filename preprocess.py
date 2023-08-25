# preprocess.py
import numpy as np

def tokenize(sentence):
    # Implement your tokenization logic here
    tokens = sentence.split()  # Split sentence into words
    return tokens

def encode_sentence(tokens, emoji_dict):
    encoded = [emoji_dict.get(token, -1) for token in tokens]  # Encode tokens using emoji dictionary
    return np.array(encoded)

# Load data.csv and preprocess sentences and labels
# Save the preprocessed data into X_train (input sentences) and Y_train (emoji labels)
