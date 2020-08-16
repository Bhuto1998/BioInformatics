# Starting with the imports
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional, Flatten, Conv1D, \
    GlobalMaxPooling1D
from keras.layers import Dense

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential



# Input file name
input_file = "test.csv"

# Constants
EPOCHS = 100
BATCH_SIZE = 128
INPUT_DIM = 4
OUTPUT_DIM = 50
RNN_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.2
MAXLEN = 100


# Loading the Dataset
def letter_to_index(letter):
    _alphabet = 'ATGC'
    letter = letter.upper()
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)


def load_data(test_split=0.001, maxlen=MAXLEN):
    print('Loading data...')
    df = pd.read_csv(input_file)
    df['Sequences'] = df['Sequences'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = df['Sequences'].values[:train_size]
    y_train = np.array(df['Target'].values[:train_size])
    X_test = np.array(df['Sequences'].values[train_size:])
    y_test = np.array(df['Target'].values[train_size:])
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test

X_train, y_train, X_test, y_test = load_data()
name = "FM-1_model-4.h5"
new_model = tf.keras.models.load_model(name)
loss, acc = new_model.evaluate(X_train,  y_train, verbose=1)

print("Test Loss: " + str(loss))
print("Test Accuracy: " + str(acc))
