# Starting with the imports
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.layers import GRU, SimpleRNN, Dense
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json

# Input file name
input_file = "train.csv"

# Constants
EPOCHS = 5
BATCH_SIZE = 128
INPUT_DIM = 4
OUTPUT_DIM = 50
RNN_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.0
MAXLEN = 100


# Loading the Dataset
def letter_to_index(letter):
    _alphabet = 'ATGC'
    letter = letter.upper()
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)


def load_data(test_split=0.2, maxlen=MAXLEN):
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


def create_lstm(input_length, rnn_hidden_dim=RNN_HIDDEN_DIM, output_dim=OUTPUT_DIM, input_dim=INPUT_DIM,
                dropout=DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim=INPUT_DIM, output_dim=output_dim, input_length=input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


# train
j = 0
x = 0
y = 0
while (j < 5):
    X_train, y_train, X_test, y_test = load_data()
    model = create_lstm(len(X_train[0]))
    j = j + 1
    print("Starting the training")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                        verbose=1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    x = x + test_acc
    y = y + test_loss
    print("Test accuracy: " + str(test_acc))
    print("Test loss: " + str(test_loss))

print(x / 5)
print(y / 5)
