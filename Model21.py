# Starting with the imports
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score
matplotlib.use('pdf')

from keras.preprocessing.sequence import pad_sequences

EPOCHS = 50
BATCH_SIZE = 128
INPUT_DIM = 4
OUTPUT_DIM = 50
RNN_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.8
MAXLEN = 100
from sklearn.ensemble import IsolationForest
# Input file name
input_file = "train.csv"
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


X_train, y_train, X_test, y_test = load_data()
model = IsolationForest(contamination=0.35, behaviour='new')
# fit on majority class
X_train = X_train[y_train==0]
model.fit(X_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))