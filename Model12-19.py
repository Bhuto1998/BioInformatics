from sklearn.svm import SVC
# Starting with the imports
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score
matplotlib.use('pdf')

from keras.preprocessing.sequence import pad_sequences


# Input file name
input_file = "train.csv"

# Constants
EPOCHS = 5
BATCH_SIZE = 128
INPUT_DIM = 4
OUTPUT_DIM = 50
RNN_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.8
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





X_train, y_train, X_test, y_test = load_data()
# Model 12
rbf_svc = SVC(kernel='rbf', gamma=1)
print("Starting the training for model 12")
rbf_svc.fit(X_train, y_train)
y_pred = rbf_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Model 13
print("Starting the training for Model 13")
linear_svc = SVC(C=10, kernel='linear')
linear_svc.fit(X_train,y_train)
y_pred = linear_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
# Model 14
print("Starting the training for Model 14")
linear_svc = SVC(C=100, kernel='linear')
linear_svc.fit(X_train,y_train)
y_pred = linear_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
# Model 15
print("Starting the training for Model 15")
linear_svc = SVC(C=1, kernel='linear')
linear_svc.fit(X_train,y_train)
y_pred = linear_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
# Model 16
print("Starting the training for Model 16")
poly_svc = SVC(kernel='poly', degree=3, coef0=7, C=10)
poly_svc.fit(X_train,y_train)
y_pred = poly_svc.predict(y_test , y_pred)
print(accuracy_score(y_test, y_pred))
# Model 17
print("Starting the training for Model 17")
poly_svc = SVC(kernel='poly', degree=5, coef0=7, C=10)
poly_svc.fit(X_train,y_train)
y_pred = poly_svc.predict(y_test , y_pred)
print(accuracy_score(y_test, y_pred))
# Model 18
print("Starting the training for model 18")
lr = sm.Logit(y_train,sm.add_constant(X_test))
lr.fit(disp = False)
y_pred = lr.pred(X_test)
print(accuracy_score(y_test, y_pred))

# Model 19
print("Starting the training for Model 19")
linear_svc = SVC(C=.1, kernel='linear')
linear_svc.fit(X_train,y_train)
y_pred = linear_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))


