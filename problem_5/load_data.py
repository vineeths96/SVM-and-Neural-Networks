import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from problem_5.parameters import *


# Function to load data from file
def load_data(train_file):
    # Load data
    df = pd.read_csv('./datasets/' + train_file)
    df = df.drop(axis=1, columns=df.columns[0])

    X = df.values[:, :-1]
    Y = df.values[:, -1]

    Y = Y - 1

    for ind in range(X.shape[0]):
        X[ind] = zscore(X[ind])

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test


# Function to load data from file
def load_data_LSTM(train_file):
    # Load data
    df = pd.read_csv('./datasets/' + train_file)
    df = df.drop(axis=1, columns=df.columns[0])

    X = df.values[:, :-1]
    Y = df.values[:, -1]

    Y = Y - 1

    for ind in range(X.shape[0]):
        X[ind] = zscore(X[ind])

    X = np.expand_dims(X, axis=-1)

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test
