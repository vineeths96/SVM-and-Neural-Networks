import glob
import scipy.io
import numpy as np
from scipy.stats import zscore
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from problem_4.parameters import *


# Function to load data from file
def load_data_SVC(data_dir):
    # Matrices to store data
    TOTAL_PEOPLE = NUM_PATIENTS + NUM_NORMALS
    X = np.zeros([TOTAL_PEOPLE, TIME_TICKS, AAL_BRAIN_REGIONS])
    Y = np.zeros(TOTAL_PEOPLE)

    class_0 = glob.glob('./datasets/' + data_dir + '/Normal_Subjects' + '/*.mat')
    class_1 = glob.glob('./datasets/' + data_dir + '/Alzheimer\'s_Subjects' + '/*.mat')

    files = class_0 + class_1

    for ind, file in enumerate(files):
        X[ind] = scipy.io.loadmat(file)['tc_rest_aal']

        for col in range(AAL_BRAIN_REGIONS):
            if np.std(X[ind, :, col]) != 0:
                X[ind, :, col] = (X[ind, :, col] - np.min(X[ind, :, col])) / (
                        np.max(X[ind, :, col]) - np.min(X[ind, :, col]))

        if "Normal" in file:
            Y[ind] = 0
        else:
            Y[ind] = 1

    X = np.reshape(X, [X.shape[0], -1])

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test


# Function to load data from file
def load_data_DNN(data_dir):
    # Matrices to store data
    TOTAL_PEOPLE = NUM_PATIENTS + NUM_NORMALS
    X = np.zeros([TOTAL_PEOPLE, TIME_TICKS, AAL_BRAIN_REGIONS])
    Y = np.zeros(TOTAL_PEOPLE)

    class_0 = glob.glob('./datasets/' + data_dir + '/Normal_Subjects' + '/*.mat')
    class_1 = glob.glob('./datasets/' + data_dir + '/Alzheimer\'s_Subjects' + '/*.mat')

    files = class_0 + class_1

    for ind, file in enumerate(files):
        X[ind] = scipy.io.loadmat(file)['tc_rest_aal']

        for col in range(AAL_BRAIN_REGIONS):
            if np.std(X[ind, :, col]) != 0:
                X[ind, :, col] = zscore(X[ind, :, col])

        if "Normal" in file:
            Y[ind] = 0
        else:
            Y[ind] = 1

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test


# Function to load data from file
def load_data_CNN(data_dir):
    # Matrices to store data
    TOTAL_PEOPLE = NUM_PATIENTS + NUM_NORMALS
    X = np.zeros([TOTAL_PEOPLE, TIME_TICKS, AAL_BRAIN_REGIONS])
    Y = np.zeros(TOTAL_PEOPLE)

    class_0 = glob.glob('./datasets/' + data_dir + '/Normal_Subjects' + '/*.mat')
    class_1 = glob.glob('./datasets/' + data_dir + '/Alzheimer\'s_Subjects' + '/*.mat')

    files = class_0 + class_1

    for ind, file in enumerate(files):
        X[ind] = scipy.io.loadmat(file)['tc_rest_aal']

        for col in range(AAL_BRAIN_REGIONS):
            if np.std(X[ind, :, col]) != 0:
                X[ind, :, col] = zscore(X[ind, :, col])

        if "Normal" in file:
            Y[ind] = 0
        else:
            Y[ind] = 1

    X = np.expand_dims(X, axis=-1)

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test
