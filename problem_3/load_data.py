import scipy.io
from sklearn.utils import shuffle
from problem_3.parameters import *


# Function to load data from file
def load_data_SVC(data_dir):
    # Load mat files
    X_train = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_data.mat')['training_data']
    Y_train = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_label.mat')['training_label']

    X_test = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_data.mat')['test_data']
    Y_test = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_label.mat')['test_label']

    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test


# Function to load data from file
def load_data_DNN(data_dir):
    # Load mat files
    X_train = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_data.mat')['training_data']
    Y_train = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_label.mat')['training_label']

    X_test = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_data.mat')['test_data']
    Y_test = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_label.mat')['test_label']

    # Converting to 4D tensors
    X_train = X_train.reshape(X_train.shape[0], PIXELS, PIXELS, 1)
    X_test = X_test.reshape(X_test.shape[0], PIXELS, PIXELS, 1)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test
