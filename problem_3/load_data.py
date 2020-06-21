import scipy.io
import numpy as np
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
    X_train_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_data.mat')['training_data']
    Y_train_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_label.mat')['training_label']

    X_train_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_training_data.mat')['train_data']
    Y_train_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_training_label.mat')['train_label']

    X_test_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_data.mat')['test_data']
    Y_test_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_label.mat')['test_label']

    X_test_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_test_data.mat')['test_data']
    Y_test_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_test_label.mat')['test_label']

    X_train = np.vstack([X_train_norm, np.reshape(X_train_rot, [X_train_rot.shape[0], -1])])
    Y_train = np.append(Y_train_norm, Y_train_rot)
    X_test = np.vstack([X_test_norm, np.reshape(X_test_rot, [X_test_rot.shape[0], -1])])
    Y_test = np.append(Y_test_norm, Y_test_rot)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test


# Function to load data from file
def load_data_CNN(data_dir):
    # Load mat files
    X_train_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_data.mat')['training_data']
    Y_train_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_training_label.mat')['training_label']

    X_train_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_training_data.mat')['train_data']
    Y_train_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_training_label.mat')['train_label']

    X_test_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_data.mat')['test_data']
    Y_test_norm = scipy.io.loadmat('./datasets/' + data_dir + '/mnist_test_label.mat')['test_label']

    X_test_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_test_data.mat')['test_data']
    Y_test_rot = scipy.io.loadmat('./datasets/' + data_dir + '/mnist-rot_test_label.mat')['test_label']

    X_train = np.vstack([X_train_norm, np.reshape(X_train_rot, [X_train_rot.shape[0], -1])])
    Y_train = np.append(Y_train_norm, Y_train_rot)
    X_test = np.vstack([X_test_norm, np.reshape(X_test_rot, [X_test_rot.shape[0], -1])])
    Y_test = np.append(Y_test_norm, Y_test_rot)

    # Converting to 4D tensors
    X_train = X_train.reshape(X_train.shape[0], PIXELS, PIXELS, 1)
    X_test = X_test.reshape(X_test.shape[0], PIXELS, PIXELS, 1)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    return X_train, Y_train, X_test, Y_test
