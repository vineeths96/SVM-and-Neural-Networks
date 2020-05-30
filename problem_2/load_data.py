import numpy as np
from sklearn.model_selection import train_test_split
from problem_2.parameters import *


# Function to load data from file
def load_data(train_file):
    # Matrices to store data
    X = np.zeros([TRAIN_SAMPLES, DIMENSIONS])
    Y = np.zeros(TRAIN_SAMPLES)

    # Read train data file
    file = open('./datasets/' + train_file)
    idx = 0
    for line in file:
        line_list = line.rstrip().split(',')

        # Remove None entries from list
        line_list = list(filter(None, line_list))

        # Skipping blank lines
        if len(line_list) == 0:
            continue

        X[idx] = np.array(line_list[:-1], dtype=float)
        Y[idx] = int(float(line_list[-1]))
        idx += 1

    file.close()

    # Train-Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SPLIT, random_state=0)

    return X_train, Y_train, X_test, Y_test
