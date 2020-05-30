import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_1.load_data import load_data
from problem_1.parameters import *


# DNN Classifier
def problem_1_DNN():
    # Load data
    X_train, Y_train, X_test, Y_test = load_data('/2class-Synthetic/data_noise_0.txt')

    # Define model
    model = Sequential()
    model.add(Input(shape=(DIMENSIONS)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=NUM_CLASS, activation='softmax'))

    # Define optimizer
    optimizer = RMSprop(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x=X_train,
                        y=Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)

    print(f"The DNN classification accuracy is: {accuracy * 100:0.2f}%")
    with open('./results/problem_1_DNN.txt', "w") as file:
        file.write(f"The DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
