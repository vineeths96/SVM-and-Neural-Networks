import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from problem_1.load_data import load_data
from problem_1.parameters import *


# DNN Classifier
def problem_1_DNN():
    try:
        os.remove('./results/problem_1_DNN.txt')
    except OSError:
        pass

    noise_level = [0, 20, 40]
    for noise in noise_level:
        # Load data
        X_train, Y_train, X_test, Y_test = load_data(f'/2class-Synthetic/data_noise_{noise}.txt')

        # Define model
        model = Sequential()
        model.add(Input(shape=DIMENSIONS))
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
                            validation_split=0.1)

        # Evaluate the model
        loss, accuracy = model.evaluate(x=X_test, y=Y_test)

        print(f"The DNN classification accuracy {noise}% label noise is: {accuracy * 100:0.2f}%")
        with open('./results/problem_1_DNN.txt', "a") as file:
            file.write(f"The DNN classification accuracy {noise}% label noise is: {accuracy * 100:0.2f}%\n")
