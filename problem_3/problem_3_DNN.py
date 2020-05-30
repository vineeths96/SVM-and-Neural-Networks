import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_3.load_data import load_data_DNN
from problem_3.parameters import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# DNN Classifier
def problem_3_DNN():
    os.remove('./results/problem_3_DNN.txt')

    # Load data
    X_train, Y_train, X_test, Y_test = load_data_DNN('MNIST')

    INPUT_SHAPE = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(NUM_DENSE_1, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_DENSE_2, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Set the optimizer values
    optimizer = RMSprop(lr=LEARNING_RATE,
                        rho=RHO,
                        epsilon=EPSILON,
                        decay=DECAY)

    # ReduceLROnPlateau callback to reduce learning rate when the validation accuracy plateaus
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=PATIENCE,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # Early stopping callback to stop training if we are not making any positive progress
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=PATIENCE)

    callbacks = [learning_rate_reduction, early_stopping]

    # Train the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x=X_train,
              y=Y_train,
              epochs=TRAINING_EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=VALIDATION_SPLIT,
              callbacks=callbacks)

    # Evaluate the model
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)

    print(f"The CNN DNN classification accuracy  is: {accuracy * 100:0.2f}%")
    with open('./results/problem_3_DNN.txt', "a") as file:
        file.write(f"The CNN DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
