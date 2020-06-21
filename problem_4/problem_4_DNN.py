import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_4.load_data import load_data_DNN
from problem_4.parameters import *


# DNN Classifier
def problem_4_DNN():
    try:
        os.remove('./results/problem_4_DNN.txt')
    except OSError:
        pass

    # Load data
    X_train, Y_train, X_test, Y_test = load_data_DNN('Neuro_dataset')

    INPUT_SHAPE = X_train.shape[1:]

    # Define model
    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(NUM_DENSE_1, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='softmax'))

    # Set the optimizer values
    optimizer = RMSprop(lr=LEARNING_RATE,
                        rho=RHO,
                        epsilon=EPSILON,
                        decay=DECAY)

    # ReduceLROnPlateau callback to reduce learning rate when the validation accuracy plateaus
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                                patience=PATIENCE,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # Early stopping callback to stop training if we are not making any positive progress
    early_stopping = EarlyStopping(monitor='loss',
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
              validation_data=(X_test, Y_test),
              callbacks=callbacks)

    # Evaluate the model
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)

    print(f"The DNN classification accuracy  is: {accuracy * 100:0.2f}%")
    with open('./results/problem_4_DNN.txt', "a") as file:
        file.write(f"The DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
