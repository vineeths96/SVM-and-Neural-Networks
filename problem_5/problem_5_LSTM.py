import os
from keras.models import Sequential
from keras.layers import InputLayer, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_5.load_data import load_data_LSTM
from problem_5.parameters import *


# LSTM Classifier
def problem_5_LSTM():
    try:
        os.remove('./results/problem_5_LSTM.txt')
    except OSError:
        pass

    # Load data
    X_train, Y_train, X_test, Y_test = load_data_LSTM('EEG/eeg_data.csv')

    INPUT_SHAPE = X_train.shape[1:]

    # Define model
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Set the optimizer values
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )

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
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    model.fit(x=X_train,
              y=Y_train,
              epochs=TRAINING_EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_test, Y_test),
              callbacks=callbacks)

    # Evaluate the model
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)

    print(f"The LSTM DNN classification accuracy  is: {accuracy * 100:0.2f}%")
    with open('./results/problem_5_LSTM.txt', "a") as file:
        file.write(f"The LSTM DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
