import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Dense, Dropout
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from problem_4.load_data import load_data_CNN
from problem_4.parameters import *


# CNN Classifier
def problem_4_CNN():
    try:
        os.remove('./results/problem_4_CNN.txt')
    except OSError:
        pass

    # Load data
    X_train, Y_train, X_test, Y_test = load_data_CNN('Neuro_dataset')

    INPUT_SHAPE = X_train.shape[1:]

    # Define model
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(BatchNormalization(axis=1))

    model.add(Conv2D(32, kernel_size=3, kernel_initializer=glorot_normal(), padding='same'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(1, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(64, kernel_size=3, kernel_initializer=glorot_normal(), padding='same'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(1, 1, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
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

    print(f"The CNN DNN classification accuracy  is: {accuracy * 100:0.2f}%")
    with open('./results/problem_4_CNN.txt', "a") as file:
        file.write(f"The CNN DNN classification accuracy is: {accuracy * 100:0.2f}%\n")
