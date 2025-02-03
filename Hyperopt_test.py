import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Flatten images for Dense layers
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Define the search space for hyperparameters
search_space = {
    'num_neurons': hp.choice('num_neurons', [32, 64, 128, 256]),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),  # log scale (exp(-5) to exp(-2))
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': hp.choice('epochs', [5, 10, 20])
}

# Define objective function to minimize
def objective(params):
    model = Sequential([
        Dense(params['num_neurons'], activation='relu', input_shape=(28*28,)),
        Dropout(params['dropout']),
        Dense(10, activation='softmax')  # Output layer (10 classes)
    ])

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
                        validation_data=(x_test, y_test), 
                        epochs=params['epochs'], 
                        batch_size=params['batch_size'], 
                        verbose=0)

    val_acc = max(history.history['val_accuracy'])  # Get best validation accuracy
    return -val_acc  # We want to maximize accuracy, so we minimize negative accuracy

# Run Hyperopt optimization
trials = Trials()
best_hyperparams = fmin(fn=objective, 
                        space=search_space, 
                        algo=tpe.suggest, 
                        max_evals=20,  # Number of trials
                        trials=trials)

print("\nBest Hyperparameters Found:", best_hyperparams)
