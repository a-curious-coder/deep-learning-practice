import numpy as np
import plotly
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import layers

def vectorize_sequences(sequences, dimension = 10000):
    """Encodes integer representations of reviews into a binary matrix

    Args:
        sequences (list): Integer representation of review
        dimension (int, optional): . Defaults to 10000.

    Returns:
        [type]: [description]
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def decode_review(review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
    print(decoded_review)


def print_dataset_related_info(train_data, train_labels):
    # Prints a review with each common word represented by a number between 1-10000
    # print(train_data[0])
    # Prints binary value indicating if the review is positive (1) or negative (0)
    # print(train_labels[0])
    # decode_review(train_data[0])
    pass


def plot_accuracy(loss_values, val_loss_values):
    epochs = range(1, 21)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def load_data():
    # Load in training/test datasets
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
    
    # Vectorise training and test data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Vectorise training and test labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    return x_train, y_train, x_test, y_test


def trial_1():
    """Breaks training data to train/test datasets
    Trains a network model on the training dataset and validates using test dataset
    plot accuracy of network - 
    """
    x_train, y_train, x_test, y_test = load_data()
    # Prepare network
    network = models.Sequential()
    network.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dense(1, activation = 'sigmoid'))

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    
    # Compile the neural network / model
    network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

    history = network.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    
    history_dict = history.history
    print(history_dict.keys())
    plot_accuracy(history_dict['loss'], history_dict['val_loss'])

def trial_2():
    """Creates basic neural network model
    Trains model over less epochs (iterations) with a larger batch size
    Evaluates trained model using test set - 88% accuracy
    """
    x_train, y_train, x_test, y_test = load_data()
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(32, activation = 'tanh', input_shape = (10000,)))
    # Hidden layer
    model.add(layers.Dense(64, activation = 'tanh'))
    # Output layer
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['accuracy'])
                
    model.fit(x_train, y_train, epochs=4, batch_size = 512)
    results = model.evaluate(x_test, y_test)
    


def main():
    # trial_1()
    trial_2()


if __name__ == "__main__":
    main()