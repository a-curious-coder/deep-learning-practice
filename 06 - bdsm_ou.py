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


def plot_loss(o_loss, s_loss, title):
    epochs = range(1, len(o_loss)+1)
    plt.plot(epochs, o_loss, 'bo', label='Original Network Loss')
    plt.plot(epochs, s_loss, 'ro', label='Smaller Network Loss')
    plt.title(title)
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


def original_network():
    """Creates basic neural network model
    Trains model over less epochs (iterations) with a larger batch size
    Evaluates trained model using test set - 88% accuracy
    """
    x_train, y_train, x_test, y_test = load_data()
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
    # Hidden layer
    model.add(layers.Dense(16, activation = 'relu'))
    # Output layer
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['acc'])
                
    history = model.fit(x_train, y_train, epochs=20, batch_size = 512)
    history_dict = history.history
    results = model.evaluate(x_test, y_test)
    print(history_dict['loss'])
    return history_dict['loss']


def smaller_network():
    """Creates basic neural network model
    Trains model over less epochs (iterations) with a larger batch size
    Evaluates trained model using test set - 88% accuracy
    """
    x_train, y_train, x_test, y_test = load_data()
    model = models.Sequential()
    # Hidden layer
    model.add(layers.Dense(4, activation = 'relu', input_shape = (10000,)))
    # Hidden layer
    model.add(layers.Dense(4, activation = 'relu'))
    # Output layer
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])
                
    history = model.fit(x_train, y_train, epochs=20, batch_size = 512)
    history_dict = history.history
    results = model.evaluate(x_test, y_test)
    print(history_dict['loss'])
    return history_dict['loss']


def main():
    o_loss = original_network()
    s_loss = smaller_network()
    print(o_loss)
    print(s_loss)
    plot_loss(o_loss, s_loss, "Original vs. Smaller Network")



if __name__ == "__main__":
    main()