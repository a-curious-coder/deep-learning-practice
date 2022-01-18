import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical

def plot_loss(loss_values, val_loss_values):
    epochs = range(1, 21)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(acc, val_acc):
    epochs = range(1, 21)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def to_one_hot(labels, dimension = 46):
    """Embeds each label with all zero vector with a 1 in the place of label index

    Returns:
        [type]: [description]
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


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


def main():
    # Load Reuters Dataset
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
    # one_hot_train_labels = to_one_hot(train_labels)
    # one_hot_test_labels = to_one_hot(test_labels)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
    # Hidden layer
    model.add(layers.Dense(64, activation = 'relu'))
    # Output layer (46 outputs)
    model.add(layers.Dense(46, activation = 'softmax'))

    model.compile(optimizer = 'rmsprop',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    # Adjusted to 9 epochs
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs = 9,
                        batch_size = 512,
                        validation_data  = (x_val, y_val))
    
    results = model.evaluate(x_test, one_hot_test_labels)
    print(history.history.keys())
    plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])
    plot_loss(history.history['loss'], history.history['val_loss'])

    predictions = model.predict(x_test)
    # print(np.argmax(predictions[0]))
    


if __name__ == "__main__":
    main()