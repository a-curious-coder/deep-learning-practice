# Import file in local working dir for plotting graphs
import plot_graphs as pg
import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical

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
    pg.verify_import()
    epochs = 20
    # Load Reuters Dataset
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
    # one_hot_train_labels = to_one_hot(train_labels)
    # one_hot_test_labels = to_one_hot(test_labels)

    x_train = vectorize_sequences(train_data)
    print(x_train)
    x_test = vectorize_sequences(test_data)

    y_train = np.array(train_labels)
    y_test= np.array(test_labels)

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    # Initialise Network
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
    # Hidden layer
    model.add(layers.Dense(32, activation = 'relu'))
    # Output layer (46 outputs)
    model.add(layers.Dense(46, activation = 'softmax'))

    model.compile(optimizer = 'rmsprop',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    # y_val = one_hot_train_labels[:1000]
    # partial_y_train = one_hot_train_labels[1000:]
    y_val = np.array(y_train[:1000])
    partial_y_train = np.array(y_train[1000:])
    
    results = model.evaluate(x_test, y_test)

    # Adjusted to 9 epochs
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs = epochs,
                        batch_size = 128,
                        validation_data  = (x_val, y_val))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    pg.plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])
    # pg.plot_loss(history.history['loss'], history.history['val_loss'])
    # pg.plot_all_graphs(loss, val_loss, acc, val_acc)
    predictions = model.evaluate(x_test, y_test)



if __name__ == "__main__":
    main()