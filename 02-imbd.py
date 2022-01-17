import numpy as np
from tensorflow.keras.datasets import imdb
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


def main():
    # Load in training/test datasets
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
    
    # Vectorise training and test data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Vectorise training and test labels
    y_train = vectorize_sequences(train_labels).astype('float32')
    y_test = vectorize_sequences(test_labels).astype('float32')

    # Prepare network
    network = models.Sequential()
    network.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
    network.add(layers.Dense(16, activation='relu'))
    network.add(1, activation = 'sigmoid')
    
if __name__ == "__main__":
    main()