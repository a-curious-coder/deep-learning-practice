import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt
from os.path import exists
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers

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
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
    print(decoded_review)


def print_dataset_related_info(train_data, train_labels):
    # Prints a review with each common word represented by a number between 1-10000
    # print(train_data[0])
    # Prints binary value indicating if the review is positive (1) or negative (0)
    # print(train_labels[0])
    # decode_review(train_data[0])
    pass


def plot_loss(network_type, o_loss, s_loss):
    fig = go.Figure()
    original = go.Scatter(
        x = list(range(1, len(o_loss)+1)),
        y = o_loss,
        name = "Original Network Loss",
        mode = 'markers',
        marker = dict(size = 16,
                        symbol = 'x'),
        hovertemplate = "<br>".join(['Epoch %{x}',
                                    'Validation Loss: %{y:.3f}',
                                    '<extra></extra>'])
    )
    fig.add_trace(original)

    if network_type == "smaller":
        smaller = go.Scatter(
            x = list(range(1, len(s_loss)+1)),
            y = s_loss,
            name = "Smaller Network Loss",
            mode = 'markers',
            marker = dict(size = 16),
            hovertemplate = "<br>".join(['Epoch %{x}',
                                        'Validation Loss: %{y:.3f}',
                                        '<extra></extra>'])
        )
        fig.add_trace(smaller)
        fig.update_layout(
            title = dict(text = "Original vs. Smaller Network Validation Loss",
                            x = 0.5),
            xaxis = dict(tickmode = 'linear',
                            tick0 = 0,
                            dtick = 1,
                            title = "Epochs"),
            yaxis = dict(title = "Validation Loss")
        )
        fig.write_html('06_os_network.html')
    elif network_type == "bigger":
        bigger = go.Scatter(
            x = list(range(1, len(s_loss)+1)),
            y = s_loss,
            name = "Bigger Network Loss",
            mode = 'markers',
            marker = dict(size = 16),
            hovertemplate = "<br>".join(['Epoch %{x}',
                                        'Validation Loss: %{y:.3f}',
                                        '<extra></extra>'])
        )
        fig.add_trace(bigger)
        fig.update_layout(
            title = dict(text = "Original vs. Bigger Network Validation Loss",
                            x = 0.5),
            xaxis = dict(tickmode = 'linear',
                            tick0 = 0,
                            dtick = 1,
                            title = "Epochs"),
            yaxis = dict(title = "Validation Loss")
        )
        fig.write_html('06_ob_network.html')
    elif network_type == "regularization":
        regularization = go.Scatter(
            x = list(range(1, len(s_loss)+1)),
            y = s_loss,
            name = "Regularization Network Loss",
            mode = 'markers',
            marker = dict(size = 16),
            hovertemplate = "<br>".join(['Epoch %{x}',
                                        'Validation Loss: %{y:.3f}',
                                        '<extra></extra>'])
        )
        fig.add_trace(regularization)
        fig.update_layout(
            title = dict(text = "Original vs. Regularization Network Validation Loss",
                            x = 0.5),
            xaxis = dict(tickmode = 'linear',
                            tick0 = 0,
                            dtick = 1,
                            title = "Epochs"),
            yaxis = dict(title = "Validation Loss")
        )
        fig.write_html('06_reg_network.html')

        
    fig.show()


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


def train_network(x_train, y_train, x_test, y_test, settings):
    """Creates basic neural network model
    Trains model over less epochs (iterations) with a larger batch size
    Evaluates trained model using test set - 88% accuracy
    """
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(settings[0], activation = 'relu', input_shape = (10000,)))
    # Hidden layer
    model.add(layers.Dense(settings[1], activation = 'relu'))
    # Output layer
    model.add(layers.Dense(settings[2], activation = 'sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])
                
    history = model.fit(x_train, y_train, epochs=20, batch_size = 512, validation_data=(x_test, y_test))
    history_dict = history.history
    results = model.evaluate(x_test, y_test)
    return history_dict['val_loss']


def train_network_regularizer(x_train, y_train, x_test, y_test, settings):
    """Creates basic neural network model
    Trains model over less epochs (iterations) with a larger batch size
    Evaluates trained model using test set - 88% accuracy
    """
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(settings[0], kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000,)))
    # Dropout layer
    model.add(layers.Dropout(0.5))
    # Hidden layer
    model.add(layers.Dense(settings[1], kernel_regularizer = regularizers.l2(0.001), activation = 'relu'))
    # Dropout layer
    model.add(layers.Dropout(0.5))
    # Output layer
    model.add(layers.Dense(settings[2], activation = 'sigmoid'))

    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])
                
    history = model.fit(x_train, y_train, epochs=20, batch_size = 512, validation_data=(x_test, y_test))
    history_dict = history.history
    results = model.evaluate(x_test, y_test)
    tf.keras.utils.plot_model(model, show_shapes = True, rankdir='LR')
    return history_dict['val_loss']


def main():
    modes = ["smaller", "bigger", "regularization"]
    for mode in modes:
        loss_stats = []
        x_train, y_train, x_test, y_test = load_data()
        if not exists(f"06vd_{mode}.csv") and mode == "regularization":
            loss_stats.append(train_network(x_train, y_train, x_test, y_test, [16,16,1]))
            loss_stats.append(train_network_regularizer(x_train, y_train, x_test, y_test, [16,16,1]))
            df = pd.DataFrame({'original': loss_stats[0], mode : loss_stats[1]})
            df.to_csv(f"06vd_{mode}.csv", index = False)
        
        if not exists(f"06v_{mode}.csv") and mode != "regularization":
            if mode == "bigger":
                settings = [[16,16,1], [512,512,1]]
            elif mode == "smaller":
                settings = [[16,16,1], [4,4,1]]
            
            for setting in settings:
                loss_stats.append(train_network(x_train, y_train, x_test, y_test, setting))

            df = pd.DataFrame({'original': loss_stats[0], mode : loss_stats[1]})
            df.to_csv(f"06v_{mode}.csv", index = False)
        else:
            df = pd.read_csv(f"06v_{mode}.csv")
        print(df.head())
        plot_loss(mode, df['original'].tolist(), df[mode].tolist())


if __name__ == "__main__":
    main()