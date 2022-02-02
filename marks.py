import plot_graphs as pg
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from keras import models
from keras import layers
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
import numpy as np
import math


def label_to_number(data, column):
    count = 0
    for label in data[column].unique():
        data[column] = data[column].replace(label, count)
        count += 1
    return data[column]


def normalize_tabular_data(_data):
    """Normalizes the values of each column (Excluding columns unrelated to the  scan itself)

    Args:
        _data (pd.DataFrame):  Data

    Returns:
        pd.DataFrame: Normalized  data
    """
    avoid = ["Study", "SID", "total CNR", "Gender", "Research Group", "Age"]
    # apply normalization techniques
    for column in _data.columns:
        if column not in avoid:
            _data[column] = _data[column] / _data[column].abs().max()
    return _data


def decision_tree(x_train, x_test, y_train, y_test):
    print("Decision Tree Model")
    model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    model.fit(x_train, y_train)
    target_pred = model.predict(x_test)
    print(f"Accuracy score: {accuracy_score(y_test, target_pred, normalize = True)*100:.2f}%")


def random_forest_pca(x_train, x_test, y_train, y_test):
    print("Random Forest Model")
    model = RandomForestClassifier(criterion='entropy', random_state=0)
    model.fit(x_train, y_train)
    target_pred = model.predict(x_test)
    print(f"Accuracy score: {accuracy_score(y_test, target_pred, normalize = True)*100:.2f}%")
    return accuracy_score(y_test, target_pred, normalize = True)*100


def k_fold_cross_validation_log(train_data, train_targets):
    print("K-Fold Cross Validation")
    k = 10
    num_val_samples = len(train_data) // k
    # Mean Absolute Error per fold during cross validation
    accuracies = []
    for i in range(k):
        print("Processing fold #", i)
        
        x_test = train_data[i * num_val_samples : (i + 1) * num_val_samples]
        y_test = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

        partial_x_train = np.concatenate(
            [
                train_data[: i * num_val_samples],
                train_data[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )

        partial_y_train = np.concatenate(
            [
                train_targets[: i * num_val_samples],
                train_targets[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )

        accuracies.append(random_forest_pca(partial_x_train, x_test, partial_y_train, y_test))
    print(*accuracies, sep="\n")


def keras_network(x_train, x_test, y_train, y_test):
     # Initialise Network
    model = models.Sequential()
    # Input layer
    model.add(layers.Dense(6, activation = 'relu', input_shape=(6,)))
    # Hidden layer
    model.add(layers.Dense(12, activation = 'relu'))
    # Output layer
    model.add(layers.Dense(18, activation = 'softmax'))

    model.compile(optimizer = 'adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

    # Size of validation set
    size = math.floor(x_train.shape[0]/10)

    x_val = x_train[:size]
    partial_x_train = x_train[size:]

    y_val = np.array(y_train[:size])
    partial_y_train = np.array(y_train[size:])

    results = model.evaluate(x_test, y_test)

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs = 2000,
                        batch_size = 512,
                        validation_data  = (x_val, y_val))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    pg.plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])
    model.save('marks_shitty_model.tf')
    predictions = model.evaluate(x_test, y_test)
    print(f"{predictions[1]*100:.2f}%")


def main():
    ohe = True
    data = read_data('chess.csv')
    # Convert strings to numbers
    encoded_letters = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8
    }
    
    dunno = ['king_x', 'rook_x', 'king2_x']
    for piece in dunno:
        for letter in encoded_letters:
            data[piece] = data[piece].replace(letter, encoded_letters[letter])
    # X contains everything except label
    x_data = data.iloc[:, :-1]
    x_data = normalize_tabular_data(x_data)

    # ONE HOT ENCODING
    if ohe:
        y_data = OHE(data, 'sucess')
        print(y_data[:5])
    else:
        y_data = label_to_number(data, 'sucess')

    k_fold_cross_validation_log(x_data, y_data)

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    decision_tree(x_train, x_test, y_train, y_test)
    random_forest_pca(x_train, x_test, y_train, y_test)
    # keras_network(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()