from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import plotly.graph_objs as go
import os.path


def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


def k_fold_cross_validation(train_data, train_targets):
    k = 4
    num_val_samples = len(train_data) // k
    all_scores = []

    for i in range(k):
        print("processing fold #", i)
        val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [
                train_data[: i * num_val_samples],
                train_data[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )
        partial_train_targets = np.concatenate(
            [
                train_targets[: i * num_val_samples],
                train_targets[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )

        model = build_model(train_data)
        model.fit(
            partial_train_data,
            partial_train_targets,
            epochs=num_epochs,
            batch_size=1,
            verbose=1,
        )
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
        all_scores.append(val_mae)
    return all_scores


def k_fold_cross_validation_log(num_epochs, train_data, train_targets):
    k = 4
    num_val_samples = len(train_data) // k
    # Mean Absolute Error per fold during cross validation
    all_mae_histories = []
    for i in range(k):
        print("Processing fold #", i)
        val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [
                train_data[: i * num_val_samples],
                train_data[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )

        partial_train_targets = np.concatenate(
            [
                train_targets[: i * num_val_samples],
                train_targets[(i + 1) * num_val_samples :],
            ],
            axis=0,
        )
        if os.path.exists("boston_model"):
            print("Loading model")
            model = models.load_model("boston_model")
        else:
            print("Building model")
            model = build_model(train_data)
            model.save("boston_model")
        
        history = model.fit(
            partial_train_data,
            partial_train_targets,
            validation_data=(val_data, val_targets),
            epochs=num_epochs,
            batch_size=1,
            verbose=1,
        )

        # print(history.history.keys())
        mae_history = history.history["mae"]
        all_mae_histories.append(mae_history)
    return all_mae_histories


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_k_fold(average_mae_history):
    # Smooth out curve, remove first 10 datapoints
    smooth_mae_history = smooth_curve(average_mae_history[10:])

    fig = go.Figure(
        data=go.Scatter(
            x=list(range(1, len(smooth_mae_history) + 1)),
            y=smooth_mae_history,
            mode="lines",
        ),
        layout=go.Layout(title=dict(text="Validation MAE by epoch", x=0.5)),
    )
    fig.write_html("05_mae.html")
    fig.show()


def main():
    num_epochs = 80
    # Load boston housing data
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    # Normalise the data
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    # K fold cross validation
    # all_scores = k_fold_cross_validation(train_data, train_targets)

    mae_histories = k_fold_cross_validation_log(
        num_epochs, train_data, train_targets
    )
    average_mae_history = [
        np.mean([x[i] for x in mae_histories]) for i in range(num_epochs)
    ]
    plot_k_fold(average_mae_history)


if __name__ == "__main__":
    main()
