import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from os.path import exists


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # Feed 3D output tensor to classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    # 10 way classification output
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype("float32") / 255

    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype("float32") / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    if not exists("models/07-mnist-model.tf"):
        print("Creating Model")
        model = create_model()
        model.fit(x_train, y_train, epochs=5, batch_size=64)
        model.save("models/07-mnist-model.tf")
    print("Loading Model")
    model = models.load_model("models/07-mnist-model.tf")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(model.summary())
    pass


if __name__ == "__main__":
    main()
