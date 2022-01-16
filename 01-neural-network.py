from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import plotly.express as px
import numpy as np

def plot_binary_image(image):
    fig = px.imshow(image, binary_string=True)
    fig.show()

def main():
    print("Starting")
    # Initialise training/test datasets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape)
    # Initialise the neural network
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    network.add(layers.Dense(10, activation = 'softmax'))
    # 
    network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Preparing datasets
    # Normalise pixel values for each image (from [0-255] to [0-1])
    train_images = train_images.reshape((60000, 28*28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28*28))
    test_images = test_images.astype('float32') / 255

    # Create labels for each dataset
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Train the network to associate labels with images
    network.fit(train_images, train_labels, epochs=5, batch_size = 128)

    # Test trained network
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    accuracy = "{:.2f}".format(test_acc*100)
    print(f"Test Accuracy: {accuracy}%")



if __name__ == "__main__":
    print("right")
    main()