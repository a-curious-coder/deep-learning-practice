from tensorflow.keras.datasets import imdb



def main():
    # Load in training/test datasets
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
    print(train_data[0])

if __name__ == "__main__":
    main()