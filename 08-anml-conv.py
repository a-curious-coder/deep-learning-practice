import os, shutil
from os.path import exists
from keras import layers
from keras import models


def prepare_directory():
    base_dir = "cat_dog_data"
    if not exists(base_dir):
        os.mkdir(base_dir)

    original_dataset_dir = "data/train/"

    train_dir = os.path.join(base_dir, "train")
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, "validation")
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, "test")
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, "cats")
    os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, "dogs")
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, "cats")
    os.mkdir(validation_cats_dir)

    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, "cats")
    os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, "dogs")
    os.mkdir(test_dogs_dir)

    # File names for cats
    fnames = ["cat.{}.jpg".format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["cat.{}.jpg".format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["cat.{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # File names for dogs
    fnames = ["dog.{}.jpg".format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["dog.{}.jpg".format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["dog.{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)


def main():
    prepare_directory()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="relu"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )
    return


if __name__ == "__main__":
    main()
