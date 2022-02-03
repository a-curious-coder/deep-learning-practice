import plot_graphs as pg
import os, shutil
from os.path import exists
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def prepare_directory():
    """Create train, validation and test sets"""
    base_dir = "cat_dog_data"
    if not exists(base_dir):
        os.mkdir(base_dir)

    original_dataset_dir = "data/train/"

    train_dir = os.path.join(base_dir, "train")
    if not exists(train_dir):
        os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, "validation")
    if not exists(validation_dir):
        os.mkdir(validation_dir)

    test_dir = os.path.join(base_dir, "test")
    if not exists(test_dir):
        os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, "cats")
    if not exists(train_cats_dir):
        os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, "dogs")
    if not exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, "cats")
    if not exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)

    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    if not exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, "cats")
    if not exists(test_cats_dir):
        os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, "dogs")
    if not exists(test_dogs_dir):
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

    return train_cats_dir, train_dir, validation_dir


def create_model():
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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )

    return model


def main():
    train_cats_dir, train_dir, validation_dir = prepare_directory()
    if not exists("models/cats_and_dogs_small_1.h5"):
        model = create_model()
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
        )

        validation_generator = test_datagen.flow_from_directory(
            validation_dir, target_size=(150, 150), batch_size=20, class_mode="binary"
        )

        for data_batch, labels_batch in train_generator:
            print("data batch shape: ", data_batch.shape)
            print("labels batch shape: ", labels_batch.shape)
            break

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50,
        )

        model.save("models/cats_and_dogs_small_1.h5")

        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        pg.plot_accuracy(acc, val_acc)
        pg.plot_loss(loss, val_loss)

    if not exists("models/cats_and_dogs_small_2.h5"):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        fnames = [
            os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)
        ]

        img_path = fnames[3]
        img = image.load_img(img_path, target_size=(150, 150))

        x = image.img_to_array(img)

        x = x.reshape((1,) + x.shape)

        i = 0

        import matplotlib.pyplot as plt

        for batch in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            i += 1
            if i % 4 == 0:
                break
        plt.show()

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        # Validation should not be augmented
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
        )

        validation_generator = test_datagen.flow_from_directory(
            validation_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
        )
        model = create_model()
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=50,
        )

        model.save("cats_and_dogs_small_2.h5")

        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        pg.plot_accuracy(acc, val_acc)
        pg.plot_loss(loss, val_loss)
        
if __name__ == "__main__":
    main()
