# Libraries.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


# Directories of data.
d_training = "C:\\Users\\Lenovo\\Desktop\\TEC\\PYTHON\\Training"
d_test = "C:\\Users\\Lenovo\\Desktop\\TEC\\PYTHON\\Test"

# Image loading into train, validation, and test.
train = tf.keras.preprocessing.image_dataset_from_directory(d_training,
                                                            image_size=(224, 224),
                                                            batch_size=32,
                                                            labels="inferred",
                                                            shuffle=True,
                                                            seed=100,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            label_mode="int")

validation = tf.keras.preprocessing.image_dataset_from_directory(d_training,
                                                                 image_size=(224, 224),
                                                                 batch_size=32,
                                                                 labels="inferred",
                                                                 shuffle=True,
                                                                 seed=100,
                                                                 validation_split=0.2,
                                                                 subset="validation",
                                                                 label_mode="int")

test = tf.keras.preprocessing.image_dataset_from_directory(d_test,
                                                           image_size=(224, 224),
                                                           batch_size=32,
                                                           labels="inferred",
                                                           shuffle=True,
                                                           label_mode="int")

# Name of the classes.
class_names = train.class_names
print(class_names)

# Data Visualization.
plt.figure(figsize=(10, 10))
for image, label in train.take(1):
    for i in range(6):
        ax = plt.subplot(3, 2, i + 1)
        plt.imshow(image[i].numpy().astype("uint8"))
        plt.title(class_names[label[i]])
        plt.axis("off")
plt.show()

# Preprocessing Sequence
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.Normalization(),
])


# Augmented Data Visualization.
plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
  for i in range(6):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 2, i + 1)
    plt.imshow(augmented_images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
