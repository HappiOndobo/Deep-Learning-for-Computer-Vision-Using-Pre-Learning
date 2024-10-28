#Deep Learning for Computer Vision: Using Pre-Learning
#Using a pretrained model for image classification

#import neccesary labraries
import numpy as np
import keras
from keras import layers, preprocessing
import matplotlib.pyplot as plt
import os

#Using the keras pretrain model VGG16
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)
conv_base.summary()  #help us get model structure

new_base_dir = "C:/Users/lordf/PycharmProjects/pythonProject/Pre_Learning"

#train dataset to train the model
train_dataset = preprocessing.image_dataset_from_directory(
    os.path.join(new_base_dir, "train"),
    image_size=(256, 256),
    batch_size=32,
    labels="inferred",
    label_mode="binary"
)
#validation dataset to validate model work
validation_dataset = preprocessing.image_dataset_from_directory(
    os.path.join(new_base_dir, "validation"),
    image_size=(256, 256),
    batch_size=32,
    labels="inferred",
    label_mode="binary"
)
#test dataset to test the model work
test_dataset = preprocessing.image_dataset_from_directory(
    os.path.join(new_base_dir, "test"),
    image_size=(256, 256),
    batch_size=32,
    labels=None,
    shuffle=False
)


# Function use to extract features and labels
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


#storing the extracted value accordingly.
train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)

#trainning the model
model = keras.Sequential([
    keras.Input(shape=(8, 8, 512)),
    layers.Flatten(),
    layers.Dense(256),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
history = model.fit(
    train_features, train_labels,
    epochs=24,
    validation_data=(val_features, val_labels))
model.save("model 1.keras")

#plotting accuracy and loss
epochs = range(1, 24 + 1)

plt.figure(figsize=(12, 4))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Точность обучения')
plt.plot(epochs, history.history['val_accuracy'], label='Точность валидации')
plt.title('Точность обучения и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Потери обучения')
plt.plot(epochs, history.history['val_loss'], label='Потери валидации')
plt.title('Потери обучения и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.show()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Точность обучения")
plt.plot(epochs, val_acc, "b", label="Точность валидации")
plt.title("Точность обучения и валидации")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Потери обучения")
plt.plot(epochs, val_loss, "b", label="Потери валидации")
plt.title("Потери обучения и валидации")
plt.legend()
plt.show()
