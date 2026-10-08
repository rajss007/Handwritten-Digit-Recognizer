import numpy as np
import keras
import tensorflow as tf
from keras import layers
from tensorflow.keras.datasets import mnist    #Importing training data from mnist dataset
(x_train, y_train), _ = mnist.load_data()

#Filtering out corrupted images and completely blank images
valid_train = np.isfinite(x_train).all(axis=(1, 2))
not_blank_train = (x_train.sum(axis=(1, 2)) > 0)
train_mask = valid_train & not_blank_train

#Using filtered dataset
x_train_clean = x_train[train_mask]
y_train_clean = y_train[train_mask]

print("Training set - Original:", len(x_train),", After Cleaning: ", len(x_train_clean))

data_augmentation_layers = [
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    #Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    previous_block_activation = x  #Set aside residual

    for size in [64, 128, 256]:
        #First convolutional layer
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        #Second convolutional layer
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        #Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  #Adding back residual
        previous_block_activation = x  #Setting aside next residual

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

#Instantiating the model with MNIST dimensions (28x28 pixels, 1 color channel, 10 digit classes)
model = make_model(input_shape=(28, 28, 1), num_classes=10)

#Compiling the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

#Preparing training dataset pipeline
x_train_clean_expanded = np.expand_dims(x_train_clean, axis=-1)
train_ds_clean = tf.data.Dataset.from_tensor_slices((x_train_clean_expanded, y_train_clean))

#Shuffling, Batching, Mapping, and Prefetching
batch_size = 64

train_ds_clean = (
    train_ds_clean
    .shuffle(buffer_size=10000)
    .batch(batch_size)
    .map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

#Training the model
epochs = 3
print("Training the model... Please wait.")
history = model.fit(train_ds_clean, epochs=epochs)
print("Training complete!")

model.save("mnist_model.keras")
print("Model saved successfully!")