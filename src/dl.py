import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

def load_image_datasets(image_dir):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )

    return train_ds, val_ds

def build_cnn(num_classes):

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(224,224,3)
        ),

        MaxPooling2D(),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(),

        Conv2D(
            128,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model


def train_model(
    model,
    train_ds,
    val_ds,
    epochs=10
):

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history

def save_model(
    model,
    path="models/furniture_cnn.keras"
):

    model.save(path)

    print(f"Model saved to {path}")