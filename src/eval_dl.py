import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_history(history):

    plt.figure(figsize=(10,5))

    plt.plot(
        history.history["accuracy"],
        label="train"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="validation"
    )

    plt.legend()

    plt.title("Accuracy")

    plt.show()
    


def plot_confusion_matrix(model, val_ds, class_names):

    y_true = []
    y_pred = []

    for images, labels in val_ds:

        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())

        y_pred.extend(
            np.argmax(predictions, axis=1)
        )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CNN Confusion Matrix")

    plt.show()