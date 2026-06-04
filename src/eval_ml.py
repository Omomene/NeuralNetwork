import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                y_pred
            ),
            4
        )
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return y_pred


def plot_confusion_matrix(
    y_test,
    y_pred
):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix(
            y_test,
            y_pred
        ),
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")

    plt.show()