import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        y="category",
        order=df["category"].value_counts().index
    )

    plt.title("Furniture Category Distribution")
    plt.show()


def plot_price_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["price"],
        bins=40
    )

    plt.title("Price Distribution")

    plt.show()


def plot_price_by_category(df):

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="price",
        y="category"
    )

    plt.title("Price by Category")

    plt.show()


def plot_top_designers(df):

    plt.figure(figsize=(10, 5))

    df["designer"] \
        .value_counts() \
        .head(10) \
        .plot(kind="bar")

    plt.title("Top Designers")

    plt.show()


def plot_sellable_online(df):

    plt.figure(figsize=(6, 6))

    df["sellable_online"] \
        .value_counts() \
        .plot(
            kind="pie",
            autopct="%1.1f%%"
        )

    plt.title("Sellable Online")

    plt.ylabel("")

    plt.show()