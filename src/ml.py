from sklearn.linear_model import LogisticRegression


def train_ridge(X_train, y_train):

    model = LogisticRegression(
        penalty="l2",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def train_lasso(X_train, y_train):

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def train_elasticnet(X_train, y_train):

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model