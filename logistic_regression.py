import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train).reshape(-1)
    x_test = np.asarray(x_test, dtype=float)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=0
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_pred.astype(int)
