import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float)

    n_samples, n_features = x_train.shape

    w = np.zeros(n_features, dtype=float)
    b = 0.0

    lr = 0.1
    num_iterations = 5000

    for _ in range(num_iterations):
        z = x_train @ w + b
        z = np.clip(z, -500, 500)
        p = 1.0 / (1.0 + np.exp(-z))

        error = p - y_train
        dw = (x_train.T @ error) / n_samples
        db = np.sum(error) / n_samples

        w -= lr * dw
        b -= lr * db

    z_test = x_test @ w + b
    z_test = np.clip(z_test, -500, 500)
    p_test = 1.0 / (1.0 + np.exp(-z_test))

    y_pred = (p_test >= 0.5).astype(int)
    return y_pred
    
