import pandas as pd
import numpy as np

def categorical_kernel(train: pd.DataFrame, synth: pd.DataFrame) -> float:
    X = train.to_numpy()
    Y = synth.to_numpy()

    # Reshape for broadcasting: (n, 1, d) vs (1, m, d)
    X_exp = X[:, np.newaxis, :]  # shape: (n, 1, d)
    Y_exp = Y[np.newaxis, :, :]  # shape: (1, m, d)

    match_matrix = (X_exp == Y_exp).mean(axis=2)
    score = match_matrix.mean()
    return score

def normalized_polynomial_kernel(X, Y, degree, gamma, coef) -> float:
    K = (gamma * X @ Y.T + coef) ** degree

    K_xx_diag = np.diag((gamma * (X @ X.T) + coef) ** degree)
    K_yy_diag = np.diag((gamma * (Y @ Y.T) + coef) ** degree) if not np.array_equal(X, Y) else K_xx_diag

    norm_x = np.sqrt(K_xx_diag)[:, np.newaxis]
    norm_y = np.sqrt(K_yy_diag)[np.newaxis, :]
    denom = norm_x * norm_y + 1e-8

    K_normalized = K / denom
    return K_normalized


def normalized_linear_kernel(X, Y) -> float:
    numerator = X @ Y.T
    denom = np.linalg.norm(X, axis=1) @ np.linalg.norm(Y, axis=1).T +1e-8
    return (numerator / (2*denom)) + 0.5

def kernel_score(XX: np.array, YY: np.array, XY: np.array) -> float:
    return XX.mean() + YY.mean() - 2 * XY.mean()