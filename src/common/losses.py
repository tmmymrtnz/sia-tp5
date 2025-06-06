import math
import numpy as np

# ------------------------------------------------------------------
def mse(y_true, y_pred) -> float:
    """Mean-Squared Error for numerical arrays."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))

# ------------------------------------------------------------------
def cross_entropy(y_true, y_pred, epsilon=1e-12) -> float:
    """Categorical Cross-Entropy loss for one-hot targets and softmax outputs."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Clamp predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # If batch of samples
    if y_pred.ndim == 2:
        # average over batch
        return float(-np.sum(y_true * np.log(y_pred)) / y_pred.shape[0])
    # single sample
    return float(-np.sum(y_true * np.log(y_pred)))

# ------------------------------------------------------------------
def binary_cross_entropy(y_true, y_pred, eps=1e-12) -> float:
    """Binary Cross-Entropy for sigmoid outputs."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))