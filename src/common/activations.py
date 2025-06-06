import math
import numpy as np

# ---------- básicas ----------
def step(x: np.ndarray) -> np.ndarray:
    """Escalón bipolar (+1 / -1), vectorizado."""
    return np.where(x >= 0, 1, -1)


def identity(x: float) -> float:          # <- lineal
    return x


def identity_deriv(_: float) -> float:    # derivada constante
    return 1.0


def tanh(x: float) -> float:              # <- no-lineal
    return np.tanh(x)


def tanh_deriv(x: float) -> float:
    return 1.0 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def step_deriv(x):
    # Derivada de step es cero en todos lados excepto en 0 (donde es indefinida)
    return np.zeros_like(x)

# ----------------- soft-max -----------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Soft-max estable numéricamente (resta el máximo).
        x : (..., D)
    Devuelve un array de la misma forma, cada “vector” normalizado a 1.
    """
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_deriv(x):
    """
    Para la combinación habitual *soft-max + cross-entropy* el gradiente
    ∂L/∂z_L = ˆy – y  **ya se calcula** en la propia función de pérdida,
    por lo que la capa de activación NO debe volver a multiplicar por la
    jacobiana completa.  
    Por compatibilidad con tu `layers.py` (que hace
        grad_z = grad * act'(z)
    )
    devolvemos simplemente 1.0 →  grad_z = grad.
    """
    return np.ones_like(x)


# -------------- diccionarios globales --------------
ACT = {
    "identity": identity,
    "tanh":     tanh,
    "sigmoid":  sigmoid,
    "relu":     relu,
    "step":     step,
    "softmax":  softmax,      
}

DACT = {
    "identity": identity_deriv,
    "tanh":     tanh_deriv,
    "sigmoid":  sigmoid_deriv,
    "relu":     relu_deriv,
    "step":     step_deriv,
    "softmax":  softmax_deriv,  
}