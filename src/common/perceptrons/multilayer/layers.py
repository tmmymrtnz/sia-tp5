# ------------------------------------------------------------
#  src/ex3/layers.py          (reemplaza por completo este archivo)
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import sys

# Aseguramos que “src” esté en el path (por si se invoca desde la raíz)
sys.path.insert(0, "src")

# Registro global de activaciones y derivadas
from common.activations import ACT, DACT


class DenseLayer:
    """
    Capa totalmente conectada:      y = ACT( x @ W + b )
        • in_dim, out_dim ─ tamaños de entrada / salida
        • act_name         ─ str (p.e. "tanh", "relu", "softmax", …)
    """
    # ------------------------------------------------------ #
    def __init__(self, in_dim: int, out_dim: int, act_name: str | None = ""):
        self.act_name = (act_name or "identity").lower()

        if self.act_name not in ACT:
            raise ValueError(f"Unknown activation '{self.act_name}'")

        # ---------- inicialización de pesos ----------
        self.W = self._init_weights(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))

        # gradientes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # cachés para back-prop
        self._x: np.ndarray | None = None   # entrada al layer
        self._z: np.ndarray | None = None   # salida lineal (pre-act)

    # ------------------------------------------------------ #
    #  He (ReLU-family)  /  Xavier-Glorot (resto)
    def _init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.act_name in {"relu", "leaky_relu"}:
            std = np.sqrt(2.0 / fan_in)               # He
            return np.random.randn(fan_in, fan_out) * std
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out)) # Xavier/Glorot
            return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    # ------------------------------------------------------ #
    #  FORWARD
    # ------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (B, in_dim)
        retorna (B, out_dim)
        """
        self._x = x
        self._z = x @ self.W + self.b
        return ACT[self.act_name](self._z)

    # ------------------------------------------------------ #
    #  BACKWARD
    # ------------------------------------------------------ #
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out = ∂L/∂a  (del layer superior)
        Devuelve ∂L/∂x para la capa inferior y deja dW/db listos.
        """
        # ∂L/∂z = ∂L/∂a  *  ∂a/∂z
        grad_z = grad_out * DACT[self.act_name](self._z)

        # Gradientes de parámetros (promedio por batch)
        self.dW[:] = self._x.T @ grad_z / len(self._x)
        self.db[:] = grad_z.mean(axis=0, keepdims=True)

        # Propaga a la capa anterior: ∂L/∂x = ∂L/∂z @ Wᵀ
        return grad_z @ self.W.T

    # ------------------------------------------------------ #
    def params_and_grads(self):
        """Generador (param, grad) – útil para los optimizadores."""
        yield self.W, self.dW
        yield self.b, self.db
