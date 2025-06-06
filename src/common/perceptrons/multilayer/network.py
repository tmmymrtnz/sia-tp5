from typing import List, Union
import numpy as np

from .layers import DenseLayer


class MLP:
    """
    Stack de capas densas con activaciones:
        layer_sizes = [in, h1, ..., out]
        activations = ["linear", "tanh", ..., "sigmoid"]
    Por convenio, activations[0] corresponde a la entrada (se ignora).
    
    Parámetros:
        dropout_rate: Probabilidad de desactivar neuronas durante el entrenamiento.
                     Aplicado a todas las capas ocultas (no a la capa de salida).
    """
    # ------------------------------------------------------ #
    def __init__(self, layer_sizes: List[int], activations: List[str], dropout_rate: float = 0.0):
        # Aceptamos tanto N  como  N‑1 activaciones (sin la de entrada).
        if len(activations) == len(layer_sizes) - 1:
            activations = ["linear"] + activations
        assert len(layer_sizes) == len(activations), \
            "`activations` debe tener len = layer_sizes  ó  layer_sizes-1"
            
        # Parámetros de dropout
        self.dropout_rate = dropout_rate
        self.is_training = True  # Por defecto en modo entrenamiento

        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    in_dim=layer_sizes[i],
                    out_dim=layer_sizes[i + 1],
                    act_name=activations[i + 1]   # se salta la de entrada
                )
            )

    # ------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante con soporte para dropout."""
        current = x
        
        # Para cada capa excepto la última
        for i, layer in enumerate(self.layers[:-1]):
            # Forward de la capa
            current = layer.forward(current)
            
            # Aplicar dropout solo durante el entrenamiento
            if self.is_training and self.dropout_rate > 0:
                # Crear máscara de dropout (1: mantener, 0: desactivar)
                mask = (np.random.rand(*current.shape) > self.dropout_rate).astype(float)
                # Aplicar máscara y escalar para mantener la magnitud esperada
                current = current * mask / (1.0 - self.dropout_rate)
        
        # Aplicar la última capa (sin dropout)
        return self.layers[-1].forward(current)

    # ------------------------------------------------------ #
    def backward(self, loss_grad: np.ndarray) -> None:
        """Propagación hacia atrás."""
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    # ------------------------------------------------------ #
    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()
            
    # ------------------------------------------------------ #
    def train_mode(self):
        """Activa el modo entrenamiento (con dropout)."""
        self.is_training = True
        
    def eval_mode(self):
        """Activa el modo evaluación/inferencia (sin dropout)."""
        self.is_training = False