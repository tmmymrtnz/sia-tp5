from typing import List
import numpy as np
from common.perceptrons.multilayer.network import MLP   # Ajusta la importación según tu estructura de carpetas


class Autoencoder:
    """
    Autoencoder simple formado por un codificador (encoder) y un decodificador (decoder),
    ambos basados en tu clase MLP.
    """

    def __init__(
        self,
        encoder_sizes: List[int],
        encoder_activations: List[str],
        decoder_sizes: List[int],
        decoder_activations: List[str],
        dropout_rate: float = 0.0
    ):
        # Instanciar el codificador
        self.encoder = MLP(
            layer_sizes=encoder_sizes,
            activations=encoder_activations,
            dropout_rate=dropout_rate
        )
        # Instanciar el decodificador
        self.decoder = MLP(
            layer_sizes=decoder_sizes,
            activations=decoder_activations,
            dropout_rate=dropout_rate
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        1) Propaga x a través del encoder → obtiene z (latente)
        2) Propaga z a través del decoder → obtiene x_recon
        """
        self._latent = self.encoder.forward(x)   # guardamos el latente
        x_recon = self.decoder.forward(self._latent)
        return x_recon

    def backward(self, grad_out: np.ndarray) -> None:
        """
        Backprop manual para capturar correctamente el gradiente intermedio:
          1) Backprop a través del decoder → grad_latent
          2) Backprop a través del encoder → actualiza gradientes del encoder

        NOTA: No usamos `self.decoder.backward(...)` directamente porque ese método
        NO devuelve el gradiente con respecto a su entrada. Por eso hacemos el
        loop manual aquí (igual que en MLP.backward).
        """
        # ----- 1) Backprop en el DECODIFICADOR para obtener grad_latent -----
        grad = grad_out
        for layer in reversed(self.decoder.layers):
            grad = layer.backward(grad)
        grad_latent = grad

        # ----- 2) Backprop en el CODIFICADOR con grad_latent -----
        grad2 = grad_latent
        for layer in reversed(self.encoder.layers):
            grad2 = layer.backward(grad2)

    def params_and_grads(self):
        """
        Iterador que va devolviendo (param_array, grad_array) para
        encoder y decoder, en el orden que espera el optimizador.
        """
        yield from self.encoder.params_and_grads()
        yield from self.decoder.params_and_grads()

    def train_mode(self):
        """Activa el modo entrenamiento (para dropout, si aplica)."""
        self.encoder.train_mode()
        self.decoder.train_mode()

    def eval_mode(self):
        """Activa el modo evaluación/inferencia (sin dropout)."""
        self.encoder.eval_mode()
        self.decoder.eval_mode()

    def save_weights(self, path: str):
        """
        Guarda todos los parámetros (W, b, ...) en un único archivo .npz.
        El orden es el que devuelve params_and_grads().
        """
        params = [param for (param, _) in self.params_and_grads()]
        np.savez(path, *params)

    def load_weights(self, path: str):
        """
        Carga un archivo .npz previamente guardado con save_weights().
        Sobrescribe cada parámetro (por índice) con el array correspondiente.
        """
        archive = np.load(path)
        params = [param for (param, _) in self.params_and_grads()]
        for i, param in enumerate(params):
            key = f"arr_{i}"
            if key not in archive:
                raise ValueError(f"Clave '{key}' no encontrada en '{path}'")
            param[:] = archive[key]
