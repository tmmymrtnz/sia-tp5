from typing import List
import numpy as np
from common.perceptrons.multilayer.network import MLP

class VAE:
    """
    Autoencoder Variacional (VAE) que extiende el autoencoder tradicional
    utilizando distribuciones probabilísticas para el espacio latente.
    """

    def __init__(
        self,
        encoder_sizes: List[int],  # Última capa debe ser 2*latent_dim
        encoder_activations: List[str],
        decoder_sizes: List[int],  # Primera capa debe ser latent_dim
        decoder_activations: List[str],
        latent_dim: int,
        dropout_rate: float = 0.0
    ):
        self.latent_dim = latent_dim
        
        # Verificar que el encoder produzca 2*latent_dim (para μ y log σ²)
        if encoder_sizes[-1] != 2 * latent_dim:
            raise ValueError(f"La última capa del encoder debe tener tamaño {2*latent_dim}")
        
        # Verificar que el decoder reciba latent_dim
        if decoder_sizes[0] != latent_dim:
            raise ValueError(f"La primera capa del decoder debe tener tamaño {latent_dim}")
        
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
        
        # Para almacenar valores intermedios durante forward/backward
        self._means = None
        self._logvars = None
        self._z = None
        self._epsilon = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propaga los datos a través del VAE:
        1) Codifica la entrada para obtener μ y log σ²
        2) Muestrea z utilizando el truco de reparametrización
        3) Decodifica z para obtener la reconstrucción
        """
        # Obtener μ y log σ² del encoder
        encoder_output = self.encoder.forward(x)
        batch_size = encoder_output.shape[0]
        
        # Separar μ y log σ²
        self._means = encoder_output[:, :self.latent_dim]
        self._logvars = encoder_output[:, self.latent_dim:]
        
        # Muestrear del espacio latente usando el truco de reparametrización
        self._epsilon = np.random.randn(batch_size, self.latent_dim)
        self._z = self._means + np.exp(0.5 * self._logvars) * self._epsilon
        
        # Decodificar
        x_recon = self.decoder.forward(self._z)
        return x_recon
    
    def backward(self, grad_out: np.ndarray) -> None:
        """
        Retropropagación para el VAE, considerando el truco de reparametrización.
        """
        # 1) Retropropagación a través del decoder
        grad = grad_out
        for layer in reversed(self.decoder.layers):
            grad = layer.backward(grad)
        
        # grad ahora es dL/dz
        grad_z = grad
        
        # 2) Retropropagación a través del muestreo (truco de reparametrización)
        grad_mean = grad_z  # dL/dμ = dL/dz
        grad_logvar = grad_z * 0.5 * np.exp(0.5 * self._logvars) * self._epsilon  # dL/d(log σ²)
        
        # 3) Combinar gradientes para la salida del encoder
        grad_encoder_output = np.concatenate([grad_mean, grad_logvar], axis=1)
        
        # 4) Retropropagación a través del encoder
        grad2 = grad_encoder_output
        for layer in reversed(self.encoder.layers):
            grad2 = layer.backward(grad2)
    
    def kl_divergence(self) -> float:
        """
        Calcula la divergencia KL entre la distribución latente q(z|x) y la prior p(z)~N(0,I).
        KL(q(z|x) || p(z)) = 0.5 * sum(exp(log σ²) + μ² - 1 - log σ²)
        """
        return 0.5 * np.sum(np.exp(self._logvars) + self._means**2 - 1 - self._logvars)
    
    def params_and_grads(self):
        """Iterador para los parámetros y gradientes."""
        yield from self.encoder.params_and_grads()
        yield from self.decoder.params_and_grads()

    def train_mode(self):
        """Activa el modo entrenamiento."""
        self.encoder.train_mode()
        self.decoder.train_mode()

    def eval_mode(self):
        """Activa el modo evaluación."""
        self.encoder.eval_mode()
        self.decoder.eval_mode()

    def save_weights(self, path: str):
        """Guarda los pesos en un archivo .npz."""
        params = [param for (param, _) in self.params_and_grads()]
        np.savez(path, *params)

    def load_weights(self, path: str):
        """Carga los pesos desde un archivo .npz."""
        archive = np.load(path)
        params = [param for (param, _) in self.params_and_grads()]
        for i, param in enumerate(params):
            key = f"arr_{i}"
            if key not in archive:
                raise ValueError(f"Clave '{key}' no encontrada en '{path}'")
            param[:] = archive[key]
    
    def generate(self, n_samples: int = 1) -> np.ndarray:
        """
        Genera muestras aleatorias decodificando puntos del espacio latente.
        """
        z_samples = np.random.randn(n_samples, self.latent_dim)
        return self.decoder.forward(z_samples)