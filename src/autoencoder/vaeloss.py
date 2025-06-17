import numpy as np

class VAELoss():
    """
    Función de pérdida para VAE que combina error de reconstrucción (MSE/BCE) y divergencia KL.
    """
    
    def __init__(self, reconstruction_loss_name='mse', beta=1.0):
        """
        Args:
            reconstruction_loss_name: 'mse' o 'bce' (binary cross entropy)
            beta: Factor para ponderar el término KL (β-VAE)
        """
        self.recon_loss_name = reconstruction_loss_name
        self.beta = beta
        
    def forward(self, y_pred, y_true, vae_model):
        """
        Calcula la pérdida total del VAE:
        Loss = ReconstrucciónLoss + β * KL
        """
        # Error de reconstrucción
        if self.recon_loss_name == 'mse':
            recon_loss = np.mean((y_pred - y_true) ** 2)
        elif self.recon_loss_name == 'bce':
            epsilon = 1e-10
            recon_loss = -np.mean(
                y_true * np.log(y_pred + epsilon) + 
                (1 - y_true) * np.log(1 - y_pred + epsilon)
            )
        else:
            raise ValueError(f"Función de pérdida no reconocida: {self.recon_loss_name}")
        
        # Divergencia KL
        kl_loss = vae_model.kl_divergence() / y_pred.shape[0]  # Normalizar por batch
        
        # Pérdida total
        total_loss = recon_loss + self.beta * kl_loss
        
        # Guardar para backward
        self.y_pred = y_pred
        self.y_true = y_true
        self.recon_loss = recon_loss
        self.kl_loss = kl_loss
        
        return total_loss
    
    def backward(self):
        """
        Calcula el gradiente de la pérdida con respecto a las predicciones.
        """
        if self.recon_loss_name == 'mse':
            grad = 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]
        elif self.recon_loss_name == 'bce':
            epsilon = 1e-10
            grad = -(self.y_true / (self.y_pred + epsilon) - 
                    (1 - self.y_true) / (1 - self.y_pred + epsilon)) / self.y_pred.shape[0]
        
        return grad