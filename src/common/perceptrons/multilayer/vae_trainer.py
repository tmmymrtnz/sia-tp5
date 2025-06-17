from common.perceptrons.multilayer.trainer import Trainer
import numpy as np

class VAETrainer(Trainer):
    """Trainer especializado para VAE que maneja la pérdida personalizada"""
    
    def __init__(
        self,
        vae,          # El modelo VAE 
        loss_name,    # "mse" o "bce" para reconstrucción
        beta=1.0,     # Factor para KL divergence
        **kwargs      # Los demás parámetros para Trainer
    ):
        super().__init__(net=vae, loss_name=loss_name, **kwargs)
        self.vae = vae
        self.beta = beta
    
    def _loss_and_grad(self, y_hat, y_true):
        """Sobreescribe para incluir KL divergence"""
        # Obtener pérdida y gradiente de reconstrucción normal
        recon_loss, recon_grad = super()._loss_and_grad(y_hat, y_true)
        
        # Añadir KL divergence
        kl_loss = self.vae.kl_divergence() / len(y_true)
        total_loss = recon_loss + self.beta * kl_loss
        
        # El gradiente es el mismo, el término KL se aplica en backward del VAE
        
        return total_loss, recon_grad