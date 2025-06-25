import numpy as np
from common.perceptrons.multilayer.trainer import Trainer

# Un trainer especializado para Autoencoders Denoising, que aplica ruido a los datos de entrada en CADA EPOCH
class DenoisingTrainer(Trainer):
    def __init__(self, noise_fn, noise_level=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_fn = noise_fn
        self.noise_level = noise_level

    def fit(self, X_clean: np.ndarray, Y_clean: np.ndarray):
        """
        Rewritten fit method to be more robust against subtle bugs.
        """
        N = len(X_clean)
        loss_hist = []
        
        # Initialize early stopping variables
        prev_loss, patience_cnt = float("inf"), 0
        
        self.net.train_mode() if hasattr(self.net, "train_mode") else None

        for epoch in range(1, self.epochs + 1):
            # Create a fresh noisy version of the dataset for this epoch
            X_noisy = self.noise_fn(X_clean, self.noise_level)

            # Create a shuffled list of indices for this epoch
            epoch_indices = np.random.permutation(N)

            epoch_loss = 0.0
            # Iterate through the data using the shuffled indices
            for i in range(0, N, self.batch):
                # Get the indices for the current batch from our shuffled list
                batch_indices = epoch_indices[i:i + self.batch]

                # Use these indices to get the correct noisy inputs and clean targets
                # This explicitly guarantees that X_noisy[k] corresponds to Y_clean[k]
                xb = X_noisy[batch_indices]
                yb = Y_clean[batch_indices]

                # Standard training steps
                y_hat = self.net.forward(xb)
                loss_b, grad_y = self._loss_and_grad(y_hat, yb)
                
                epoch_loss += loss_b * len(xb)

                self.net.backward(grad_y)
                self.optim.update(self.net.params_and_grads())

            # Calculate the average loss for the epoch.
            epoch_loss /= N
            loss_hist.append(epoch_loss)

            if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:>4d}/{self.epochs} | loss={epoch_loss:.6f}")

            # Early stopping logic
            if self.early_stopping:
                if abs(prev_loss - epoch_loss) < self.min_delta:
                    patience_cnt += 1
                else:
                    patience_cnt = 0
                
                if patience_cnt >= self.patience:
                    print(f"Early stop en epoch {epoch} (sin mejora en {self.patience} ep.)")
                    break
                prev_loss = epoch_loss

        self.net.eval_mode() if hasattr(self.net, "eval_mode") else None
        return loss_hist