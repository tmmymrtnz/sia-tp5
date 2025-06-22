import numpy as np
from common.perceptrons.multilayer.trainer import Trainer

class VAETrainer(Trainer):
    """
    Entrenador especializado para VAE.
    – Calcula ELBO = recon + β·KL
    – Soporta annealing lineal de β durante kl_ramp_epochs
    """

    def __init__(
        self,
        vae,
        loss_name,
        beta=1.0,
        kl_ramp_epochs=0,
        **kwargs
    ):
        super().__init__(net=vae, loss_name=loss_name, **kwargs)
        self.vae            = vae
        self.beta_final     = beta
        self.kl_ramp_epochs = kl_ramp_epochs
        self.beta           = 0.0 if kl_ramp_epochs else beta

    # -------------------------------------------------- #
    def _loss_and_grad(self, y_hat: np.ndarray, y_true: np.ndarray):
        recon_loss, recon_grad = super()._loss_and_grad(y_hat, y_true)
        kl_loss  = self.vae.kl_divergence() / len(y_true)
        total    = recon_loss + self.beta * kl_loss
        return total, recon_grad

    # -------------------------------------------------- #
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Entrena la red; incluye early-stopping y annealing de β.
        """
        N = len(X)
        prev_loss, patience_cnt = float("inf"), 0
        loss_hist = []

        # -------- modo entrenamiento (dropout on) -------
        self.net.train_mode() if hasattr(self.net, "train_mode") else None

        for epoch in range(1, self.epochs + 1):

            # β annealing lineal 0→β_final
            if self.kl_ramp_epochs:
                ramp = min(1.0, epoch / self.kl_ramp_epochs)
                self.beta = self.beta_final * ramp

            # shuffle
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0
            for i in range(0, N, self.batch):
                xb, yb = X[i : i + self.batch], Y[i : i + self.batch]

                y_hat = self.net.forward(xb)
                loss_b, grad_y = self._loss_and_grad(y_hat, yb)
                epoch_loss += loss_b * len(xb)

                self.net.backward(grad_y)
                self.optim.update(self.net.params_and_grads())

            epoch_loss /= N
            loss_hist.append(epoch_loss)

            if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:>4d}/{self.epochs} | β={self.beta:.3f} | loss={epoch_loss:.6f}")

            # ---------- early-stopping ----------
            if self.early_stopping:
                if abs(prev_loss - epoch_loss) < self.min_delta:
                    patience_cnt += 1
                else:
                    patience_cnt = 0
                prev_loss = epoch_loss
                if patience_cnt >= self.patience:
                    print(f"Early stop en epoch {epoch} (sin mejora en {self.patience} ep.)")
                    break

        # -------- modo evaluación (dropout off) ---------
        self.net.eval_mode() if hasattr(self.net, "eval_mode") else None
        return loss_hist
