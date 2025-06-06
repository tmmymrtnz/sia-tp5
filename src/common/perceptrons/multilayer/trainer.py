import numpy as np
from common.losses      import mse, cross_entropy, binary_cross_entropy
from common.optimizers  import SGD, Momentum, Adam
from .network           import MLP
from sklearn.metrics    import accuracy_score, f1_score

LOSS_FUNS = {
    "mse"           : mse,
    "cross_entropy": cross_entropy,
    "bce"           : binary_cross_entropy
}

OPTIMIZERS = {
    "sgd"     : SGD,
    "momentum": Momentum,
    "adam"    : Adam
}

class Trainer:
    """
    Handles the training loop:
        • shuffles every epoch
        • splits into mini-batches
        • forward → loss → grad
        • back-prop → optimiser update
        • optional weight/stat logging
    """
    def __init__(
        self,
        net            : MLP,
        loss_name      : str,
        optimizer_name : str,
        optim_kwargs   : dict,
        batch_size     : int,
        max_epochs     : int,
        log_every      : int = 1000,
        log_weights    : bool = False,
        early_stopping : bool = True,
        patience       : int = 10,
        min_delta      : float = 1e-4
    ):
        self.net            = net
        self.loss_fn        = LOSS_FUNS[loss_name]
        self.optim          = OPTIMIZERS[optimizer_name](**optim_kwargs)
        self.batch          = batch_size
        self.epochs         = max_epochs
        self.log_every      = max(1, log_every)
        self.log_weights    = log_weights
        self.weight_hist    = []
        self.early_stopping = early_stopping
        self.patience       = patience
        self.min_delta      = min_delta

    def _loss_and_grad(self, y_hat: np.ndarray, y_true: np.ndarray):
        """
        Returns (scalar_loss, grad_wrt_y_hat).
        Supports:
          - mse
          - binary_cross_entropy (sigmoid output + BCE)
          - cross_entropy     (softmax output + CCE)
        """
        if self.loss_fn is mse:
            loss = mse(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        if self.loss_fn is binary_cross_entropy:
            loss = binary_cross_entropy(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        if self.loss_fn is cross_entropy:
            loss = cross_entropy(y_true, y_hat)
            grad = (y_hat - y_true) / len(y_true)
            return loss, grad

        raise ValueError(f"Loss function not supported: {self.loss_fn}")

    def _log_weights(self, epoch: int):
        flat = np.concatenate([w.ravel() for w, _ in self.net.params_and_grads()])
        mean, abs_max = flat.mean(), np.abs(flat).max()
        print(f"    ↳ weights: mean={mean:+.4e}, |w|_∞={abs_max:.4e}")
        if self.log_weights:
            snapshot = [w.copy() for w, _ in self.net.params_and_grads()]
            self.weight_hist.append((epoch, snapshot))
    
    def _log_metrics(self, X, Y, epoch: int):
        y_probs = self.net.forward(X)
        y_pred = np.argmax(y_probs, axis=1)
        y_true = np.argmax(Y, axis=1)

        acc = accuracy_score(y_true, y_pred)

        # Detect if binary or multiclass
        is_binary = Y.shape[1] == 2

        if is_binary:
            f1 = f1_score(y_true, y_pred, average="binary")
        else:
            f1 = f1_score(y_true, y_pred, average="macro")  # or 'weighted' if preferred

        print(f"    ↳ train metrics: acc={acc:.4f}, f1={f1:.4f}")

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Entrena la red con los datos proporcionados.
        Maneja automáticamente los modos de entrenamiento/evaluación si están disponibles.
        
        Parámetros:
            X: Datos de entrada
            Y: Etiquetas/objetivos
            
        Retorna:
            loss_history: Lista con el valor de pérdida en cada época
        """
        # Activar modo entrenamiento (con dropout) si está disponible
        if hasattr(self.net, 'train'):
            self.net.train()
        
        N = len(X)
        prev_loss = float('inf')
        patience_counter = 0
        loss_history = []   # Para guardar el historial de pérdida

        for epoch in range(1, self.epochs + 1):
            # shuffle
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0
            for i in range(0, N, self.batch):
                xb = X[i : i + self.batch]
                yb = Y[i : i + self.batch]

                # forward
                y_hat = self.net.forward(xb)

                # loss & gradient
                loss_batch, grad_yhat = self._loss_and_grad(y_hat, yb)
                epoch_loss += loss_batch * len(xb)

                # backward + update
                self.net.backward(grad_yhat)
                self.optim.update(self.net.params_and_grads())

            epoch_loss /= N
            loss_history.append(epoch_loss)
            
            if epoch % self.log_every == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:>5d}/{self.epochs} | loss={epoch_loss:.6f}")
                self._log_weights(epoch)
                self._log_metrics(X, Y, epoch)

            # ----- early stopping condition -----
            if self.early_stopping:
                if abs(prev_loss - epoch_loss) < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
                prev_loss = epoch_loss

                if patience_counter >= self.patience:
                    print(f"Stopping early at epoch {epoch} (no improvement for {self.patience} epochs)")
                    self._log_weights(epoch)
                    self._log_metrics(X, Y, epoch)
                    break
    
        # Al terminar el entrenamiento, activar modo evaluación (sin dropout)
        if hasattr(self.net, 'eval'):
            self.net.eval()
        
        return loss_history