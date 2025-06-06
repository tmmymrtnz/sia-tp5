import random
from typing import Callable, Sequence, List


class Perceptron:
    """
    Perceptrón “genérico”:
        • si activation = identity  ->   lineal / ADALINE
        • si activation = tanh      ->   no lineal
        • si activation = step      ->   clasificador duro (Ej. 1)
    """
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        input_size: int,
        learning_rate: float,
        max_epochs: int,
        activation_func: Callable[[float], float],
        activation_deriv: Callable[[float], float] = lambda _: 1.0,
        bias_init: float | None = None,
    ):
        self.weights: List[float] = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias: float = bias_init if bias_init is not None else random.uniform(-1, 0)
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.act = activation_func
        self.act_deriv = activation_deriv

    # ------------------------------------------------------------------ #
    def _weighted_sum(self, inputs: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    # ------------------------------------------------------------------ #
    def predict(self, inputs: Sequence[float]) -> float:
        return self.act(self._weighted_sum(inputs))

    # ------------------------------------------------------------------ #
    def train(self, X: list[list[float]], y: list[float]) -> None:
        for epoch in range(1, self.max_epochs + 1):
            sq_error = 0.0
            for xi, target in zip(X, y):
                a = self._weighted_sum(xi)
                y_pred = self.act(a)

                # Δ-regla (gradiente MSE)
                error = target - y_pred
                grad = error * self.act_deriv(a)

                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * grad * xi[i]
                self.bias += self.lr * grad

                sq_error += error * error

            mse = sq_error / len(X)
            print(f"Epoch {epoch:3d} \tMSE = {mse:.6f}")
            # Heurística de parada: cambio casi nulo
            if mse < 1e-6:
                break
