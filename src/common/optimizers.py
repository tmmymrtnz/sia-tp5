"""
Optimizers compatibles con el Trainer (un solo iterable param‑grad).
"""

import math
from collections import defaultdict
import numpy as np


# ---------------------------------------------------------------------
class Optimizer:
    def update(self, params_and_grads):
        raise NotImplementedError


# ---------------------------------------------------------------------
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, **_):
        self.learning_rate = learning_rate

    def update(self, params_and_grads):
        # params_and_grads: iterable of (param, grad)
        for param, grad in params_and_grads:
            param -= self.learning_rate * grad


# ---------------------------------------------------------------------
class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, **_):
        self.learning_rate   = learning_rate
        self.beta = beta
        self.vel  = defaultdict(lambda: 0.0)

    def update(self, params_and_grads):
        for param, grad in params_and_grads:
            k = id(param)
            self.vel[k] = self.beta * self.vel[k] + (1 - self.beta) * grad
            param -= self.learning_rate * self.vel[k]


# ---------------------------------------------------------------------
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **_):
        self.learning_rate, self.b1, self.b2, self.eps = learning_rate, beta1, beta2, eps
        self.t = 0
        self.m = defaultdict(lambda: 0.0)
        self.v = defaultdict(lambda: 0.0)

    def update(self, params_and_grads):
        self.t += 1
        for param, grad in params_and_grads:
            k = id(param)
            # 1er y 2º momento
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grad
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grad * grad)

            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
