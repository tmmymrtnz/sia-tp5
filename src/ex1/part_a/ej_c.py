# src/experimentos/graficar_latente.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, "src")

from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h
from common.perceptrons.multilayer.network import MLP

def main():
    font_path = "data/font.h"
    weights_path = "checkpoints/ae_weights.npz"

    # Verifica existencia de archivos
    if not os.path.exists(font_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("Falta font.h o ae_weights.npz")

    # Cargar datos (X tiene shape (32, 35))
    X = parse_font_h(font_path)

    ae = Autoencoder(
        encoder_sizes=[35, 16, 2],
        encoder_activations=["", "tanh"],
        decoder_sizes=[2, 16, 35],
        decoder_activations=["tanh", "sigmoid"]
    )
    ae.load_weights(weights_path)
    ae.eval_mode()

    # Obtener espacio latente
    z = ae.encoder.forward(X)  # shape: (32, 2)

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c='skyblue', edgecolor='black')

    for i, (x, y) in enumerate(z):
        print("Point i: (", x, ", ", y, ")")
        plt.text(x, y, str(i), fontsize=9, ha='center', va='center')

    plt.title("Espacio latente (dim=2) de los 32 caracteres")
    plt.xlabel("Latente 1")
    plt.ylabel("Latente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()