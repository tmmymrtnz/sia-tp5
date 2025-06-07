import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from autoencoder.autoencoder import Autoencoder


def decode_latent_point(x: float, y: float, ae: Autoencoder) -> np.ndarray:
    z = np.array([[x, y]])  # shape (1, 2)
    out = ae.decoder.forward(z)  # shape (1, 35)
    out_bin = (out > 0.5).astype(int)[0]
    return out_bin.reshape(7, 5)


def matrix_to_string(matrix: np.ndarray) -> str:
    return "\n".join("".join("*" if x else " " for x in row) for row in matrix)


def print_latent_grid(ae: Autoencoder, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), steps=5):
    x_vals = np.linspace(*x_range, steps)
    y_vals = np.linspace(*y_range, steps)[::-1]  # invert Y so top is top

    print("\n=== Generación de caracteres en la grilla del espacio latente ===\n")

    char_matrices = [[decode_latent_point(x, y, ae) for x in x_vals] for y in y_vals]

    # Imprimir una fila por vez (7 líneas por fila, cada celda de 5 caracteres)
    for row in char_matrices:
        for i in range(7):  # cada letra tiene 7 filas
            line = "   ".join("".join("*" if x else " " for x in char[i]) for char in row)
            print(line)
        print("\n")  # espacio entre filas


def plot_latent_grid(ae: Autoencoder, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), steps=6):
    x_vals = np.linspace(*x_range, steps)
    y_vals = np.linspace(*y_range, steps)[::-1]

    fig, axs = plt.subplots(steps, steps, figsize=(steps, steps))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            ax = axs[i, j]
            letter = decode_latent_point(x, y, ae)

            ax.imshow(letter, cmap="Greys", vmin=0, vmax=1)
            ax.axis("off")

    plt.suptitle("Grid de generación desde el espacio latente", fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    weights_path = "checkpoints/ae_weights_A.npz"

    if not os.path.exists(weights_path):
        raise FileNotFoundError("Missing weights at checkpoints/ae_weights_A.npz")

    ae = Autoencoder(
        encoder_sizes=[35, 64, 32, 16, 8, 2],
        encoder_activations=["tanh", "tanh", "tanh", "tanh", "identity"],
        decoder_sizes=[2, 8, 16, 32, 64, 35],
        decoder_activations=["tanh", "tanh", "tanh", "tanh", "sigmoid"]
    )
    ae.load_weights(weights_path)
    ae.eval_mode()

    print_latent_grid(ae, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), steps=6)
    plot_latent_grid(ae, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), steps=6)


if __name__ == "__main__":
    main()