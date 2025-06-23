import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json

sys.path.insert(0, "src")
from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h
from common.perceptrons.multilayer.trainer import Trainer

def decode_latent_point(x: float, y: float, ae: Autoencoder) -> np.ndarray:
    z = np.array([[x, y]])  # shape (1, 2)
    out = ae.decoder.forward(z)  # shape (1, 35)
    out_bin = (out > 0.5).astype(int)[0]
    return out_bin.reshape(7, 5)

def plot_latent_grid(ae: Autoencoder, x_range, y_range, steps):
    x_vals = np.linspace(*x_range, steps)
    y_vals = np.linspace(*y_range, steps)[::-1]  # de arriba hacia abajo

    fig, axs = plt.subplots(steps, steps, figsize=(steps, steps))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            ax = axs[i, j]
            letter = decode_latent_point(x, y, ae)
            ax.imshow(letter, cmap="Greys", vmin=0, vmax=1)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    config_path = "configs/ex1/optimal.json"
    font_path = "data/font.h"

    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config file at configs/ex1/optimal.json")
    if not os.path.exists(font_path):
        raise FileNotFoundError("Missing font.h")

    X = parse_font_h(font_path)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    ae = Autoencoder(
        encoder_sizes       = cfg["encoder"]["layer_sizes"],
        encoder_activations = cfg["encoder"]["activations"],
        decoder_sizes       = cfg["decoder"]["layer_sizes"],
        decoder_activations = cfg["decoder"]["activations"]
    )

    ae.train_mode()
    trainer = Trainer(
        net             = ae,
        loss_name       = cfg["loss"],
        optimizer_name  = cfg["optimizer"],
        optim_kwargs    = {"learning_rate": cfg["lr"]},
        batch_size      = cfg["batch_size"],
        max_epochs      = cfg["max_epochs"],
        log_every       = cfg["log_every"],
        early_stopping  = True,
        patience        = cfg["patience"],
        min_delta       = cfg["min_delta"]
    )
    trainer.fit(X, X)
    ae.eval_mode()

    # Obtener latentes reales
    z = ae.encoder.forward(X)  # (32, 2)
    x_min, x_max = z[:, 0].min(), z[:, 0].max()
    y_min, y_max = z[:, 1].min(), z[:, 1].max()

    # Expandir márgenes un poco
    margin = 1.0
    x_range = (x_min - margin, x_max + margin)
    y_range = (y_min - margin, y_max + margin)

    # Mostrar rangos
    print(f"Espacio latente X: {x_range}")
    print(f"Espacio latente Y: {y_range}")

    # Graficar grids
    for steps in [5, 10, 20]:
        print(f"\nGenerando grid de {steps}x{steps} …")
        plot_latent_grid(ae, x_range, y_range, steps)

if __name__ == "__main__":
    main()