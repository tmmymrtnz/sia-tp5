import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.insert(0, "src")

from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h

def main():
    font_path = "data/font.h"
    config_path = "configs/ex1/optimal.json"

    if not os.path.exists(font_path):
        raise FileNotFoundError("Falta el archivo font.h")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Falta el archivo de configuración")

    # Lista corregida (32 caracteres, empieza en ` y termina en ~)
    characters = list("`abcdefghijklmnopqrstuvwxyz{|}~") + ["DEL"]

    # Cargar datos (X tiene shape (32, 35))
    X = parse_font_h(font_path)

    # Leer configuración óptima desde JSON
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ae = Autoencoder(
        encoder_sizes       = cfg["encoder"]["layer_sizes"],
        encoder_activations = cfg["encoder"]["activations"],
        decoder_sizes       = cfg["decoder"]["layer_sizes"],
        decoder_activations = cfg["decoder"]["activations"]
    )

    ae.train_mode()
    from common.perceptrons.multilayer.trainer import Trainer
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
        min_delta       = cfg["min_delta"],
    )
    trainer.fit(X, X)
    ae.eval_mode()

    # Obtener espacio latente
    z = ae.encoder.forward(X)  # shape: (32, 2)

    # 🎨 Gráfico
    plt.figure(figsize=(10, 8))

    # Distintos colores para cada punto
    scatter = plt.scatter(z[:, 0], z[:, 1], c=range(32), cmap='tab20', s=300, edgecolors='black')

    # Etiquetas con los caracteres reales
    for i, (x, y) in enumerate(z):
        plt.text(x, y, characters[i], fontsize=12, ha='center', va='center', weight='bold')

    plt.title("Espacio latente (dim=2) de los 32 caracteres")
    plt.xlabel("Latente 1")
    plt.ylabel("Latente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()