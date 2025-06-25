#!/usr/bin/env python3

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from common.perceptrons.multilayer.denoising_trainer import DenoisingTrainer
from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h

# --- Noise Functions ---
def add_binary_noise(X: np.ndarray, noise_level: float, seed: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) < noise_level
    Xn = X.copy()
    Xn[mask] = 1.0 - Xn[mask]
    return Xn

def add_gaussian_noise(X: np.ndarray, std_dev: float, seed: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, std_dev, X.shape)
    Xn = X + noise
    return np.clip(Xn, 0, 1)

# --- Plotting ---
def plot_loss(loss_hist: list, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist, label="Loss", linewidth=1.5)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("checkpoints/dae_loss_plot.png")
    plt.close()
    print(">>> Gráfico de pérdida guardado en 'checkpoints/dae_loss_plot.png'.")

def mostrar_reconstruccion(idx: int, X_noisy, X_clean, X_rec, nombre: str):
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))
    axs[0].imshow(X_noisy[idx].reshape(7, 5), cmap="Greys", vmin=0, vmax=1)
    axs[0].set_title("Entrada ruidosa")
    axs[1].imshow((X_rec[idx] > 0.5).reshape(7, 5), cmap="Greys", vmin=0, vmax=1)
    axs[1].set_title("Reconstrucción")
    axs[2].imshow(X_clean[idx].reshape(7, 5), cmap="Greys", vmin=0, vmax=1)
    axs[2].set_title("Original limpia")
    for ax in axs:
        ax.axis("off")
    plt.suptitle(f"Ejemplo {nombre}")
    plt.tight_layout()
    output_path = f"checkpoints/ejemplo_{nombre.lower()}.png"
    plt.savefig(output_path)
    plt.close()
    print(f">>> Guardado: {output_path}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Entrena un Denoising AE con configuración JSON.")
    parser.add_argument("config_path", help="Ruta al JSON de configuración (p. ej. optimal.json)")
    parser.add_argument("font_path", help="Ruta al archivo font.h con Font3[32][7]")
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No se encontró config: {args.config_path}")
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"No se encontró font.h: {args.font_path}")

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    X_clean = parse_font_h(args.font_path)
    noise_level = cfg.get("noise_level", 0.1)
    noise_type = cfg.get("noise_type", "binary")

    if noise_type == "binary":
        noise_fn = add_binary_noise
    elif noise_type == "gaussian":
        noise_fn = add_gaussian_noise
    else:
        raise ValueError(f"Tipo de ruido no soportado: '{noise_type}'")

    print(f">>> Entrenando con ruido '{noise_type}' nivel={noise_level} …")

    ae = Autoencoder(
        encoder_sizes       = cfg["encoder"]["layer_sizes"],
        encoder_activations = cfg["encoder"]["activations"],
        decoder_sizes       = cfg["decoder"]["layer_sizes"],
        decoder_activations = cfg["decoder"]["activations"],
        dropout_rate        = cfg.get("dropout_rate", 0.0)
    )

    trainer = DenoisingTrainer(
        noise_fn       = noise_fn,
        noise_level    = noise_level,
        net            = ae,
        loss_name      = cfg["loss"],
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {"learning_rate": cfg["lr"]}),
        batch_size     = cfg["batch_size"],
        max_epochs     = cfg["max_epochs"],
        log_every      = cfg.get("log_every", 100),
        early_stopping = True,
        patience       = cfg["patience"],
        min_delta      = cfg["min_delta"]
    )

    print(">>> Entrenando Denoising Autoencoder...")
    loss_hist = trainer.fit(X_clean, X_clean)

    # Evaluar
    ae.eval_mode()
    X_noisy = noise_fn(X_clean, noise_level)
    X_rec = ae.forward(X_noisy)
    X_bin = (X_rec > 0.5).astype(int)
    diffs = np.abs(X_bin - X_clean)
    errs = diffs.sum(axis=1).astype(int)

    print(">>> Evaluación:")
    print("   → Promedio de bits incorrectos:", float(errs.mean()))
    print("   → Máximo de bits incorrectos:", int(errs.max()))
    print("   → Errores por carácter:", errs.tolist())

    os.makedirs("checkpoints", exist_ok=True)
    ae.save_weights("checkpoints/dae_weights.npz")
    np.save("checkpoints/dae_loss_history.npy", np.array(loss_hist))
    plot_loss(loss_hist, "Pérdida del Denoising Autoencoder")

    idx_correcto = next((i for i, e in enumerate(errs) if e == 0), None)
    idx_erroneo  = next((i for i, e in enumerate(errs) if e > 0), None)

    if idx_correcto is not None:
        mostrar_reconstruccion(idx_correcto, X_noisy, X_clean, X_rec, "Correcto")
    if idx_erroneo is not None:
        mostrar_reconstruccion(idx_erroneo, X_noisy, X_clean, X_rec, "Erroneo")

if __name__ == "__main__":
    main()