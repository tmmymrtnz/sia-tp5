import os
import re
import json
import argparse
from common.perceptrons.multilayer.denoising_trainer import DenoisingTrainer
import numpy as np
import matplotlib.pyplot as plt

from common.perceptrons.multilayer.network import MLP
from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h

def add_binary_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    rng = np.random.RandomState()
    mask = rng.rand(*X.shape) < noise_level
    Xn = X.copy()
    Xn[mask] = 1.0 - Xn[mask]
    return Xn

def add_gaussian_noise(X: np.ndarray, std_dev: float) -> np.ndarray:
    noise = np.random.normal(0, std_dev, X.shape)
    Xn = X + noise
    return np.clip(Xn, 0, 1)  

def plot_loss_avg(losses: list, title: str):
    min_len = min(len(hist) for hist in losses)
    losses_cropped = [hist[:min_len] for hist in losses]
    losses_np = np.array(losses_cropped)
    avg = losses_np.mean(axis=0)
    std = losses_np.std(axis=0)

    plt.figure(figsize=(8, 4))
    epochs = np.arange(min_len)
    plt.plot(epochs, avg, label="Promedio de pérdida", color="blue")
    plt.fill_between(epochs, avg - std, avg + std, color="blue", alpha=0.2, label="Desvío estándar")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("checkpoints/dae_loss_avg_plot.png")
    plt.close()
    print(">>> Gráfico de pérdida promedio guardado en 'checkpoints/dae_loss_avg_plot.png'.")

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

def main():
    parser = argparse.ArgumentParser(description="Entrena múltiples veces un Denoising AE.")
    parser.add_argument("config_path", help="Ruta al JSON de configuración")
    parser.add_argument("font_path", help="Ruta al archivo font.h")
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No se encontró config: {args.config_path}")
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"No se encontró font.h: {args.font_path}")

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    X_clean = parse_font_h(args.font_path)
    noise_level = cfg.get("noise_level", 0.1)

    N = 5
    all_loss_hist = []
    all_errs = []
    recon_samples = []

    for run in range(N):
        print(f"\n>>> Corrida {run + 1}/{N}")

        ae = Autoencoder(
            encoder_sizes       = cfg["encoder"]["layer_sizes"],
            encoder_activations = cfg["encoder"]["activations"],
            decoder_sizes       = cfg["decoder"]["layer_sizes"],
            decoder_activations = cfg["decoder"]["activations"],
            dropout_rate        = cfg.get("dropout_rate", 0.0)
        )

        # Seleccionamos qué tipo de ruido usar según el config
        noise_type = cfg.get("noise_type", "binary")
        if noise_type == "binary":
            noise_fn = add_binary_noise
        elif noise_type == "gaussian":
            noise_fn = add_gaussian_noise
        else:
            raise ValueError(f"Tipo de ruido no soportado: '{noise_type}'")

        trainer = DenoisingTrainer(
            noise_fn       = noise_fn,
            noise_level    = noise_level,
            net            = ae,
            loss_name      = cfg["loss"],
            optimizer_name = cfg["optimizer"],
            optim_kwargs   = {"learning_rate": cfg["lr"]},
            batch_size     = cfg["batch_size"],
            max_epochs     = cfg["max_epochs"],
            log_every      = cfg.get("log_every", 100),
            early_stopping = True,
            patience       = cfg["patience"],
            min_delta      = cfg["min_delta"]
        )

        loss_hist = trainer.fit(X_clean, X_clean)
        all_loss_hist.append(loss_hist)

        # Evaluar con un nuevo batch ruidoso generado ahora
        ae.eval_mode()
        X_noisy = noise_fn(X_clean, noise_level)
        X_rec = ae.forward(X_noisy)
        X_bin = (X_rec > 0.5).astype(int)
        diffs = np.abs(X_bin - X_clean)
        errs = diffs.sum(axis=1).astype(int)
        all_errs.append(errs)

        recon_samples.append((X_noisy, X_clean, X_rec))

    all_errs_np = np.stack(all_errs)
    avg_error_per_char = all_errs_np.mean(axis=0)
    std_error_per_char = all_errs_np.std(axis=0)
    total_avg = avg_error_per_char.mean()
    total_std = avg_error_per_char.std()
    max_error = int(avg_error_per_char.max())

    print("\n>>> Evaluación PROMEDIO sobre múltiples corridas para noise level: ", noise_level)
    print(f"→ Promedio de bits incorrectos: {total_avg:.4f} ± {total_std:.4f}")
    print(f"→ Máximo error promedio (por char): {max_error}")
    print("→ Promedio de errores por carácter:", avg_error_per_char.round(3).tolist())

    plot_loss_avg(all_loss_hist, "Pérdida promedio (5 corridas)")

    idx_mejor = int(np.argmin(avg_error_per_char))
    idx_peor  = int(np.argmax(avg_error_per_char))

    Xn, Xc, Xr = recon_samples[0] 

    mostrar_reconstruccion(idx_mejor, Xn, Xc, Xr, "Mejor")
    mostrar_reconstruccion(idx_peor,  Xn, Xc, Xr, "Peor")

if __name__ == "__main__":
    main()