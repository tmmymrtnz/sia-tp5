# dae_analysis.py
#
# Este script realiza N corridas de un Denoising Autoencoder (DAE) para
# distintos niveles de ruido, y grafica:
#   - Línea del error promedio de bits con barras de min/max
#   - Línea del error máximo de bits con barras de min/max

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "src")

from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder import Autoencoder


def parse_font_h(path: str) -> np.ndarray:
    """
    Lee Font3[32][7] de `font.h` y devuelve un array NumPy (32,35) binario.
    """
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"Font3\s*\[\s*32\s*\]\s*\[\s*7\s*\]\s*=\s*\{(.*?)\};", text, re.DOTALL)
    if not m:
        raise ValueError("No se encontró Font3[32][7] en font.h")
    content = m.group(1)
    rows = re.findall(r"\{\s*(0x[0-9A-Fa-f]+(?:\s*,\s*0x[0-9A-Fa-f]+){6})\s*\}", content)
    if len(rows) != 32:
        raise ValueError(f"Se esperaban 32 filas, se encontraron {len(rows)}")
    data = np.zeros((32, 35), dtype=np.float32)
    for i, row in enumerate(rows):
        hexs = [h.strip() for h in row.split(',')]
        bits = []
        for h in hexs:
            v = int(h, 16)
            bits.extend([float(b) for b in format(v, '05b')])
        data[i] = bits
    return data


def add_binary_noise(X: np.ndarray, noise_level: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Flips each bit of X with probability `noise_level`.
    """
    mask = rng.rand(*X.shape) < noise_level
    Xn = X.copy()
    Xn[mask] = 1.0 - Xn[mask]
    return Xn


def main():
    parser = argparse.ArgumentParser(
        description="Analiza el desempeño de un DAE vs nivel de ruido"
    )
    parser.add_argument("config_path", help="Ruta al JSON de configuración")
    parser.add_argument("font_path", help="Ruta al archivo font.h")
    args = parser.parse_args()

    # Cargar configuración
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No se encontró config: {args.config_path}")
    cfg = json.load(open(args.config_path))

    # Cargar datos limpios
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"No se encontró font.h: {args.font_path}")
    X_clean = parse_font_h(args.font_path)

    noise_levels = cfg.get("noise_levels", [])
    num_runs     = cfg.get("num_runs", 10)

    # Preparar arrays para resultados
    mean_max_err = []
    min_max_err  = []
    max_max_err  = []
    mean_avg_err = []
    min_avg_err  = []
    max_avg_err  = []

    # Loop por nivel de ruido
    for nl in noise_levels:
        errs_max = []
        errs_avg = []
        print(f">>> Noise level: {nl}")
        for run in range(num_runs):
            seed = run + int(nl*100)
            rng  = np.random.RandomState(seed)
            X_noisy = add_binary_noise(X_clean, nl, rng)

            # Instanciar DAE según cfg["dae"]
            dae_cfg = cfg["dae"]
            ae = Autoencoder(
                encoder_sizes       = dae_cfg["encoder"]["layer_sizes"],
                encoder_activations = dae_cfg["encoder"]["activations"],
                decoder_sizes       = dae_cfg["decoder"]["layer_sizes"],
                decoder_activations = dae_cfg["decoder"]["activations"],
                dropout_rate        = dae_cfg.get("dropout_rate", 0.0)
            )
            trainer = Trainer(
                net             = ae,
                loss_name       = dae_cfg["loss"],
                optimizer_name  = dae_cfg["optimizer"],
                optim_kwargs    = dae_cfg.get("optim_kwargs", {}),
                batch_size      = dae_cfg["batch_size"],
                max_epochs      = dae_cfg["max_epochs"],
                log_every       = dae_cfg.get("log_every", num_runs+1),
                early_stopping  = True,
                patience        = dae_cfg.get("patience", 10),
                min_delta       = dae_cfg.get("min_delta", 1e-4)
            )
            ae.train_mode()
            trainer.fit(X_noisy, X_clean)
            ae.eval_mode()

            # Evaluar errores
            X_rec = ae.forward(X_noisy)
            X_bin = (X_rec > 0.5).astype(int)
            diffs = np.abs(X_bin - X_clean)
            errs = diffs.sum(axis=1)  # shape (32,)
            errs_max.append(int(errs.max()))
            errs_avg.append(float(errs.mean()))

        # Estadísticas
        mean_max_err.append(np.mean(errs_max))
        min_max_err.append(np.min(errs_max))
        max_max_err.append(np.max(errs_max))
        mean_avg_err.append(np.mean(errs_avg))
        min_avg_err.append(np.min(errs_avg))
        max_avg_err.append(np.max(errs_avg))

    # Graficar barras por separado

    indices = np.arange(len(noise_levels))
    width = 0.6

    # 1) Gráfico de Error Máximo
    plt.figure(figsize=(6,4))
    plt.bar(
        indices,
        mean_max_err,
        width,
        yerr=[np.array(mean_max_err)-np.array(min_max_err),
            np.array(max_max_err)-np.array(mean_max_err)],
        capsize=5,
        label="Max bits error",
        color="C0",
        alpha=0.8
    )
    plt.xticks(indices, noise_levels)
    plt.title("Max bits error vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Bits Error (max)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("dae_max_error_bars.png")
    plt.show()

    # 2) Gráfico de Error Promedio
    plt.figure(figsize=(6,4))
    plt.bar(
        indices,
        mean_avg_err,
        width,
        yerr=[np.array(mean_avg_err)-np.array(min_avg_err),
            np.array(max_avg_err)-np.array(mean_avg_err)],
        capsize=5,
        label="Avg bits error",
        color="C1",
        alpha=0.8
    )
    plt.xticks(indices, noise_levels)
    plt.title("Avg bits error vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Bits Error (avg)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("dae_avg_error_bars.png")
    plt.show()

if __name__ == "__main__":
    main()
