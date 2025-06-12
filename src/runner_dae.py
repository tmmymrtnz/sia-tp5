#!/usr/bin/env python3
# runner_dae.py
#
# Este script:
#  1) Parsea `font.h` para extraer los 32 patrones de 7×5 bits.
#  2) Crea versiones ruidosas de esos patrones según un nivel de ruido.
#  3) Construye un Autoencoder según la configuración en un JSON.
#  4) Entrena el Autoencoder (X_noisy → X_clean) usando Trainer.
#  5) Guarda los pesos entrenados en `checkpoints/dae_weights.npz` y muestra métricas de desruido.

import os
import re
import json
import argparse
import numpy as np

from common.perceptrons.multilayer.network import MLP
from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder import Autoencoder

def parse_font_h(path: str) -> np.ndarray:
    """Idéntico a runner.py: lee Font3[32][7] y devuelve array (32,35) binario."""
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"Font3\s*\[\s*32\s*\]\s*\[\s*7\s*\]\s*=\s*\{(.*?)\};",
                  text, re.DOTALL)
    if not m:
        raise ValueError("No se encontró Font3[32][7].")
    content = m.group(1)
    rows = re.findall(r"\{\s*(0x[0-9A-Fa-f]+(?:\s*,\s*0x[0-9A-Fa-f]+){6})\s*\}", content)
    if len(rows) != 32:
        raise ValueError(f"Se encontraron {len(rows)} filas; se esperaban 32.")
    data = np.zeros((32, 35), dtype=np.float32)
    for i, row in enumerate(rows):
        hexs = [h.strip() for h in row.split(",")]
        bits = []
        for h in hexs:
            v = int(h, 16)
            bits.extend([float(b) for b in format(v, "05b")])
        data[i] = bits
    return data

def add_binary_noise(X: np.ndarray, noise_level: float, seed: int = 0) -> np.ndarray:
    """
    Flupea cada bit con probabilidad noise_level.
    """
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) < noise_level
    Xn = X.copy()
    Xn[mask] = 1.0 - Xn[mask]
    return Xn

def main():
    p = argparse.ArgumentParser(description="Entrena un Denoising AE según JSON de configuración.")
    p.add_argument("config_path", help="Ruta al JSON de configuración (config_dae.json).")
    p.add_argument("font_path", help="Ruta al archivo font.h con Font3[32][7].")
    args = p.parse_args()

    # 1) cargar config
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No se encontró config: {args.config_path}")
    cfg = json.load(open(args.config_path))

    # 2) cargar y parsear font.h
    if not os.path.exists(args.font_path):
        raise FileNotFoundError(f"No se encontró font.h: {args.font_path}")
    print(f">>> Parseando '{args.font_path}' …")
    X_clean = parse_font_h(args.font_path)

    # 3) crear X_noisy
    noise_level = cfg.get("noise_level", 0.1)
    print(f">>> Añadiendo ruido binario (level={noise_level}) …")
    X_noisy = add_binary_noise(X_clean, noise_level, seed=0)

    # 4) construir Autoencoder
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    print(">>> Construyendo Denoising Autoencoder …")
    ae = Autoencoder(
        encoder_sizes       = enc["layer_sizes"],
        encoder_activations = enc["activations"],
        decoder_sizes       = dec["layer_sizes"],
        decoder_activations = dec["activations"],
        dropout_rate        = cfg.get("dropout_rate", 0.0)
    )

    # 5) configurar Trainer
    print(">>> Configurando entrenador …")
    trainer = Trainer(
        net             = ae,
        loss_name       = cfg["loss"],
        optimizer_name  = cfg["optimizer"],
        optim_kwargs    = cfg.get("optim_kwargs", {}),
        batch_size      = cfg["batch_size"],
        max_epochs      = cfg["max_epochs"],
        log_every       = cfg.get("log_every", 100),
        early_stopping  = True,
        patience        = cfg.get("patience", 10),
        min_delta       = cfg.get("min_delta", 1e-4)
    )

    # 6) entrenar
    print(">>> Entrenando DAE (X_noisy → X_clean) …")
    ae.train_mode()
    loss_hist = trainer.fit(X_noisy, X_clean)

    # 7) guardar pesos
    os.makedirs("checkpoints", exist_ok=True)
    wpath = "checkpoints/dae_weights.npz"
    print(f">>> Guardando pesos en '{wpath}' …")
    ae.save_weights(wpath)

    # 8) evaluar desruido
    ae.eval_mode()
    X_rec = ae.forward(X_noisy)
    X_bin = (X_rec > 0.5).astype(int)
    diffs = np.abs(X_bin - X_clean)
    errs = diffs.sum(axis=1).astype(int)
    print(">>> Evaluación de desruido")
    print("Avg bits error:", float(errs.mean()))
    print("Max bits error:", int(errs.max()))
    print("Errors per char:", errs.tolist())

    # 9) guardar historial de pérdida
    np.save("checkpoints/dae_loss_history.npy", np.array(loss_hist))
    print(">>> Historial de pérdida guardado en 'checkpoints/dae_loss_history.npy'.")

if __name__ == "__main__":
    main()
