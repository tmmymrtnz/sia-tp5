#!/usr/bin/env python3
# src/experimentos_archs.py
#
# Este script lee un JSON con varias configuraciones (Experiment A, B, C, …),
# entrena un Autoencoder con cada una y guarda resultados en disco.

import json
import numpy as np
import os
import argparse
import sys

sys.path.insert(0, "src")

from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder import Autoencoder
from runner_autoencoder import parse_font_h  # reutilizamos la función de parsing

def entrenar_evaluar(exp_id, conf_exp, global_conf, X):
    """
    Entrena un Autoencoder según conf_exp (la subconfig del experimento exp_id),
    usando parámetros comunes en global_conf. Devuelve:
      (pérdida_final, errores_por_char (array de tamaño 32), epochs_totales)
    """
    # → fijar semilla para inicializar pesos idénticos en cada experimento
    np.random.seed(0)

    # 1) Construir el Autoencoder con pesos deterministas
    ae = Autoencoder(
        encoder_sizes       = conf_exp["encoder"]["layer_sizes"],
        encoder_activations = conf_exp["encoder"]["activations"],
        decoder_sizes       = conf_exp["decoder"]["layer_sizes"],
        decoder_activations = conf_exp["decoder"]["activations"],
        dropout_rate        = conf_exp.get("dropout_rate", 0.0)
    )

    # Resto inalterado…
    trainer = Trainer(
        net             = ae,
        loss_name       = conf_exp["loss"],
        optimizer_name  = conf_exp["optimizer"],
        optim_kwargs    = {"learning_rate": conf_exp["lr"]},
        batch_size      = global_conf["batch_size"],
        max_epochs      = global_conf["max_epochs"],
        log_every       = global_conf["log_every"],
        early_stopping  = True,
        patience        = global_conf["patience"],
        min_delta       = global_conf["min_delta"]
    )

    ae.train_mode()
    loss_hist = trainer.fit(X, X)
    epochs_tot = len(loss_hist)

    # Guardar pesos del experimento
    os.makedirs("checkpoints", exist_ok=True)
    weight_path = f"checkpoints/ae_weights_{exp_id}.npz"
    ae.save_weights(weight_path)
    print(f">>> Pesos guardados en: {weight_path}")

    ae.eval_mode()
    X_recon = ae.forward(X)
    X_bin   = (X_recon > 0.5).astype(int)
    diffs   = np.abs(X_bin - X)
    errores = diffs.sum(axis=1).astype(int)

    perdida_final = loss_hist[-1]
    return perdida_final, errores, epochs_tot


def main():
    parser = argparse.ArgumentParser(
        description="Prueba múltiples arquitecturas de Autoencoder según un JSON de configuración."
    )
    parser.add_argument(
        "config_path",
        help="Ruta al JSON de configuración (p. ej. configs/config_experiments.json)."
    )
    args = parser.parse_args()

    config_path = args.config_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: '{config_path}'")

    # 1) Leer configuración global
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 2) Parsear font.h
    font_h_path = cfg["font_h_path"]
    if not os.path.exists(font_h_path):
        raise FileNotFoundError(f"No se encontró el archivo font.h: '{font_h_path}'")
    print(f">>> Parseando '{font_h_path}' …")
    X = parse_font_h(font_h_path)  # (32,35)

    # 3) Crear carpeta para resultados
    os.makedirs("experimentos", exist_ok=True)
    resultados = {}

    # 4) Recorrer cada experimento definido en el JSON
    for exp_id, conf_exp in cfg["experiments"].items():
        print(f"\n>>> Ejecutando experimento {exp_id} ...")
        loss_final, errores_por_char, epocas = entrenar_evaluar(exp_id, conf_exp, cfg, X)

        max_error = int(errores_por_char.max())
        avg_error = float(errores_por_char.mean())

        resultados[exp_id] = {
            "loss_final": loss_final,
            "epochs": epocas,
            "max_error_bits": max_error,
            "avg_error_bits": avg_error,
            "errores_por_char": errores_por_char.tolist()
        }

        # Guardar resultados parciales en JSON
        with open(f"experimentos/resultados_{exp_id}.json", "w") as f:
            json.dump(resultados[exp_id], f, indent=2)

        print(f"   → Experimento {exp_id}: loss={loss_final:.6f}, epochs={epocas}, "
              f"max_error_bits={max_error}, avg_error_bits={avg_error:.2f}")

    # 5) Guardar todos los resultados juntos
    with open("experimentos/todos_resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print("\n>>> Todos los experimentos completados. Resultados en carpeta 'experimentos/'.")


if __name__ == "__main__":
    main()