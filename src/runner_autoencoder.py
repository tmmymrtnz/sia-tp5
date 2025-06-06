#!/usr/bin/env python3
# runner.py
#
# Este script:
#  1) Parsea `font.h` para extraer los 32 patrones de 7×5 bits.
#  2) Construye un Autoencoder según la configuración en un archivo JSON.
#  3) Entrena el Autoencoder (X→X) usando Trainer.
#  4) Guarda los pesos entrenados en `checkpoints/ae_weights.npz`.

import os
import re
import json
import argparse
import numpy as np

# Ajusta estas importaciones según la ubicación de tu paquete:
# - `network.MLP` es la clase de red multicapa (en tu network.py).
# - `trainer.Trainer` es la clase de entrenamiento (en tu trainer.py).
# - `autoencoder.Autoencoder` es la clase de AE (en el archivo autoencoder.py).
from common.perceptrons.multilayer.network import MLP
from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder import Autoencoder

def parse_font_h(path: str) -> np.ndarray:
    """
    Lee el archivo C `Font3[32][7]` y devuelve un array NumPy de forma (32,35),
    cada fila es la concatenación de los 7 renglones de 5 bits (en orden fila-mayor).
    
    Asume que cada renglón se expresa con un entero hexadecimal (0x00..0x1F).
    Ejemplo de bloque en font.h:
       {0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00},   // caracter
    """
    with open(path, "r") as f:
        text = f.read()

    # Buscar el bloque Font3[32][7] = { { ... }, { ... }, ... };
    bloque_match = re.search(
        r"Font3\s*\[\s*32\s*\]\s*\[\s*7\s*\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL
    )
    if not bloque_match:
        raise ValueError("No se encontró el bloque `Font3[32][7]` en el archivo.")

    contenido = bloque_match.group(1)
    filas = re.findall(r"\{\s*(0x[0-9A-Fa-f]+(?:\s*,\s*0x[0-9A-Fa-f]+){6})\s*\}", contenido)
    if len(filas) != 32:
        raise ValueError(f"Se esperaban 32 filas; se encontraron {len(filas)}.")

    datos = np.zeros((32, 35), dtype=np.float32)

    for i, fila_str in enumerate(filas):
        hex_vals = [s.strip() for s in fila_str.split(",")]
        if len(hex_vals) != 7:
            raise ValueError(f"Fila {i} no tiene exactamente 7 valores: {hex_vals}")

        bits_35 = []
        for h in hex_vals:
            val = int(h, 16)
            bin5 = format(val, "05b")
            bits_35.extend([float(b) for b in bin5])

        if len(bits_35) != 35:
            raise RuntimeError(f"Interno: no se generaron 35 bits en la fila {i}.")
        datos[i, :] = np.array(bits_35, dtype=np.float32)

    return datos

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un Autoencoder con datos de font.h y configuración JSON."
    )
    parser.add_argument(
        "config_path",
        help="Ruta al archivo de configuración JSON (por ejemplo, config.json)."
    )
    parser.add_argument(
        "font_path",
        help="Ruta al archivo font.h que contiene `Font3[32][7]`."
    )
    args = parser.parse_args()

    config_path = args.config_path
    font_path = args.font_path

    # Verificar que los archivos existan
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: '{config_path}'")
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"No se encontró el archivo font.h: '{font_path}'")

    # 1) Cargar configuración
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Arquitectura del encoder/decoder
    enc_sizes       = cfg["encoder"]["layer_sizes"]
    enc_acts        = cfg["encoder"]["activations"]
    dec_sizes       = cfg["decoder"]["layer_sizes"]
    dec_acts        = cfg["decoder"]["activations"]
    dropout_rate    = cfg.get("dropout_rate", 0.0)

    # Parámetros de entrenamiento
    loss_name       = cfg["loss"]
    optimizer_name  = cfg["optimizer"]
    optim_kwargs    = cfg.get("optim_kwargs", {})
    batch_size      = cfg["batch_size"]
    max_epochs      = cfg["max_epochs"]
    log_every       = cfg.get("log_every", 100)
    patience        = cfg.get("patience", 10)
    min_delta       = cfg.get("min_delta", 1e-4)

    # 2) Parsear font.h → X (shape (32,35))
    print(f">>> Parseando '{font_path}' …")
    X = parse_font_h(font_path)

    # 3) Preparar Y = X (autoencoder)
    Y = X.copy()

    # 4) Instanciar el Autoencoder
    print(">>> Construyendo Autoencoder …")
    ae = Autoencoder(
        encoder_sizes       = enc_sizes,
        encoder_activations = enc_acts,
        decoder_sizes       = dec_sizes,
        decoder_activations = dec_acts,
        dropout_rate        = dropout_rate
    )

    # 5) Instanciar el Trainer
    print(">>> Configurando entrenador …")
    trainer = Trainer(
        net             = ae,
        loss_name       = loss_name,
        optimizer_name  = optimizer_name,
        optim_kwargs    = optim_kwargs,
        batch_size      = batch_size,
        max_epochs      = max_epochs,
        log_every       = log_every,
        early_stopping  = True,
        patience        = patience,
        min_delta       = min_delta
    )

    # 6) Entrenamiento
    print(">>> Entrenando Autoencoder (X → X) …")
    ae.train_mode()
    loss_history = trainer.fit(X, Y)

    # 7) Guardar pesos y pérdida
    os.makedirs("checkpoints", exist_ok=True)
    w_path = "checkpoints/ae_weights.npz"
    print(f">>> Guardando pesos en '{w_path}' …")
    ae.save_weights(w_path)

    loss_path = "checkpoints/loss_history.npy"
    np.save(loss_path, np.array(loss_history, dtype=np.float32))
    print(f">>> Historial de pérdida guardado en '{loss_path}'.")

    print(">>> Entrenamiento completado.")

if __name__ == "__main__":
    main()
