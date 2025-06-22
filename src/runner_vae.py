#!/usr/bin/env python3
# runner_vae.py
#
# Este script:
#  1) Descarga el dataset OpenMoji o lo carga si ya está descargado
#  2) Construye un VAE (Autoencoder Variacional) según la configuración JSON
#  3) Entrena el VAE usando una función de pérdida especial para VAE
#  4) Guarda los pesos entrenados y genera muestras

import json
import argparse
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import fnmatch, os, zipfile, io, requests, glob, shutil
from pathlib import Path

# Imports locales
from autoencoder.autoencoder_vae import VAE
from autoencoder.vaeloss import VAELoss
from common.perceptrons.multilayer.vae_trainer import VAETrainer

# --- at the top of runner_vae.py (or datasets/emoji_dataset.py) ---------------
FACE_MIN = 0x1F600   # 😀
FACE_MAX = 0x1F64F   # 🙏

def is_yellow_face(filename: str) -> bool:
    """
    Returns True if `…/1f600.png` ∈ [FACE_MIN, FACE_MAX].
    Works even for sequences like '1f979.png' (🥹).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]   # '1f600'
    try:
        cp = int(stem.split('-')[0], 16)  # take first code point in sequences
    except ValueError:
        return False
    return FACE_MIN <= cp <= FACE_MAX


def download_openmoji(target_dir="data/emojis", asset_res="618", refresh=False):
    """
    Downloads *either* the 72×72 or 618×618 colour PNG asset from the latest
    OpenMoji release.  Returns the path that contains the PNG files.
    """
    asset_name = (
        "openmoji-72x72-color.zip"  if asset_res == "72"
        else "openmoji-618x618-color.zip"
    )
    url = (
        f"https://github.com/hfg-gmuend/openmoji/releases/latest/download/{asset_name}"
    )
    extract_dir = Path(target_dir, "raw")
    png_dir     = None

    # -------- 0) Clean up if --refresh  ------------------------------------
    if refresh and extract_dir.exists():
        print(">>> --refresh_dataset: borrando dataset anterior…")
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1) Reuse if PNGs already there  ------------------------------
    for root, _, files in os.walk(extract_dir):
        if any(fnmatch.fnmatch(f, "*.png") for f in files):
            png_dir = root
            if not refresh:
                print(f">>> Dataset listo en {png_dir} (reuse)")
                return png_dir
            # else: we’ll download again below
            break

    # -------- 2) Download & extract  ---------------------------------------
    print(f">>> Descargando {asset_name} …")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    print(">>> Extrayendo archivos…")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(extract_dir)

    # -------- 3) Locate the PNG folder  ------------------------------------
    for root, _, files in os.walk(extract_dir):
        if any(fnmatch.fnmatch(f, "*.png") for f in files):
            png_dir = root
            break

    if png_dir is None:
        raise RuntimeError("No se encontró ningún PNG tras la extracción.")
    print(f">>> Dataset listo en {png_dir}")
    return png_dir

def load_emoji_dataset(emoji_dir, size=28, max_emojis=500):
    pattern = os.path.join(emoji_dir, "**", "*.png")
    all_files = sorted(glob.glob(pattern, recursive=True))
    face_files = [f for f in all_files if is_yellow_face(f)]

    if not face_files:
        raise RuntimeError("No yellow-face emojis found — check paths or ranges.")

    files = face_files[:max_emojis]           # keep at most N
    print(f">>> Usando {len(files)} yellow faces de {len(all_files)} PNGs totales")

    dataset = []
    for file in files:
        img = Image.open(file).convert("L").resize((size, size), Image.LANCZOS)
        dataset.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)

    return np.vstack(dataset)

def save_sample_images(images, shape=(28, 28), path="sample_emojis.png"):
    """
    Guarda una muestra de las imágenes del dataset como una imagen PNG.
    """
    
    n = min(25, len(images))
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < n:
                axes[i, j].imshow(images[idx].reshape(shape), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f">>> Muestra guardada en {path}")

def save_vae_samples(vae, n_samples=25, shape=(28, 28), path="vae_samples.png"):
    """
    Genera y guarda muestras del VAE entrenado
    """    
    # Generar muestras
    samples = vae.generate(n_samples=n_samples)
    
    # Visualizar
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < n_samples:
                axes[i, j].imshow(samples[idx].reshape(shape), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f">>> Muestras generadas guardadas en {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un VAE (Autoencoder Variacional) con emojis y configuración JSON."
    )
    parser.add_argument(
        "config_path",
        help="Ruta al archivo de configuración JSON (por ejemplo, config_vae.json)."
    )
    parser.add_argument(
        "--max_emojis", type=int, default=500,
        help="Número máximo de emojis a cargar (default: 500)"
    )
    parser.add_argument(
        "--img_size", type=int, default=28,
        help="Tamaño de las imágenes redimensionadas (default: 28)"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Factor beta para la divergencia KL en VAE (default: 1.0)"
    )
    parser.add_argument(
    "--asset_res", choices=["72", "618"], default="618",
    help="Which OpenMoji asset to download: 72-px (tiny) or 618-px (hi-res)."
    )
    parser.add_argument(
    "--refresh_dataset", action="store_true",
    help="Force re-download + re-extract the asset even if images already exist."
    )
    args = parser.parse_args()

    config_path = args.config_path
    max_emojis = args.max_emojis
    img_size = args.img_size
    beta = args.beta

    # Verificar que el archivo exista
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: '{config_path}'")

    # 1) Cargar configuración
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 2) Arquitectura del encoder/decoder
    enc_sizes = cfg["encoder"]["layer_sizes"]
    enc_acts = cfg["encoder"]["activations"]
    dec_sizes = cfg["decoder"]["layer_sizes"]
    dec_acts = cfg["decoder"]["activations"]
    latent_dim = cfg.get("latent_dim", enc_sizes[-1] // 2)  # Dimensión del espacio latente
    dropout_rate = cfg.get("dropout_rate", 0.0)

    # 3) Parámetros de entrenamiento
    reconstruction_loss = cfg.get("reconstruction_loss", "mse")  # mse o bce
    optimizer_name = cfg["optimizer"]
    optim_kwargs = cfg.get("optim_kwargs", {})
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    log_every = cfg.get("log_every", 100)
    patience = cfg.get("patience", 10)
    min_delta = cfg.get("min_delta", 1e-4)

    # 4) Descargar y cargar el dataset
    emoji_dir = download_openmoji(asset_res=args.asset_res, refresh=args.refresh_dataset)
    X = load_emoji_dataset(emoji_dir, size=img_size, max_emojis=max_emojis)
    
    # Verificar si X está vacío
    if len(X) == 0:
        print(">>> No se pudieron cargar imágenes. Generando datos aleatorios...")
        X = np.random.rand(50, img_size*img_size)  # 50 imágenes aleatorias
    
    print(f">>> Dataset cargado: {X.shape[0]} imágenes de {X.shape[1]} elementos")
    
    # Guardar muestra de imágenes
    save_sample_images(X, shape=(img_size, img_size), path="data/sample_emojis.png")

    # 5) Preparar Y = X (autoencoder)
    Y = X.copy()

    # 6) Instanciar el VAE
    print(">>> Construyendo VAE...")
    vae = VAE(
        encoder_sizes=enc_sizes,
        encoder_activations=enc_acts,
        decoder_sizes=dec_sizes,
        decoder_activations=dec_acts,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate
    )

    # 7) Crear y configurar la función de pérdida VAE
    vae_loss = VAELoss(reconstruction_loss_name=reconstruction_loss, beta=beta)
    vae_loss.vae_model = vae  # Conectar el modelo con la función de pérdida

    # 8) Instanciar el Trainer con pérdida personalizada
    print(">>> Configurando entrenador...")
    trainer = VAETrainer(
        vae=vae,
        loss_name=reconstruction_loss,
        beta=beta,
        optimizer_name=optimizer_name,
        optim_kwargs=optim_kwargs,
        batch_size=batch_size,
        max_epochs=max_epochs,
        log_every=log_every,
        early_stopping=True,
        patience=patience,
        min_delta=min_delta
    )

    # 9) Entrenamiento
    print(f">>> Entrenando VAE (X → X) con beta={beta}...")
    vae.train_mode()
    loss_history = trainer.fit(X, Y)

    # 10) Guardar pesos y pérdida
    os.makedirs("checkpoints", exist_ok=True)
    w_path = "checkpoints/vae_weights.npz"
    print(f">>> Guardando pesos en '{w_path}'...")
    vae.save_weights(w_path)

    loss_path = "checkpoints/vae_loss_history.npy"
    np.save(loss_path, np.array(loss_history, dtype=np.float32))
    print(f">>> Historial de pérdida guardado en '{loss_path}'.")

    # 11) Generar muestras con el VAE entrenado
    print(">>> Generando nuevas muestras con el VAE...")
    vae.eval_mode()
    save_vae_samples(vae, n_samples=25, shape=(img_size, img_size), path="checkpoints/vae_samples.png")

    print(">>> Entrenamiento VAE completado.")

if __name__ == "__main__":
    main()