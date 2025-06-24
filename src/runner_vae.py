#!/usr/bin/env python3
# runner_vae.py — Entrena un VAE sobre datasets ligeros (OpenMoji, dSprites,
#                 FFHQ-128 thumbnails y LFW vía TensorFlow-Datasets).

import argparse, json, os, io, glob, shutil, zipfile, urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image

from autoencoder.autoencoder_vae import VAE
from common.perceptrons.multilayer.vae_trainer import VAETrainer
import tensorflow_datasets as tfds, tensorflow as tf


# ───────────────────────── Smileys filter (OpenMoji) ────────────────────────
FACE_RANGE = (0x1F600, 0x1F64F)  # 😀 … 🙏

def is_yellow_face(fname: str) -> bool:
    stem = os.path.splitext(os.path.basename(fname))[0]
    try:
        cp = int(stem.split('-')[0], 16)
    except ValueError:
        return False
    lo, hi = FACE_RANGE
    return lo <= cp <= hi

# ─────────────────────────── OpenMoji download/load ─────────────────────────
def download_openmoji(target="data/emojis", res="618", refresh=False) -> Path:
    asset = "openmoji-72x72-color.zip" if res == "72" else "openmoji-618x618-color.zip"
    url   = f"https://github.com/hfg-gmuend/openmoji/releases/latest/download/{asset}"
    root  = Path(target, "raw")

    if refresh and root.exists():
        print(">>> --refresh_dataset: borrando dataset anterior…")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    # Re-usar si ya hay PNGs
    for r, _, f in os.walk(root):
        if any(fn.endswith(".png") for fn in f):
            print(f">>> Dataset listo en {r} (reuse)")
            return Path(r)

    print(f">>> Descargando {asset} …")
    buf = requests.get(url, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        zf.extractall(root)

    for r, _, f in os.walk(root):
        if any(fn.endswith(".png") for fn in f):
            print(f">>> Dataset listo en {r}")
            return Path(r)
    raise RuntimeError("OpenMoji: no se encontraron PNG tras la extracción.")

def load_emoji_dataset(emoji_dir: Path, size=32, max_imgs=500, faces_only=True):
    files = sorted(glob.glob(str(emoji_dir / "**/*.png"), recursive=True))
    if faces_only:
        files = [f for f in files if is_yellow_face(f)]
    if not files:
        raise RuntimeError("No PNGs encontrados con el filtro actual.")
    files = files[:max_imgs]
    print(f">>> Usando {len(files)} PNGs ({'solo caras' if faces_only else 'todos'})")

    data = []
    for f in files:
        img = Image.open(f).convert("L").resize((size, size), Image.LANCZOS)
        data.append(np.asarray(img, np.float32).flatten() / 255.0)
    return np.vstack(data)

# ───────────────────────────── dSprites (64×64) ─────────────────────────────
def load_dsprites(max_imgs=500):
    url  = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    path = Path("data/dsprites.npz")
    if not path.exists():
        print(">>> Descargando dSprites…")
        urllib.request.urlretrieve(url, path)
    imgs = np.load(path)["imgs"][:max_imgs]               # (N, 64, 64)
    return imgs.astype("float32").reshape(len(imgs), -1)

# ─────────────────────── LFW-deepfunneled vía TF-Datasets ───────────────────
def load_lfw_tfds(max_imgs=8_000, img_px=64):
    print(">>> Cargando LFW-deepfunneled con TF-Datasets… (auto-download/cache)")
    ds = tfds.load("lfw", split="train", shuffle_files=False, as_supervised=False)

    ds = (
        ds.map(
            lambda x: tf.image.rgb_to_grayscale(
                tf.image.resize(x["image"], (img_px, img_px))
            ) / 255.0,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .take(max_imgs)
    )

    # tfds.as_numpy convierte cada elemento del pipeline a ndarray
    return np.stack([img.reshape(-1) for img in tfds.as_numpy(ds)])


# ────────────────────────────── helper grilla ───────────────────────────────
def save_grid(vectors, shape, out="grid.png", upscale=4):
    n, (h, w) = min(25, len(vectors)), shape
    big = (w*upscale, h*upscale)
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(25):
        a = ax[i//5, i%5]; a.axis("off")
        if i < n:
            arr = (vectors[i].reshape(shape)*255).astype("uint8")
            img = Image.fromarray(arr, mode="L").resize(big, Image.LANCZOS)
            a.imshow(img, cmap="gray")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f">>> Grilla guardada en {out}")

# ────────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--dataset",
                    choices=["faces_only", "emoji_full",
                             "dsprites", "faces_ffhq", "faces_lfw"],
                    default="faces_only")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--max_emojis", type=int, default=500)
    ap.add_argument("--asset_res", choices=["72", "618"], default="618")
    ap.add_argument("--refresh_dataset", action="store_true")
    ap.add_argument("--beta_final", type=float, default=1.0)
    ap.add_argument("--kl_ramp_epochs", type=int, default=0)
    args = ap.parse_args()

    # ────────────── carga / pre-procesado del dataset elegido ──────────────
    if args.dataset in ("faces_only", "emoji_full"):
        emoji_dir = download_openmoji(res=args.asset_res, refresh=args.refresh_dataset)
        X = load_emoji_dataset(emoji_dir, size=args.img_size,
                               max_imgs=args.max_emojis,
                               faces_only=(args.dataset == "faces_only"))

    elif args.dataset == "dsprites":
        X = load_dsprites(max_imgs=args.max_emojis)
        args.img_size = 64

    elif args.dataset == "faces_lfw":
        X = load_lfw_tfds(max_imgs=args.max_emojis)
        args.img_size = 64

    else:
        raise ValueError("Dataset no soportado.")

    print(f">>> Dataset cargado: {X.shape}")

    # Muestra de entrada
    sample_path = Path("data", f"sample_{args.dataset}.png")
    save_grid(X, shape=(args.img_size, args.img_size), out=sample_path)

    # ─────────────── leer configuración JSON del modelo ────────────────────
    with open(args.config_path) as f:
        cfg = json.load(f)

    in_dim = args.img_size * args.img_size
    enc_sizes = cfg["encoder"]["layer_sizes"];  enc_sizes[0]  = in_dim
    dec_sizes = cfg["decoder"]["layer_sizes"];  dec_sizes[-1] = in_dim

    vae = VAE(
        encoder_sizes   = enc_sizes,
        encoder_activations = cfg["encoder"]["activations"],
        decoder_sizes   = dec_sizes,
        decoder_activations = cfg["decoder"]["activations"],
        latent_dim      = cfg.get("latent_dim", enc_sizes[-1] // 2),
        dropout_rate    = cfg.get("dropout_rate", 0.0)
    )

    trainer = VAETrainer(
        vae            = vae,
        loss_name      = cfg.get("reconstruction_loss", "mse"),
        beta           = args.beta_final,
        kl_ramp_epochs = args.kl_ramp_epochs,
        optimizer_name = cfg["optimizer"],
        optim_kwargs   = cfg.get("optim_kwargs", {}),
        batch_size     = cfg["batch_size"],
        max_epochs     = cfg["max_epochs"],
        log_every      = cfg.get("log_every", 100),
        early_stopping = True,
        patience       = cfg.get("patience", 10),
        min_delta      = cfg.get("min_delta", 1e-4)
    )

    # ─────────────────────────── entrenamiento ─────────────────────────────
    print(f">>> Entrenando VAE con β={args.beta_final}…")
    vae.train_mode()
    history = trainer.fit(X, X)

    # ────────────────────────── salvado resultados ─────────────────────────
    tag     = f"{args.dataset}_img{args.img_size}_beta{args.beta_final}"
    out_dir = Path("checkpoints", f"{tag}_{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    vae.save_weights(out_dir / "vae_weights.npz")
    np.save(out_dir / "loss_history.npy", np.array(history, "float32"))

    vae.eval_mode()
    gen = vae.generate(25)
    save_grid(gen, shape=(args.img_size, args.img_size),
              out=out_dir / "vae_generated.png")

    print(">>> Entrenamiento VAE completado.")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
