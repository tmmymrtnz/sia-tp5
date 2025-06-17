#!/usr/bin/env python3
# runner_vae.py
#
# Este script:
#  1) Descarga el dataset OpenMoji o lo carga si ya está descargado
#  2) Construye un VAE (Autoencoder Variacional) según la configuración JSON
#  3) Entrena el VAE usando una función de pérdida especial para VAE
#  4) Guarda los pesos entrenados y genera muestras

import os
import json
import argparse
import numpy as np
import requests
import zipfile
import io
from PIL import Image

# Imports locales
from common.perceptrons.multilayer.trainer import Trainer
from autoencoder.autoencoder_vae import VAE
from autoencoder.vaeloss import VAELoss
from common.perceptrons.multilayer.vae_trainer import VAETrainer

def download_openmoji(target_dir="data/emojis"):
    """
    Descarga el dataset de OpenMoji y lo prepara para entrenar el VAE.
    """
    # URL del dataset de OpenMoji en formato PNG
    url = "https://github.com/hfg-gmuend/openmoji/releases/download/14.0.0/openmoji-png-72x72.zip"
    
    # Crear directorio para los datos
    os.makedirs(target_dir, exist_ok=True)
    
    emoji_dir = os.path.join(target_dir, "raw/72x72")
    
    # Si ya existe el directorio con los archivos, no descargar de nuevo
    if os.path.exists(emoji_dir) and len(os.listdir(emoji_dir)) > 0:
        print(f">>> Dataset ya descargado en {emoji_dir}")
        return emoji_dir
    
    # Intentar descargar el archivo
    print(">>> Descargando OpenMoji dataset...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Error HTTP: {response.status_code}")
        
        # Extraer el archivo ZIP
        print(">>> Extrayendo archivos...")
        extract_dir = os.path.join(target_dir, "raw")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f">>> Dataset extraído en {emoji_dir}")
        return emoji_dir
    
    except Exception as e:
        print(f">>> Error al descargar el dataset: {e}")
        print("\n>>> INSTRUCCIONES PARA DESCARGA MANUAL:")
        print(f">>> 1. Descarga el archivo ZIP desde: {url}")
        print(f">>> 2. Crea la carpeta: {os.path.join(target_dir, 'raw')}")
        print(f">>> 3. Extrae el contenido del ZIP en esa carpeta")
        print(f">>> 4. Ejecuta nuevamente el script\n")
        
        # Preguntar al usuario si desea continuar con datos aleatorios
        use_random = input(">>> ¿Deseas continuar con datos aleatorios para probar? (s/n): ")
        if use_random.lower().startswith('s'):
            print(">>> Generando datos aleatorios de prueba...")
            os.makedirs(emoji_dir, exist_ok=True)
            
            # Generar imágenes aleatorias para prueba
            for i in range(100):  # Crear 100 imágenes aleatorias
                img_array = np.random.rand(72, 72) * 255  # Valores entre 0-255
                img = Image.fromarray(img_array.astype('uint8'))
                img.save(os.path.join(emoji_dir, f"random_{i:03d}.png"))
            
            print(f">>> Se generaron 100 imágenes aleatorias en {emoji_dir}")
            return emoji_dir
        else:
            raise Exception("Descarga fallida. Por favor intenta la descarga manual.")

def load_emoji_dataset(emoji_dir, size=28, max_emojis=500):
    """
    Carga los emojis, los redimensiona y los convierte en arrays NumPy.
    
    Args:
        emoji_dir: Directorio con los archivos PNG
        size: Tamaño final de las imágenes (size x size)
        max_emojis: Número máximo de emojis a cargar
        
    Returns:
        np.ndarray: Dataset de emojis normalizado [0,1]
    """
    print(f">>> Cargando hasta {max_emojis} emojis de {emoji_dir}...")
    
    # Listar todos los archivos PNG
    if not os.path.exists(emoji_dir):
        raise FileNotFoundError(f"Directorio no encontrado: {emoji_dir}")
        
    files = [f for f in os.listdir(emoji_dir) if f.endswith('.png')]
    if not files:
        print(f">>> ADVERTENCIA: No se encontraron archivos PNG en {emoji_dir}")
        # Generar algunos datos aleatorios para evitar errores
        dataset = np.random.rand(50, size*size)  # 50 imágenes aleatorias
        return dataset
    
    files = files[:max_emojis]  # Limitar la cantidad
    
    # Cargar y preprocesar imágenes
    dataset = []
    for file in files:
        img_path = os.path.join(emoji_dir, file)
        try:
            # Abrir imagen
            img = Image.open(img_path).convert('L')  # Convertir a escala de grises
            
            # Redimensionar
            img = img.resize((size, size), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # Convertir a array y normalizar
            img_array = np.array(img) / 255.0
            
            dataset.append(img_array.flatten())
        except Exception as e:
            print(f"Error procesando {file}: {e}")
    
    return np.array(dataset)

def save_sample_images(images, shape=(28, 28), path="sample_emojis.png"):
    """
    Guarda una muestra de las imágenes del dataset como una imagen PNG.
    """
    import matplotlib.pyplot as plt
    
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
    import matplotlib.pyplot as plt
    
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
    emoji_dir = download_openmoji()
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