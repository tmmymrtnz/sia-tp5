#!/usr/bin/env python3
# filepath: /Users/saints/Desktop/ITBA/SIA/sia-tp5/src/analysis/vae_latent_space.py
# Genera proyecciones 2D del espacio latente para visualizar la estructura

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import json

# Importaciones locales
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from autoencoder.autoencoder_vae import VAE

def load_vae_model(checkpoint_dir, config_path):
    """Carga un modelo VAE desde un checkpoint."""
    # Cargar configuración
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Extraer información del directorio para ajustar el modelo
    dirname = os.path.basename(checkpoint_dir)
    parts = dirname.split('_')
    img_size = None
    
    for part in parts:
        if part.startswith('img'):
            try:
                img_size = int(part[3:])
                break
            except ValueError:
                pass
    
    if img_size is None:
        raise ValueError(f"No se pudo determinar img_size del directorio: {dirname}")
    
    # Ajustar tamaños de capas
    in_dim = img_size * img_size
    enc_sizes = cfg["encoder"]["layer_sizes"].copy()
    dec_sizes = cfg["decoder"]["layer_sizes"].copy()
    enc_sizes[0] = in_dim
    dec_sizes[-1] = in_dim
    
    # Instanciar VAE
    vae = VAE(
        encoder_sizes=enc_sizes,
        encoder_activations=cfg["encoder"]["activations"],
        decoder_sizes=dec_sizes,
        decoder_activations=cfg["decoder"]["activations"],
        latent_dim=cfg.get("latent_dim", enc_sizes[-1] // 2),
        dropout_rate=cfg.get("dropout_rate", 0.0)
    )
    
    # Cargar pesos
    weights_path = Path(checkpoint_dir) / "vae_weights.npz"
    vae.load_weights(str(weights_path))
    vae.eval_mode()
    
    return vae, img_size

def load_dataset(dataset_name, img_size, max_samples=1000):
    """Carga el dataset original utilizado para el entrenamiento."""
    # Importamos desde el script original para reutilizar funciones
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from runner_vae import load_emoji_dataset, download_openmoji, load_dsprites, load_lfw_tfds
    
    if dataset_name.startswith('faces_only'):
        emoji_dir = download_openmoji()
        return load_emoji_dataset(emoji_dir, size=img_size, max_imgs=max_samples, faces_only=True)
    
    elif dataset_name.startswith('emoji_full'):
        emoji_dir = download_openmoji()
        return load_emoji_dataset(emoji_dir, size=img_size, max_imgs=max_samples, faces_only=False)
    
    elif dataset_name.startswith('dsprites'):
        return load_dsprites(max_imgs=max_samples)
    
    elif dataset_name.startswith('faces_lfw'):
        try:
            return load_lfw_tfds(max_imgs=max_samples, img_px=img_size)
        except Exception as e:
            print(f"Error al cargar LFW: {e}")
            print("Generando datos aleatorios como alternativa...")
            return np.random.rand(100, img_size*img_size)
    
    else:
        print(f"Dataset {dataset_name} no reconocido. Generando datos aleatorios...")
        return np.random.rand(100, img_size*img_size)

def encode_dataset(vae, X, batch_size=32):
    """Codifica el dataset para obtener representaciones en espacio latente."""
    means = []
    logvars = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        # Obtener representación latente
        encoder_output = vae.encoder.forward(batch)
        
        # Separar medias y log-varianzas
        batch_means = encoder_output[:, :vae.latent_dim]
        batch_logvars = encoder_output[:, vae.latent_dim:]
        
        means.append(batch_means)
        logvars.append(batch_logvars)
    
    return np.vstack(means), np.vstack(logvars)

def visualize_latent_space_tsne(latent_vectors, output_path, title="Proyección t-SNE del Espacio Latente", perplexity=30):
    """Visualiza el espacio latente usando t-SNE."""
    print("Aplicando t-SNE para proyección 2D...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=2000, random_state=42)
    z_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], 
                        c=np.arange(len(z_2d)), cmap='viridis', 
                        alpha=0.6, s=10)
    plt.colorbar(scatter, label="Índice de muestra")
    plt.title(title, fontsize=14)
    plt.xlabel("Dimensión 1", fontsize=12)
    plt.ylabel("Dimensión 2", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return z_2d

def analyze_latent_distribution(latent_means, latent_logvars, output_path):
    """Analiza la distribución del espacio latente."""
    # Mostrar histogramas para algunas dimensiones
    num_dims = min(10, latent_means.shape[1])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_dims, len(axes))):
        ax = axes[i]
        # Histograma de medias
        ax.hist(latent_means[:, i], bins=30, density=True, alpha=0.7, color='blue')
        
        # Superponer distribución normal
        x = np.linspace(-3, 3, 100)
        ax.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'r--', linewidth=2)
        
        ax.set_title(f"Dim {i+1}")
        ax.set_xlim(-3, 3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analiza y visualiza el espacio latente de VAEs.")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Directorio del checkpoint a analizar")
    parser.add_argument("--config", type=str, required=True, 
                        help="Archivo de configuración JSON del modelo")
    parser.add_argument("--max_samples", type=int, default=1000, 
                        help="Número máximo de muestras para analizar")
    parser.add_argument("--perplexity", type=int, default=30, 
                        help="Perplexity para t-SNE")
    parser.add_argument("--output_dir", type=str, default="analysis_results", 
                        help="Directorio para guardar resultados")
    parser.add_argument("--no_dataset_load", action="store_true", 
                        help="No cargar dataset original (usar samples aleatorias)")
    args = parser.parse_args()

    # Crear directorio para resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_name = checkpoint_dir.name
    
    # Extraer información del experimento
    dataset_name = checkpoint_name.split('_')[0]
    
    print(f"Analizando espacio latente de {checkpoint_name}...")
    
    # Cargar modelo VAE
    vae, img_size = load_vae_model(checkpoint_dir, args.config)
    print(f"Modelo cargado con latent_dim={vae.latent_dim}, img_size={img_size}")
    
    # Cargar dataset o generar datos aleatorios
    if args.no_dataset_load:
        print("Generando datos aleatorios como entrada...")
        X = np.random.rand(min(args.max_samples, 500), img_size*img_size)
    else:
        print(f"Cargando dataset {dataset_name}...")
        try:
            X = load_dataset(dataset_name, img_size, args.max_samples)
            print(f"Dataset cargado: {X.shape}")
        except Exception as e:
            print(f"Error al cargar dataset: {e}")
            print("Continuando con datos aleatorios...")
            X = np.random.rand(min(args.max_samples, 500), img_size*img_size)
    
    # Codificar muestras al espacio latente
    print("Codificando muestras al espacio latente...")
    latent_means, latent_logvars = encode_dataset(vae, X)
    
    # Visualizar espacio latente con t-SNE
    print("Generando visualización t-SNE...")
    tsne_output_path = output_dir / f"latent_tsne_{checkpoint_name}.png"
    z_2d = visualize_latent_space_tsne(
        latent_means, 
        tsne_output_path,
        title=f"Proyección Latente: {dataset_name}, {img_size}x{img_size}",
        perplexity=args.perplexity
    )
    print(f"Visualización t-SNE guardada en {tsne_output_path}")
    
    # Analizar distribución del espacio latente
    print("Analizando distribución del espacio latente...")
    dist_output_path = output_dir / f"latent_dist_{checkpoint_name}.png"
    analyze_latent_distribution(latent_means, latent_logvars, dist_output_path)
    print(f"Análisis de distribución guardado en {dist_output_path}")
    
    # Guardar datos para uso futuro
    data_output_path = output_dir / f"latent_data_{checkpoint_name}.npz"
    np.savez(
        data_output_path,
        latent_means=latent_means,
        latent_logvars=latent_logvars,
        tsne_2d=z_2d
    )
    print(f"Datos guardados en {data_output_path}")
    
    print("Análisis del espacio latente completado.")

if __name__ == "__main__":
    main()