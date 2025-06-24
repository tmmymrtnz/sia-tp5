#!/usr/bin/env python3
# filepath: /Users/saints/Desktop/ITBA/SIA/sia-tp5/src/analysis/vae_interpolation.py
# Genera interpolaciones en el espacio latente para visualizar transiciones suaves

import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import re

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

def extract_beta(dirname):
    """Extrae el valor de beta del nombre del directorio."""
    match = re.search(r'beta(\d+(?:\.\d+)?)', dirname)
    if match:
        return float(match.group(1))
    return None

def linear_interpolation(z1, z2, steps=10):
    """Interpola linealmente entre dos puntos en el espacio latente."""
    return np.array([z1 * (1-t) + z2 * t for t in np.linspace(0, 1, steps)])

def spherical_interpolation(z1, z2, steps=10):
    """Interpola esféricamente entre dos puntos en el espacio latente."""
    # Normalizar vectores
    z1_norm = z1 / np.linalg.norm(z1)
    z2_norm = z2 / np.linalg.norm(z2)
    
    # Calcular el ángulo entre los vectores
    omega = np.arccos(np.clip(np.dot(z1_norm, z2_norm), -1.0, 1.0))
    sin_omega = np.sin(omega)
    
    if np.abs(sin_omega) < 1e-6:  # Si los vectores son casi paralelos
        return linear_interpolation(z1, z2, steps)
    
    # Realizar interpolación esférica
    t_values = np.linspace(0, 1, steps)
    result = []
    
    for t in t_values:
        s1 = np.sin((1-t) * omega) / sin_omega
        s2 = np.sin(t * omega) / sin_omega
        interp = s1 * z1 + s2 * z2
        result.append(interp)
    
    return np.array(result)

def generate_random_latent_points(latent_dim, n_points=2):
    """Genera puntos aleatorios en el espacio latente (distribución normal)."""
    return np.random.randn(n_points, latent_dim)

def create_interpolation_grid(vae, img_size, output_path, n_pairs=5, steps=10, method='linear'):
    """Crea una cuadrícula con interpolaciones entre pares de puntos aleatorios."""
    # Generar pares de puntos aleatorios
    points = generate_random_latent_points(vae.latent_dim, n_pairs*2)
    
    # Configurar figura
    fig = plt.figure(figsize=(12, n_pairs * 2))
    
    for i in range(n_pairs):
        # Obtener par de puntos
        z1 = points[i*2]
        z2 = points[i*2+1]
        
        # Realizar interpolación
        if method == 'spherical':
            interp_points = spherical_interpolation(z1, z2, steps)
        else:
            interp_points = linear_interpolation(z1, z2, steps)
        
        # Decodificar puntos interpolados
        interp_images = vae.decoder.forward(interp_points)
        
        # Mostrar imágenes en la cuadrícula
        for j in range(steps):
            plt.subplot(n_pairs, steps, i * steps + j + 1)
            plt.imshow(interp_images[j].reshape(img_size, img_size), cmap='gray')
            plt.axis('off')
    
    plt.suptitle(f"Interpolaciones {method}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_latent_traversals(vae, img_size, output_path, n_dims=5, steps=11):
    """
    Genera visualización de traversals para dimensiones específicas del espacio latente.
    Muestra cómo varía la generación al modificar cada dimensión individual.
    """
    # Limitar a las primeras n_dims dimensiones, o menos si el espacio latente es menor
    n_dims = min(n_dims, vae.latent_dim)
    
    # Crear figura
    fig = plt.figure(figsize=(12, 2*n_dims))
    
    # Para cada dimensión
    for dim in range(n_dims):
        # Crear vector base (ceros)
        z_base = np.zeros((1, vae.latent_dim))
        
        # Variar la dimensión seleccionada de -3 a 3
        values = np.linspace(-3, 3, steps)
        
        # Generar y mostrar imágenes
        for i, val in enumerate(values):
            # Modificar dimensión específica
            z = z_base.copy()
            z[0, dim] = val
            
            # Decodificar
            img = vae.decoder.forward(z)
            
            # Mostrar
            plt.subplot(n_dims, steps, dim * steps + i + 1)
            plt.imshow(img[0].reshape(img_size, img_size), cmap='gray')
            plt.axis('off')
            
            # Añadir etiquetas
            if i == 0:
                plt.ylabel(f"Dim {dim+1}")
            if dim == 0:
                plt.title(f"{val:.1f}σ")
    
    plt.suptitle("Traversal de Dimensiones Latentes", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300)
    plt.close()

def compare_beta_interpolations(models_info, output_dir):
    """
    Compara interpolaciones entre VAEs con diferentes valores de beta.
    models_info es una lista de tuplas (vae, img_size, beta).
    """
    # Asegurar que todos tienen la misma dimensión latente
    latent_dim = models_info[0][0].latent_dim
    for vae, _, _ in models_info:
        if vae.latent_dim != latent_dim:
            print(f"Advertencia: Dimensiones latentes inconsistentes {vae.latent_dim} vs {latent_dim}")
    
    # Generar puntos fijos para comparación consistente
    z1 = np.random.randn(latent_dim)
    z2 = np.random.randn(latent_dim)
    
    # Parámetros de la figura
    steps = 11
    n_models = len(models_info)
    
    # Crear figura
    fig = plt.figure(figsize=(12, 2 * n_models))
    
    # Para cada modelo
    for i, (vae, img_size, beta) in enumerate(models_info):
        # Interpolar
        interp_points = linear_interpolation(z1, z2, steps)
        interp_images = vae.decoder.forward(interp_points)
        
        # Mostrar resultados
        for j in range(steps):
            plt.subplot(n_models, steps, i * steps + j + 1)
            plt.imshow(interp_images[j].reshape(img_size, img_size), cmap='gray')
            plt.axis('off')
            
            # Añadir etiquetas
            if j == 0:
                plt.ylabel(f"β = {beta}")
            if i == 0:
                plt.title(f"{j/(steps-1):.1f}")
    
    plt.suptitle("Interpolación con diferentes valores β", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_dir / "beta_comparison_interpolation.png", dpi=300)
    plt.close()
    print(f"Comparación de interpolaciones guardada")

def main():
    parser = argparse.ArgumentParser(description="Genera interpolaciones en el espacio latente de VAEs.")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True,
                       help="Directorios de checkpoints a analizar (puede ser múltiple)")
    parser.add_argument("--config", type=str, required=True,
                       help="Archivo de configuración JSON del modelo")
    parser.add_argument("--output_dir", type=str, default="analysis_results/interpolations",
                       help="Directorio para guardar resultados")
    parser.add_argument("--n_pairs", type=int, default=5,
                       help="Número de pares de puntos para interpolación")
    parser.add_argument("--steps", type=int, default=11,
                       help="Número de pasos en la interpolación")
    args = parser.parse_args()
    
    # Crear directorio para resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lista para almacenar información de modelos
    models_info = []
    
    # Para cada checkpoint
    for checkpoint_dir in args.checkpoints:
        checkpoint_name = os.path.basename(checkpoint_dir)
        beta = extract_beta(checkpoint_name)
        
        print(f"Analizando modelo desde {checkpoint_name} (β={beta})...")
        
        # Cargar modelo
        vae, img_size = load_vae_model(checkpoint_dir, args.config)
        
        # Añadir a la lista para comparación
        models_info.append((vae, img_size, beta))
        
        # Directorio específico para este modelo
        model_dir = output_dir / checkpoint_name
        model_dir.mkdir(exist_ok=True)
        
        # Crear interpolaciones lineales
        print("Generando interpolaciones lineales...")
        create_interpolation_grid(
            vae, img_size,
            model_dir / "linear_interpolation.png",
            n_pairs=args.n_pairs,
            steps=args.steps,
            method='linear'
        )
        
        # Crear interpolaciones esféricas
        print("Generando interpolaciones esféricas...")
        create_interpolation_grid(
            vae, img_size,
            model_dir / "spherical_interpolation.png",
            n_pairs=args.n_pairs,
            steps=args.steps,
            method='spherical'
        )
        
        # Generar traversals de dimensiones latentes
        print("Generando traversals del espacio latente...")
        generate_latent_traversals(
            vae, img_size,
            model_dir / "latent_traversals.png",
            n_dims=min(5, vae.latent_dim),
            steps=args.steps
        )
    
    # Si hay más de un modelo, comparar interpolaciones con diferentes beta
    if len(models_info) > 1:
        print("Generando comparativa de interpolaciones entre diferentes valores β...")
        models_info.sort(key=lambda x: x[2] if x[2] is not None else float('inf'))  # Ordenar por beta
        compare_beta_interpolations(models_info, output_dir)
    
    print("Análisis de interpolaciones completado.")

if __name__ == "__main__":
    main()