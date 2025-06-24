#!/usr/bin/env python3
# filepath: /Users/saints/Desktop/ITBA/SIA/sia-tp5/src/analysis/vae_convergence_analysis.py
# Analiza y compara las curvas de convergencia de diferentes experimentos VAE

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def extract_info_from_dirname(dirname):
    """Extrae información relevante del nombre del directorio de experimento."""
    info = {}
    
    # Extraer dataset, tamaño de imagen y beta
    pattern = r"(?P<dataset>[^_]+)_img(?P<img_size>\d+)_beta(?P<beta>[\d\.]+)"
    match = re.search(pattern, dirname)
    
    if match:
        info["dataset"] = match.group("dataset")
        info["img_size"] = int(match.group("img_size"))
        info["beta"] = float(match.group("beta"))
    
    return info

def create_example_data(checkpoints_dir):
    """Crea datos de ejemplo para demostración si no existen experimentos."""
    print("Creando datos de ejemplo para demostración...")
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Crear algunos directorios de ejemplo con historiales de pérdida simulados
    datasets = ["faces_only", "emoji_full", "dsprites"]
    betas = [0.1, 1.0, 4.0]
    img_sizes = [28, 32, 64]
    
    for dataset, img_size in zip(datasets, img_sizes):
        for beta in betas:
            # Crear directorio simulado
            exp_dir = checkpoints_dir / f"{dataset}_img{img_size}_beta{beta}_20250101-120000"
            exp_dir.mkdir(exist_ok=True)
            
            # Generar historial de pérdida sintético
            epochs = 100
            # Simulamos convergencia más rápida con beta pequeño
            noise_factor = 0.1 * (1 + beta)
            base_loss = max(0.5, 2.0 - (1.0 / (1 + beta)))  # Mayor beta, mayor pérdida base
            
            loss_history = np.zeros(epochs)
            for i in range(epochs):
                decay = np.exp(-i / (20 * (1 + 0.5 * beta)))  # Decaimiento más lento con beta mayor
                noise = np.random.normal(0, noise_factor * decay)
                loss_history[i] = base_loss * decay + abs(noise)
            
            # Guardar historial simulado
            np.save(exp_dir / "loss_history.npy", loss_history)
            
            # Opcionalmente, crear componentes de pérdida simulados
            if dataset == "dsprites" and beta == 4.0:
                # Simular rampa KL para un experimento
                recon_loss = np.zeros_like(loss_history)
                kl_loss = np.zeros_like(loss_history)
                kl_ramp = min(50, epochs)
                
                for i in range(epochs):
                    kl_factor = min(1.0, i / kl_ramp) if kl_ramp > 0 else 1.0
                    recon_loss[i] = loss_history[i] * 0.7
                    kl_loss[i] = loss_history[i] * 0.3 * beta * kl_factor
                
                np.savez(exp_dir / "loss_components.npz", 
                        recon_loss=recon_loss,
                        kl_loss=kl_loss,
                        total_loss=loss_history)
    
    print("Datos de ejemplo creados. Continuando con el análisis...")

def plot_convergence_by_dataset(experiment_dirs, output_dir):
    """Genera gráficas de convergencia agrupadas por dataset."""
    # Agrupar experimentos por dataset
    datasets = {}
    for exp_dir in experiment_dirs:
        info = extract_info_from_dirname(exp_dir.name)
        if "dataset" in info:
            if info["dataset"] not in datasets:
                datasets[info["dataset"]] = []
            datasets[info["dataset"]].append((exp_dir, info))
    
    # Para cada dataset, graficar convergencia con diferentes betas
    for dataset, experiments in datasets.items():
        plt.figure(figsize=(10, 6))
        
        for exp_dir, info in sorted(experiments, key=lambda x: x[1].get("beta", 0)):
            # Cargar el historial de pérdida
            loss_path = exp_dir / "loss_history.npy"
            if loss_path.exists():
                loss_history = np.load(loss_path)
                beta = info.get("beta", "?")
                
                epochs = range(1, len(loss_history) + 1)
                plt.plot(epochs, loss_history, linewidth=2, label=f"β={beta}")
        
        plt.title(f"Convergencia para dataset: {dataset}")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"convergence_{dataset}.png", dpi=300)
        print(f"Gráfico de convergencia para {dataset} guardado")
        plt.close()

def plot_convergence_by_beta(experiment_dirs, output_dir):
    """Genera gráficas de convergencia agrupadas por valor beta."""
    # Agrupar experimentos por beta
    betas = {}
    for exp_dir in experiment_dirs:
        info = extract_info_from_dirname(exp_dir.name)
        if "beta" in info:
            beta_key = str(info["beta"])
            if beta_key not in betas:
                betas[beta_key] = []
            betas[beta_key].append((exp_dir, info))
    
    # Para cada beta, comparar convergencia en diferentes datasets
    for beta, experiments in betas.items():
        if len(experiments) <= 1:
            continue  # Saltamos si solo hay un experimento con este beta
            
        plt.figure(figsize=(10, 6))
        
        for exp_dir, info in sorted(experiments, key=lambda x: x[1].get("dataset", "")):
            # Cargar el historial de pérdida
            loss_path = exp_dir / "loss_history.npy"
            if loss_path.exists():
                loss_history = np.load(loss_path)
                dataset = info.get("dataset", "?")
                
                epochs = range(1, len(loss_history) + 1)
                plt.plot(epochs, loss_history, linewidth=2, label=f"Dataset: {dataset}")
        
        plt.title(f"Comparación de datasets con β={beta}")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"beta_comparison_{beta}.png", dpi=300)
        print(f"Gráfico de comparación para β={beta} guardado")
        plt.close()

def analyze_loss_components(experiment_dirs, output_dir):
    """Analiza los componentes de la pérdida si están disponibles."""
    for exp_dir in experiment_dirs:
        components_path = exp_dir / "loss_components.npz"
        if components_path.exists():
            try:
                components = np.load(components_path)
                recon_loss = components['recon_loss']
                kl_loss = components['kl_loss']
                total_loss = components['total_loss']
                
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(total_loss) + 1)
                
                plt.plot(epochs, recon_loss, label="Reconstrucción", linewidth=2)
                plt.plot(epochs, kl_loss, label="KL Divergencia", linewidth=2)
                plt.plot(epochs, total_loss, label="Pérdida Total", linewidth=2, linestyle='--')
                
                info = extract_info_from_dirname(exp_dir.name)
                beta = info.get("beta", "?")
                dataset = info.get("dataset", "?")
                
                plt.title(f"Componentes de pérdida: {dataset}, β={beta}")
                plt.xlabel("Épocas")
                plt.ylabel("Pérdida")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f"components_{exp_dir.name}.png", dpi=300)
                print(f"Componentes de pérdida para {exp_dir.name} guardados")
                plt.close()
            except Exception as e:
                print(f"Error procesando componentes para {exp_dir.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analiza y visualiza la convergencia de experimentos VAE.")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", 
                        help="Directorio donde se encuentran los experimentos")
    parser.add_argument("--output_dir", type=str, default="analysis_results", 
                        help="Directorio donde guardar los resultados")
    parser.add_argument("--create_example_data", action="store_true", 
                        help="Crear datos de ejemplo si no hay experimentos")
    args = parser.parse_args()

    # Configurar directorios
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Buscar directorios de experimentos
    experiment_dirs = sorted(list(checkpoints_dir.glob("*_img*_beta*")))
    
    # Si no hay experimentos, posiblemente crear datos de ejemplo
    if not experiment_dirs and args.create_example_data:
        create_example_data(checkpoints_dir)
        experiment_dirs = sorted(list(checkpoints_dir.glob("*_img*_beta*")))
    
    if not experiment_dirs:
        print("No se encontraron experimentos. Verifica la carpeta de checkpoints.")
        return
    
    print(f"Se encontraron {len(experiment_dirs)} experimentos para analizar.")
    
    # Generar gráficas de convergencia
    plot_convergence_by_dataset(experiment_dirs, output_dir)
    plot_convergence_by_beta(experiment_dirs, output_dir)
    analyze_loss_components(experiment_dirs, output_dir)
    
    print(f"Análisis de convergencia completado. Resultados guardados en {output_dir}")

if __name__ == "__main__":
    main()