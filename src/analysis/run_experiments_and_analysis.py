#!/usr/bin/env python3
# filepath: /Users/saints/Desktop/ITBA/SIA/sia-tp5/src/run_experiments_and_analysis.py
# Script para ejecutar experimentos y análisis completos

import subprocess
import os
import glob
from pathlib import Path
import time
import argparse

def run_command(command, description=None):
    """Ejecuta un comando y registra su éxito o fracaso."""
    if description:
        print(f"\n=== {description} ===")
    print(f"Ejecutando: {command}")
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        print("✓ Comando ejecutado exitosamente")
        return True
    else:
        print(f"✗ Error al ejecutar comando (código {result.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de VAE y análisis automatizados")
    parser.add_argument("--config", type=str, default="configs/vae.json",
                       help="Archivo de configuración JSON para VAE")
    parser.add_argument("--skip_training", action="store_true",
                       help="Omitir entrenamiento y solo ejecutar análisis")
    parser.add_argument("--img_size", type=int, default=28,
                       help="Tamaño de imagen para emojis (dSprites siempre usa 64x64)")
    parser.add_argument("--max_emojis", type=int, default=500,
                       help="Número máximo de emojis a usar")
    args = parser.parse_args()
    
    # Configuraciones de experimentos
    experiments = [
        {"dataset": "faces_only", "beta": 0.1, "img_size": args.img_size},
        {"dataset": "faces_only", "beta": 1.0, "img_size": args.img_size},
        {"dataset": "faces_only", "beta": 4.0, "img_size": args.img_size},
        {"dataset": "dsprites", "beta": 1.0},  # dsprites siempre es 64x64
        {"dataset": "dsprites", "beta": 4.0, "kl_ramp_epochs": 100}
    ]
    
    # 1. Ejecutar entrenamientos
    if not args.skip_training:
        print("\n=== FASE 1: ENTRENAMIENTO DE MODELOS ===")
        
        for exp in experiments:
            dataset = exp["dataset"]
            beta = exp["beta"]
            img_size = exp.get("img_size", args.img_size)
            kl_ramp = exp.get("kl_ramp_epochs", 0)
            
            cmd = [
                f"python src/runner_vae.py {args.config}",
                f"--dataset {dataset}",
                f"--img_size {img_size}",
                f"--beta_final {beta}",
                f"--max_emojis {args.max_emojis}"
            ]
            
            if kl_ramp > 0:
                cmd.append(f"--kl_ramp_epochs {kl_ramp}")
            
            command = " ".join(cmd)
            run_command(command, f"Entrenando VAE con {dataset}, β={beta}")
            
        print("\n✓ Entrenamiento de todos los modelos completado\n")
    else:
        print("\n=== FASE 1: ENTRENAMIENTO DE MODELOS (OMITIDO) ===")
    
    # Esperar un momento para asegurarse de que los archivos estén completos
    time.sleep(1)
    
    # 2. Ejecutar análisis
    print("\n=== FASE 2: ANÁLISIS DE RESULTADOS ===")
    
    # Encontrar los checkpoints generados
    checkpoints = sorted(glob.glob("checkpoints/*_beta*"))
    
    if not checkpoints:
        print("No se encontraron checkpoints para analizar. Verifica si el entrenamiento fue exitoso.")
        return
    
    print(f"Se encontraron {len(checkpoints)} checkpoints para analizar.")
    
    # Análisis de convergencia
    run_command("python src/analysis/vae_convergence_analysis.py", 
                "Analizando convergencia de todos los modelos")
    
    # Análisis individual del espacio latente
    for checkpoint in checkpoints:
        run_command(
            f"python src/analysis/vae_latent_space_analysis.py --checkpoint {checkpoint} --config {args.config}",
            f"Analizando espacio latente de {os.path.basename(checkpoint)}"
        )
    
    # Agrupar checkpoints por dataset para comparaciones
    datasets = {}
    for checkpoint in checkpoints:
        base_name = os.path.basename(checkpoint)
        dataset = base_name.split('_')[0]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(checkpoint)
    
    # Análisis de interpolaciones por dataset
    for dataset, dataset_checkpoints in datasets.items():
        if len(dataset_checkpoints) > 1:  # Necesitamos al menos dos para comparar
            checkpoint_args = " ".join(dataset_checkpoints)
            run_command(
                f"python src/analysis/vae_interpolation_analysis.py --checkpoints {checkpoint_args} --config {args.config}",
                f"Generando interpolaciones para dataset {dataset}"
            )
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Todos los resultados están disponibles en la carpeta 'analysis_results/'")

if __name__ == "__main__":
    main()