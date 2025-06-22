# TP 5 – SIA · Deep Learning  
*Autoencoders, Denoising AE y Variational AE*

> **Sistemas de Inteligencia Artificial**  
> ITBA – 2025  

[📑 Enunciado oficial](docs/sia-tp5.pdf)

---

## Introducción

Este TP explora tres variantes de autoencoders sobre distintos datasets:

* **AE básico** – aprende los 32 caracteres 5 × 7 de `font.h`  
* **Denoising AE** – quita ruido binario de los mismos caracteres  
* **VAE** – genera emojis (OpenMoji) y disentangled sprites (dSprites)

El código es *NumPy-only* y corre en CPU en minutos.

---

## ⚙️ Instalación rápida

1. Clonar el repositorio  
   ```git clone https://github.com/tu-usuario/sia-tp5.git```
2. Crear y activar entorno virtual  
   ```python -m venv venv && source venv/bin/activate```
3. Instalar dependencias  
   ```pip install -r requirements.txt```

---

## ▶️ Ejecución

### Ejercicio 1a — Autoencoder básico (font.h)

Runner `src/runner_autoencoder.py`  
Flags principales  
• `config_path` – JSON con arquitectura  
• `font_path`  – ruta a font.h  

Ejemplo mínimo  
```python src/runner_autoencoder.py configs/ae.json data/font.h```

Genera  
checkpoints/ae_weights.npz   ·  checkpoints/loss_history.npy

---

### Ejercicio 1b — Denoising Autoencoder

Runner `src/runner_dae.py`  
Flags extra  
• `noise_level` (en el JSON) – prob. de voltear cada bit  

Ejemplo con 10 % de ruido  
```python src/runner_dae.py configs/dae.json data/font.h```

Salida adicional: métrica *bits error* por carácter.

---

### Ejercicio 2 — Variational AE

Runner `src/runner_vae.py`  
Flags clave  
• `--dataset`        faces_only | emoji_full | dsprites  
• `--img_size`       28/32/64 (según dataset)  
• `--beta_final`     valor final de β  
• `--kl_ramp_epochs` épocas para annealing 0 → β  

Comandos sugeridos  

| Dataset | Comando | Comentario |
|---------|---------|------------|
| Caras amarillas (197) | ```python src/runner_vae.py configs/vae.json --dataset faces_only``` | demo rápida |
| OpenMoji completo | ```python src/runner_vae.py configs/vae.json --dataset emoji_full --max_emojis 2000``` | ~4 200 PNG |
| dSprites 64×64 | ```python src/runner_vae.py configs/vae.json --dataset dsprites --beta_final 4 --kl_ramp_epochs 100``` | muestra disentanglement |

Cada run guarda en  
checkpoints/`<dataset>`_img`<size>`_beta`<β>`_`<timestamp>`/  
    vae_weights.npz · loss_history.npy · vae_generated.png  

Un *sample raw* del dataset queda en  
data/sample_`<dataset>`.png

---

## 📁 Estructura de carpetas

src/              · implementación AE / VAE / trainer  
configs/          · JSON de hiper-parámetros  
data/             · datasets y samples de referencia  
checkpoints/      · pesos + salidas organizadas por experimento  
docs/             · enunciado y apuntes útiles  

---

## 📊 Análisis de resultados  *(pendiente)*

Se agregarán:  
• gráficas de convergencia,  
• proyección 2-D del espacio latente (AE) y  
• interpolaciones en el VAE con distintas β.