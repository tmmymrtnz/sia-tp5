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
1. Para encontrar la arquitectura óptima, se debe correr:
   ```bash
   python src/ex1/part_a/ej_b.py configs/ex1/config_a_1.json
   ```
   Donde el archivo de configuración contiene distintos experimentos y el directorio de font.h
   En el directorio `/experimentos` se guardará el gráfico de pérdida por época y un archivo json con los resultados finales de cada experimento.

2. Para graficar el espacio latente, se debe correr:
   ```bash 
   python src/ex1/part_a/ej_c.py
   ```
   Tomará directamente el archivo font.h y la configuración `configs/ex1/optimal.json` donde se encuentran los detalles de la arquitectura óptima hallada en el item anterior.
   Genera un gráfico del espacio latente 2D con los caracteres proyectados sobre el punto (z) que los codifica.

3. Para generar nuevas letras que no pertenecen al conjunto de entrenamiento, se debe correr:
   ```bash
   python src/ex1/part_a/ej_d.py
   ```
   Entrena al autoencoder con las configuraciones óptimas de `configs/ex1/optimal.json`, luego se realizan tres gráficos: primero se divide el espacio latente en 25 bloques (5x5), cada uno con un punto central (x,y). Se usa este punto (x,y) como entrada del decodificador, generando un caracter para este punto. Se genera un gráfico con los 25 caracteres generados. Se repite el proceso para 10x10 y 20x20.

---

### Ejercicio 1b — Denoising Autoencoder
Para distorsionar las entradas y estudiar la capacidad del Autoencoder de eliminar el ruido, se debe correr:
```bash
python src/analisis_dae.py configs/ex1/optimal.json data/font.h
```
En la configuración se especifica la arquitectura óptima y el nivel de ruido para la prueba.
Se harán 5 ejecuciones de esta prueba de denoising.
Se generarán tres archivos en `/checkpoints`
* `dae_loss_avg_plot.png` con la pérdida promedio a lo largo de las épocas en las 5 ejecuciones
* `ejemplo_mejor.png` y `ejemplo_peor.png` con una visualización de la entrada con ruido, la reconstrucción del AE y la entrada original para el caracter con menos bits de error en promedio, y el caracter con más bits de error en promedio.

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