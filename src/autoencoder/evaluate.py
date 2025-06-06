import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

def load_font_data():
    """
    Misma función que en `train_autoencoder.py` para obtener el array (32,35).
    """
    if os.path.exists("font_data.npy"):
        return np.load("font_data.npy")
    else:
        raise FileNotFoundError("No se encontró 'font_data.npy'. Primero entrena y genera este archivo.")


def plot_reconstruction(original: np.ndarray, reconstructed: np.ndarray, idx: int):
    """
    Dibuja el carácter original y su reconstrucción para el índice idx.
    Cada uno es un vector de 35 bits ⇒ lo convertimos a un array 5×7.
    """
    orig_img = original.reshape((7, 5)).T  # Transponemos para que 5 ancho × 7 alto
    recon_img = (reconstructed > 0.5).astype(int).reshape((7, 5)).T  # Umbral al 0.5

    fig, axes = plt.subplots(1, 2, figsize=(3, 5))
    axes[0].imshow(orig_img, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"Original #{idx}")
    axes[0].axis("off")

    axes[1].imshow(recon_img, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruido #{idx}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # 1) Cargar datos
    X = load_font_data()  # (32,35)

    # 2) Reconstruir con el Autoencoder entrenado
    #    Debemos instanciar la misma arquitectura que en train, y luego cargar pesos.
    #    Para simplificar, repetimos la lectura de config.json aquí:
    import json
    with open("config.json", "r") as f:
        cfg = json.load(f)

    # Mismos parámetros de arquitectura (encoder/decoder) que en el entrenamiento
    encoder_sizes       = cfg["encoder"]["layer_sizes"]
    encoder_activations  = cfg["encoder"]["activations"]
    decoder_sizes       = cfg["decoder"]["layer_sizes"]
    decoder_activations  = cfg["decoder"]["activations"]
    dropout_rate        = cfg.get("dropout_rate", 0.0)

    ae = Autoencoder(
        encoder_sizes=encoder_sizes,
        encoder_activations=encoder_activations,
        decoder_sizes=decoder_sizes,
        decoder_activations=decoder_activations,
        dropout_rate=dropout_rate
    )

    # Cargar pesos
    checkpoint_path = "checkpoints/ae_weights.npz"
    ae.load_weights(checkpoint_path)
    ae.eval_mode()

    # 3) Obtener reconstrucciones y latentes
    X_recon = ae.forward(X)         # (32,35)
    latentes = ae._latent.copy()    # (32, 2)  — acceso a la variable interna guardada por forward()

    # 4) Mostrar algunas reconstrucciones (por ejemplo, los primeros 5 caracteres)
    for i in range(min(5, X.shape[0])):
        plot_reconstruction(X, X_recon, idx=i)

    # 5) Mostrar en un scatter plot los puntos latentes para los 32 caracteres
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.scatter(latentes[:, 0], latentes[:, 1], c="blue", marker="o")
    for i, (x0, x1) in enumerate(latentes):
        plt.text(x0 + 0.02, x1 + 0.02, str(i), fontsize=9)
    plt.title("Visualización 2D del espacio latente")
    plt.xlabel("Latente dim 1")
    plt.ylabel("Latente dim 2")
    plt.grid(alpha=0.3)
    plt.show()

    # 6) Generar MUY SENCILLO: tomar un punto latente intermedio y pasar por decoder
    #    (por ejemplo, media de todos los latentes) → generar “nuevo carácter” no visto.
    z_mean = latentes.mean(axis=0, keepdims=True)  # forma (1,2)
    x_gen = ae.decoder.forward(z_mean)             # reconstrucción desde el latente medio
    print("Reconstrucción desde el punto latente medio (2D):")
    plot_reconstruction(np.zeros_like(x_gen), x_gen, idx=-1)  # No hay “original” para idx=-1


if __name__ == "__main__":
    import os
    main()
