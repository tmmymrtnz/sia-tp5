import json
import numpy as np
from common.perceptrons.multilayer.trainer import Trainer             # Asumiendo que trainer.py (con la clase Trainer) está en el mismo paquete.
from autoencoder import Autoencoder
import os

def load_font_data():
    """
    Cargar los 32 patrones 5×7 del archivo "font.h" como un array binario de NumPy.
    Cada carácter se convierte en un vector de 35=5×7 bits {0,1}.
    A modo de ejemplo, aquí sólo se deja el _esqueleto_: asumimos que existe un script
    o función que lee "font.h" y devuelve un array de forma (32,35) dtype=float32.
    
    Por ejemplo, podría parsear líneas como:
        0x1F,0x11,0x1F,... 
    según el formato de font.h original.  
    """
    # >>> REEMPLAZA este bloque con tu rutina real de lectura de "font.h" <<<
    # Supongamos que hemos convertido la fuente en un archivo .npy con shape (32,35).
    # Para que funcione el ejemplo sin errores, crearemos datos ficticios:
    if os.path.exists("font_data.npy"):
        return np.load("font_data.npy")
    else:
        # Generamos 32 vectores binarios aleatorios a modo de placeholder.
        np.random.seed(0)
        dummy = np.random.randint(0, 2, size=(32, 35)).astype(np.float32)
        np.save("font_data.npy", dummy)
        return dummy


def main():
    # 1) Leer configuración
    with open("config.json", "r") as f:
        cfg = json.load(f)

    # Arquitectura del encoder/decoder
    encoder_sizes      = cfg["encoder"]["layer_sizes"]
    encoder_activations = cfg["encoder"]["activations"]
    decoder_sizes      = cfg["decoder"]["layer_sizes"]
    decoder_activations = cfg["decoder"]["activations"]
    dropout_rate       = cfg.get("dropout_rate", 0.0)

    # Parámetros de entrenamiento
    loss_name      = cfg["loss"]
    optimizer_name = cfg["optimizer"]
    optim_kwargs   = cfg.get("optim_kwargs", {})
    batch_size     = cfg["batch_size"]
    max_epochs     = cfg["max_epochs"]
    log_every      = cfg.get("log_every", 100)
    patience       = cfg.get("patience", 10)
    min_delta      = cfg.get("min_delta", 1e-4)

    # 2) Cargar datos (X es (32,35) con valores 0.0/1.0)
    X = load_font_data()  # Forma (32, 35)
    # En un autoencoder básico, Y = X
    Y = X.copy()

    # 3) Crear el Autoencoder
    ae = Autoencoder(
        encoder_sizes=encoder_sizes,
        encoder_activations=encoder_activations,
        decoder_sizes=decoder_sizes,
        decoder_activations=decoder_activations,
        dropout_rate=dropout_rate
    )

    # 4) Crear el Trainer (entrenará sobre X→X minimizando, por ejemplo, MSE)
    trainer = Trainer(
        net=ae,
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        optim_kwargs=optim_kwargs,
        batch_size=batch_size,
        max_epochs=max_epochs,
        log_every=log_every,
        early_stopping=True,
        patience=patience,
        min_delta=min_delta
    )

    # Entrenar
    print(">>> Entrenando Autoencoder...")
    ae.train_mode()
    loss_history = trainer.fit(X, Y)

    # 5) Guardar pesos en disco
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/ae_weights.npz"
    ae.save_weights(checkpoint_path)
    print(f">>> Pesos guardados en '{checkpoint_path}'")

    # (Opcional) Guardar el historial de pérdida
    np.save("checkpoints/loss_history.npy", np.array(loss_history))
    print(">>> Entrenamiento finalizado. Historial de pérdida guardado en 'checkpoints/loss_history.npy'.")


if __name__ == "__main__":
    main()
