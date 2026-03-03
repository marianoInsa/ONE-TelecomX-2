"""
Módulo de modelado.

Funciones para entrenar, serializar y gestionar modelos de clasificación.
Estandariza la salida de entrenamiento y la persistencia de artefactos
para mantener los notebooks limpios y reproducibles.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODELS_DIR


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str = "",
) -> object:
    """Entrena un modelo de clasificación y reporta el resultado.

    Parameters
    ----------
    model : estimator
        Instancia de un clasificador de scikit-learn (no ajustado).
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.
    model_name : str
        Nombre descriptivo del modelo (para los mensajes de salida).

    Returns
    -------
    object
        El modelo ya ajustado (fitted).
    """
    name = model_name or type(model).__name__

    print(f"=== Entrenando: {name} ===")
    print(f"  Samples  : {X_train.shape[0]:,}")
    print(f"  Features : {X_train.shape[1]}")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    print(f"  Tiempo   : {elapsed:.2f}s")
    print(f"\n✅ {name} entrenado correctamente.")

    return model


def save_model(
    model,
    model_name: str,
    *,
    directory: Path | str | None = None,
) -> Path:
    """Serializa un modelo entrenado en disco.

    El archivo se guarda con el nombre ``{model_name}.pkl`` dentro del
    directorio indicado (por defecto ``MODELS_DIR``).

    Parameters
    ----------
    model : estimator
        Modelo ya ajustado.
    model_name : str
        Nombre base para el archivo (sin extensión).
    directory : Path | str | None
        Directorio destino. Si es ``None``, usa ``MODELS_DIR``.

    Returns
    -------
    Path
        Ruta absoluta al archivo serializado.
    """
    directory = Path(directory) if directory is not None else MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{model_name}.pkl"
    joblib.dump(model, path)

    print(f"💾 Modelo guardado: {path.relative_to(directory.parent)}")
    return path
