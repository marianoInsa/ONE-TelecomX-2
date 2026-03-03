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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calcula las métricas estándar de clasificación binaria."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
    }


def evaluate_model(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train_true: pd.Series | np.ndarray | None = None,
    y_pred_train: np.ndarray | None = None,
) -> dict:
    """Evalúa un modelo de clasificación e imprime un reporte formateado.

    Parameters
    ----------
    model_name : str
        Nombre descriptivo del modelo.
    y_true : array-like
        Etiquetas reales del conjunto de **test**.
    y_pred : array-like
        Predicciones del modelo en el conjunto de **test**.
    y_train_true : array-like, optional
        Etiquetas reales del conjunto de **train** (para análisis de
        overfitting).
    y_pred_train : array-like, optional
        Predicciones del modelo en el conjunto de **train**.

    Returns
    -------
    dict
        ``{"model": str, "test": {métricas}, "train": {métricas} | None}``
    """

    result: dict = {"model": model_name}

    # --- Test ---
    test_m = _compute_metrics(y_true, y_pred)
    result["test"] = test_m

    # --- Train (opcional) ---
    if y_train_true is not None and y_pred_train is not None:
        train_m = _compute_metrics(y_train_true, y_pred_train)
        result["train"] = train_m
    else:
        result["train"] = None

    # --- Reporte ---
    print(f"{'=' * 50}")
    print(f"  📊 Evaluación: {model_name}")
    print(f"{'=' * 50}")

    def _print_block(label: str, m: dict) -> None:
        print(f"\n  ▸ {label}")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1-score  : {m['f1']:.4f}")

    _print_block("Test set", test_m)
    if result["train"] is not None:
        _print_block("Train set", result["train"])

    print()
    return result
