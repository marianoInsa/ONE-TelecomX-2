"""
Módulo de preprocesamiento de datos.

Contiene funciones para la preparación del dataset previo al modelado:
eliminación de columnas, encoding de variables categóricas, división
train/test, balanceo con SMOTE y estandarización.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED_DIR, RANDOM_STATE, TEST_SIZE


# Resultado del pipeline de split + balanceo
@dataclass
class SplitResult:
    """Contenedor con todos los artefactos generados por ``split_and_balance``."""

    X_train_bal: pd.DataFrame
    y_train_bal: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_train: pd.DataFrame
    y_train: pd.Series
    scaler: StandardScaler | None = None


def drop_non_predictive(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Elimina columnas sin poder predictivo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    columns : list[str]
        Columnas a eliminar.

    Returns
    -------
    pd.DataFrame
        DataFrame sin las columnas indicadas.
    """
    df = df.drop(columns=[c for c in columns if c in df.columns])
    print(f'✅ Columnas eliminadas: {columns}')
    print(f'📊 Nuevo shape: {df.shape[0]} filas × {df.shape[1]} columnas')
    return df


def encode_features(
    df: pd.DataFrame,
    binary_text_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    *,
    drop_first: bool = True,
    encoder: OneHotEncoder | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Codifica variables categóricas para ML.

    Realiza dos transformaciones:
    1. Mapeo binario Yes/No → 1/0 para columnas dicotómicas en texto.
    2. One-Hot Encoding para columnas categóricas multi-clase usando
       ``sklearn.preprocessing.OneHotEncoder``.

    Cuando se usa con train/test separados, pasar ``fit=True`` para el
    train set (ajusta y transforma) y ``fit=False`` con el ``encoder``
    devuelto para el test set (solo transforma). Esto **previene data
    leakage** al no exponer categorías del test al encoder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas categóricas originales.
    binary_text_cols : list[str] | None
        Columnas binarias almacenadas como 'Yes'/'No'.
        Por defecto ``['account_paperlessbilling']``.
    categorical_cols : list[str] | None
        Columnas categóricas multi-clase para OHE.
    drop_first : bool
        Si True, elimina la primera categoría por variable (evita multicolinealidad).
    encoder : OneHotEncoder | None
        Encoder ya ajustado. Si se proporciona, se usa para transformar
        (``fit`` debe ser ``False``).
    fit : bool
        Si True, ajusta el encoder a los datos. Si False, usa ``encoder``
        existente (para el test set).

    Returns
    -------
    tuple[pd.DataFrame, OneHotEncoder]
        ``(df_encoded, fitted_encoder)`` — el DataFrame codificado y el
        encoder ajustado para reutilizar en el test set.
    """
    df = df.copy()
    binary_text_cols = binary_text_cols or ["account_paperlessbilling"]
    categorical_cols = categorical_cols or [
        "customer_gender",
        "phone_multiplelines",
        "internet_internetservice",
        "internet_onlinesecurity",
        "internet_onlinebackup",
        "internet_deviceprotection",
        "internet_techsupport",
        "internet_streamingtv",
        "internet_streamingmovies",
        "account_contract",
        "account_paymentmethod",
    ]

    # Filtrar solo columnas que existen en el DataFrame
    binary_text_cols = [c for c in binary_text_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # A) Columnas binarias de texto → 1/0
    for col in binary_text_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    if binary_text_cols:
        print(f"✅ Columnas binarias de texto convertidas a 0/1: {binary_text_cols}")

    # B) One-Hot Encoding con sklearn (sin data leakage)
    cols_antes = df.shape[1]

    if not categorical_cols:
        print("ℹ️  No hay columnas categóricas para codificar.")
        return df, encoder  # type: ignore[return-value]

    drop_strategy = "first" if drop_first else None

    if fit:
        encoder = OneHotEncoder(
            drop=drop_strategy,
            sparse_output=False,
            dtype=int,
            handle_unknown="infrequent_if_exist",
        )
        ohe_array = encoder.fit_transform(df[categorical_cols])
    else:
        if encoder is None:
            raise ValueError("Se debe proporcionar un encoder ajustado cuando fit=False.")
        ohe_array = encoder.transform(df[categorical_cols])

    # Obtener nombres de las columnas generadas
    ohe_columns = encoder.get_feature_names_out(categorical_cols).tolist()

    # Construir DataFrame final: columnas no-categóricas + columnas OHE
    other_cols = [c for c in df.columns if c not in categorical_cols]
    df_encoded = pd.concat(
        [
            df[other_cols].reset_index(drop=True),
            pd.DataFrame(ohe_array, columns=ohe_columns),
        ],
        axis=1,
    )
    cols_despues = df_encoded.shape[1]

    print(f"📊 Columnas antes del encoding : {cols_antes}")
    print(f"📊 Columnas después del encoding: {cols_despues} (+{cols_despues - cols_antes} columnas dummy)")
    print(f"\n✅ Todas las columnas son numéricas: {all(df_encoded.dtypes != object)}")

    return df_encoded, encoder


def split_and_balance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    apply_smote: bool = True,
    scale_before_smote: bool = True,
) -> SplitResult:
    """Divide el dataset y balancea las clases con SMOTE.

    SMOTE se aplica **únicamente** sobre el conjunto de entrenamiento
    para evitar data leakage.

    Cuando ``scale_before_smote=True`` (por defecto), el orden es:
    split → scale (fit en train) → SMOTE. Esto es necesario porque
    SMOTE usa distancia euclidiana para encontrar vecinos y las features
    sin escalar con rangos dispares sesgan la generación de ejemplos
    sintéticos.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Variable objetivo.
    test_size : float
        Proporción del test set.
    random_state : int
        Semilla para reproducibilidad.
    apply_smote : bool
        Si True, aplica SMOTE al train set.
    scale_before_smote : bool
        Si True, escala las features antes de SMOTE para que la distancia
        euclidiana sea equitativa. El scaler se incluye en ``SplitResult``.

    Returns
    -------
    SplitResult
        Objeto con todos los artefactos de split y balanceo.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("=== División Train / Test ===")
    print(f"  Train : {X_train.shape[0]} registros  ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Test  : {X_test.shape[0]}  registros  ({X_test.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Features: {X_train.shape[1]} columnas")
    print(f"\n  Distribución en Train | Churn: {y_train.mean() * 100:.2f}%")
    print(f"  Distribución en Test  | Churn: {y_test.mean() * 100:.2f}%")

    # Escalado previo a SMOTE
    fitted_scaler: StandardScaler | None = None
    if scale_before_smote:
        X_train_for_smote, X_test, fitted_scaler = scale_features(X_train, X_test)
        print("\n✅ Escalado aplicado ANTES de SMOTE (distancia euclidiana corregida).")
    else:
        X_train_for_smote = X_train

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_for_smote, y_train)
        y_train_bal = pd.Series(y_train_bal, name=y.name)

        # Preservar como DataFrame si la entrada lo era
        if isinstance(X_train_for_smote, pd.DataFrame):
            X_train_bal = pd.DataFrame(X_train_bal, columns=X_train_for_smote.columns)

        bal_counts = y_train_bal.value_counts()
        print("\n=== Distribución después de SMOTE (Train) ===")
        print(f"  Clase 0 | No canceló : {bal_counts[0]:>5} registros ({bal_counts[0] / len(y_train_bal) * 100:.1f}%)")
        print(f"  Clase 1 | Canceló    : {bal_counts[1]:>5} registros ({bal_counts[1] / len(y_train_bal) * 100:.1f}%)")
        print(f"\n  Ejemplos sintéticos generados: {bal_counts[1] - y_train.sum()} nuevos registros de churn")
        print(f"\n✅ Test set permanece intacto: {len(y_test)} registros (sin modificar)")
    else:
        X_train_bal = X_train_for_smote
        y_train_bal = y_train

    return SplitResult(
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        scaler=fitted_scaler,
    )


def scale_features(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Estandariza features con StandardScaler.

    Ajusta (fit) exclusivamente sobre el train set y aplica (transform)
    a ambos conjuntos para evitar data leakage.

    Parameters
    ----------
    X_train : array-like
        Datos de entrenamiento.
    X_test : array-like
        Datos de prueba.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        ``(X_train_scaled, X_test_scaled, scaler)``
        Los DataFrames conservan los nombres de columna e índice originales.
    """
    # Preservar nombres de columna e índices si vienen como DataFrame
    columns = X_train.columns if hasattr(X_train, "columns") else None
    train_index = X_train.index if hasattr(X_train, "index") else None
    test_index = X_test.index if hasattr(X_test, "index") else None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Envolver en DataFrame para preservar trazabilidad de features
    if columns is not None:
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns, index=train_index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns, index=test_index)

    means = X_train_scaled.mean(axis=0) if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.values.mean(axis=0)
    stds = X_train_scaled.std(axis=0) if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.values.std(axis=0)

    print("=== Verificación del escalado (Train) ===")
    print(f"  Media promedio de todas las features : {means.mean():.6f}  (esperado ≈ 0)")
    print(f"  Std  promedio de todas las features  : {stds.mean():.6f}  (esperado ≈ 1)")
    print(f"\n  Shape X_train_scaled : {X_train_scaled.shape}")
    print(f"  Shape X_test_scaled  : {X_test_scaled.shape}")
    print(f"\n✅ Estandarización completada.")

    return X_train_scaled, X_test_scaled, scaler


def load_selected_features(
    path: Path | str | None = None,
) -> list[str]:
    """Carga la lista de features seleccionadas desde ``selected_features.json``.

    Lee el artefacto generado por el notebook 02 (correlación y selección)
    y devuelve únicamente la lista de nombres de columnas.

    Parameters
    ----------
    path : Path | str | None
        Ruta al archivo JSON. Si es ``None``, usa la ruta por defecto
        ``DATA_PROCESSED_DIR / 'selected_features.json'``.

    Returns
    -------
    list[str]
        Nombres de las features seleccionadas.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    KeyError
        Si el JSON no contiene la clave ``'selected_features'``.
    """
    path = Path(path) if path is not None else DATA_PROCESSED_DIR / "selected_features.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de features seleccionadas: {path}\n"
            "Asegúrate de ejecutar primero el notebook 02_correlacion_seleccion."
        )

    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    if isinstance(artifact, dict) and "selected_features" in artifact:
        features = artifact["selected_features"]
        threshold = artifact.get("threshold", "N/A")
        method = artifact.get("method", "N/A")
    elif isinstance(artifact, list):
        features = artifact
        threshold = "N/A"
        method = "N/A"
    else:
        raise KeyError(
            "El archivo JSON no contiene la clave 'selected_features' "
            "ni es una lista directa de features."
        )

    print(f"📂 Features cargadas desde: {path.name}")
    print(f"   Método de selección : {method}")
    print(f"   Umbral aplicado     : |r| >= {threshold}")
    print(f"   Features retenidas  : {len(features)} columnas")

    return features
