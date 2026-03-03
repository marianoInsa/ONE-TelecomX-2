"""
Módulo de análisis de correlación y selección de features.

Proporciona funciones para calcular la correlación de Pearson con la
variable objetivo, información mutua para relaciones no lineales,
VIF para multicolinealidad, y seleccionar features por umbral.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import CORR_THRESHOLD, RANDOM_STATE


def compute_correlation(
    df_encoded: pd.DataFrame,
    target: str = "Churn",
) -> pd.Series:
    """Calcula la correlación de Pearson de cada feature con el target.

    Parameters
    ----------
    df_encoded : pd.DataFrame
        Dataset con todas las columnas numéricas (post-encoding).
    target : str
        Nombre de la columna objetivo.

    Returns
    -------
    pd.Series
        Correlaciones ordenadas de mayor a menor (excluyendo el target).
    """
    corr_matrix = df_encoded.corr(numeric_only=True)
    corr_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)

    print(f"📊 Features analizadas: {len(corr_with_target)}")
    print(f"\nTop 10 correlaciones POSITIVAS con {target}:")
    print(corr_with_target.head(10).to_string())
    print(f"\nTop 10 correlaciones NEGATIVAS con {target}:")
    print(corr_with_target.tail(10).to_string())

    return corr_with_target


def select_features_by_correlation(
    corr_series: pd.Series,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Selecciona features cuya correlación absoluta supera un umbral.

    Parameters
    ----------
    corr_series : pd.Series
        Correlaciones de cada feature con el target (salida de ``compute_correlation``).
    threshold : float
        Umbral mínimo de ``|correlación|``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(selected_features, discarded_features)``
    """
    selected = corr_series[corr_series.abs() >= threshold].index.tolist()
    discarded = corr_series[corr_series.abs() < threshold].index.tolist()

    print(f"=== Selección de Variables (umbral |r| >= {threshold}) ===")
    print(f"\n  Features totales     : {len(corr_series)}")
    print(f"  Features seleccionadas: {len(selected)}")
    print(f"  Features descartadas  : {len(discarded)}")

    print(f"\n📋 Features seleccionadas (ordenadas por |correlación|):")
    ordered = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    for feat in ordered.index:
        if feat in selected:
            print(f"   {corr_series[feat]:+.4f}  {feat}")

    print(f"\n🗑  Features descartadas (|r| < {threshold}):")
    for feat in discarded:
        print(f"   {corr_series[feat]:+.4f}  {feat}")

    return selected, discarded


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
    n_neighbors: int = 3,
) -> pd.Series:
    """Calcula la Información Mutua de cada feature con el target binario.

    A diferencia de Pearson, la Información Mutua (MI) captura relaciones
    **no lineales** y **no monótonas** entre las features y la variable
    objetivo, lo que la convierte en un complemento indispensable.

    Las features binarias (solo valores 0/1) se marcan como discretas
    para mejorar la estimación de MI y el rendimiento.

    Parameters
    ----------
    X : pd.DataFrame
        Features (sin la columna target).
    y : pd.Series
        Variable objetivo binaria.
    random_state : int
        Semilla para reproducibilidad.
    n_neighbors : int
        Número de vecinos para la estimación de MI.

    Returns
    -------
    pd.Series
        Scores de MI ordenados de mayor a menor.
    """
    # Detectar features discretas (binarias 0/1) para optimizar MI
    discrete_mask = np.array([
        set(X[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        for col in X.columns
    ])

    mi_scores = mutual_info_classif(
        X, y,
        discrete_features=discrete_mask,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    mi_series = pd.Series(mi_scores, index=X.columns, name="MI").sort_values(ascending=False)

    print(f"📊 Información Mutua — Features analizadas: {len(mi_series)}")
    print(f"\nTop 10 features por MI con {y.name or 'target'}:")
    for feat, score in mi_series.head(10).items():
        print(f"   {score:.4f}  {feat}")

    bottom = mi_series.tail(5)
    print(f"\nBottom 5 (menor señal):")
    for feat, score in bottom.items():
        print(f"   {score:.4f}  {feat}")

    return mi_series


def compute_vif(
    X: pd.DataFrame,
    max_features: int | None = None,
) -> pd.DataFrame:
    """Calcula el Variance Inflation Factor (VIF) para detectar multicolinealidad.

    Un VIF > 5 indica multicolinealidad moderada; > 10 indica alta.
    Features con VIF elevado pueden inflar los coeficientes de modelos
    lineales (Regresión Logística, SVM lineal).

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de features numéricas (sin target).
    max_features : int | None
        Si se indica, solo calcula VIF para las N features con mayor
        varianza (optimización cuando hay muchas columnas dummy).

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``['feature', 'VIF']`` ordenado descendente.
    """
    if max_features is not None:
        # Seleccionar las features con mayor varianza (las más informativas)
        top_cols = X.var().nlargest(max_features).index.tolist()
        X_vif = X[top_cols].copy()
    else:
        X_vif = X.copy()

    # Agregar constante para el cálculo de VIF
    X_vif = X_vif.assign(_const=1.0)

    vif_data = []
    for i, col in enumerate(X_vif.columns):
        if col == "_const":
            continue
        vif_val = variance_inflation_factor(X_vif.values, i)
        vif_data.append({"feature": col, "VIF": round(vif_val, 2)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)

    print("=== Análisis de Multicolinealidad (VIF) ===")
    high = vif_df[vif_df["VIF"] > 5]
    if len(high) > 0:
        print(f"\n⚠️  {len(high)} features con VIF > 5 (multicolinealidad):")
        for _, row in high.iterrows():
            flag = "🔴 ALTO" if row["VIF"] > 10 else "🟡 MODERADO"
            print(f"   {flag}  VIF={row['VIF']:<8}  {row['feature']}")
    else:
        print("\n✅ Ninguna feature con VIF > 5.")

    print(f"\n📊 VIF promedio: {vif_df['VIF'].mean():.2f}")
    return vif_df
