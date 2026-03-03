"""
Módulo de carga de datos.

Provee una función unificada para cargar el dataset de Telecom X,
priorizando la lectura local desde data/raw/ con fallback a la URL
remota de GitHub. Incluye validación del esquema esperado.
"""

from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR

# Configuración de fuentes de datos
DATA_URL = (
    "https://raw.githubusercontent.com/marianoInsa/ONE-TelecomX/"
    "243bebbeb92b071c5d05ed0bf47f1fb9fe25ca2c/data/telecom_data_processed.csv"
)
LOCAL_DATA_PATH = DATA_RAW_DIR / "telecom_data_processed.csv"

# Columnas esperadas en el dataset
EXPECTED_COLUMNS = {
    "customerID",
    "Churn",
    "customer_gender",
    "customer_seniorcitizen",
    "customer_partner",
    "customer_dependents",
    "customer_tenure",
    "phone_phoneservice",
    "phone_multiplelines",
    "internet_internetservice",
    "internet_onlinesecurity",
    "internet_onlinebackup",
    "internet_deviceprotection",
    "internet_techsupport",
    "internet_streamingtv",
    "internet_streamingmovies",
    "account_contract",
    "account_paperlessbilling",
    "account_paymentmethod",
    "account_charges_monthly",
    "account_charges_total",
    "cuentas_diarias",
}


def load_data(
    local_path: Path | str | None = None,
    url: str | None = None,
) -> pd.DataFrame:
    """Carga el dataset de Telecom X.

    Intenta primero la ruta local; si no existe, descarga desde la URL.

    Parameters
    ----------
    local_path : Path | str | None
        Ruta al archivo CSV local. Por defecto ``LOCAL_DATA_PATH``.
    url : str | None
        URL de fallback. Por defecto ``DATA_URL``.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.
    """
    local_path = Path(local_path) if local_path else LOCAL_DATA_PATH
    url = url or DATA_URL

    if local_path.exists():
        df = pd.read_csv(local_path)
        print(f"📂 Datos cargados desde: {local_path}")
    else:
        df = pd.read_csv(url)
        print(f"🌐 Datos cargados desde URL (archivo local no encontrado)")

    print(
        f"📊 Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas"
    )

    # Validación del esquema
    actual_cols = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_cols
    extra = actual_cols - EXPECTED_COLUMNS

    if missing:
        print(f"⚠️  Columnas esperadas no encontradas: {sorted(missing)}")
    if extra:
        print(f"ℹ️  Columnas adicionales detectadas: {sorted(extra)}")

    # Verificar que no haya valores nulos inesperados
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"⚠️  Columnas con valores nulos: {dict(null_cols)}")
    else:
        print("✅ Validación: esquema OK, sin valores nulos.")

    return df
