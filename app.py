"""
Telecom X — Predicción de Churn (Cancelación de Clientes)

Aplicación Gradio que despliega dos modelos de Machine Learning:
  • Regresión Logística (20 features seleccionadas)
  • Random Forest (30 features completas)

Los modelos fueron entrenados en el notebook 03_modelado_predictivo.ipynb
y se encuentran serializados en la carpeta models/.
"""

import pickle
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Carga de modelos y scaler al inicio
# Los .pkl fueron serializados con joblib (usa pickle + compresión LZ4),
# por lo que se cargan con joblib.load() que es compatible con pickle.
# ---------------------------------------------------------------------------
lr_model = joblib.load(MODELS_DIR / "logistic_regression.pkl")
rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
scaler   = joblib.load(MODELS_DIR / "scaler_modelado.pkl")

# ---------------------------------------------------------------------------
# Configuración de features (debe coincidir con el entrenamiento)
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "customer_seniorcitizen", "customer_partner", "customer_dependents",
    "customer_tenure", "phone_phoneservice", "account_paperlessbilling",
    "account_charges_monthly", "account_charges_total",
    "customer_gender_Male",
    "phone_multiplelines_No phone service", "phone_multiplelines_Yes",
    "internet_internetservice_Fiber optic", "internet_internetservice_No",
    "internet_onlinesecurity_No internet service", "internet_onlinesecurity_Yes",
    "internet_onlinebackup_No internet service", "internet_onlinebackup_Yes",
    "internet_deviceprotection_No internet service", "internet_deviceprotection_Yes",
    "internet_techsupport_No internet service", "internet_techsupport_Yes",
    "internet_streamingtv_No internet service", "internet_streamingtv_Yes",
    "internet_streamingmovies_No internet service", "internet_streamingmovies_Yes",
    "account_contract_One year", "account_contract_Two year",
    "account_paymentmethod_Credit card (automatic)",
    "account_paymentmethod_Electronic check",
    "account_paymentmethod_Mailed check",
]

LR_FEATURES = [
    "account_paymentmethod_Electronic check",
    "internet_internetservice_Fiber optic",
    "account_paperlessbilling", "account_charges_monthly",
    "customer_seniorcitizen", "customer_partner", "customer_dependents",
    "internet_onlinesecurity_Yes", "internet_techsupport_Yes",
    "account_contract_One year", "account_charges_total",
    "internet_internetservice_No",
    "internet_streamingtv_No internet service",
    "internet_onlinesecurity_No internet service",
    "internet_onlinebackup_No internet service",
    "internet_deviceprotection_No internet service",
    "internet_streamingmovies_No internet service",
    "internet_techsupport_No internet service",
    "account_contract_Two year", "customer_tenure",
]


# ---------------------------------------------------------------------------
# Función auxiliar: convierte inputs de la UI a DataFrame codificado + escalado
# ---------------------------------------------------------------------------
def _build_features(
    gender: str,
    senior_citizen: str,
    partner: str,
    dependents: str,
    tenure: int,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless_billing: str,
    payment_method: str,
    monthly_charges: float,
    total_charges: float,
) -> pd.DataFrame:
    """Convierte los valores de la UI a un DataFrame con las 30 features
    codificadas y escaladas, listo para `.predict()`."""

    # --- Mapeo binario ---
    si_no = {"Sí": 1, "No": 0}

    row = {
        "customer_seniorcitizen": si_no[senior_citizen],
        "customer_partner":       si_no[partner],
        "customer_dependents":    si_no[dependents],
        "customer_tenure":        tenure,
        "phone_phoneservice":     si_no[phone_service],
        "account_paperlessbilling": si_no[paperless_billing],
        "account_charges_monthly": monthly_charges,
        "account_charges_total":   total_charges,
        # --- One-Hot: customer_gender (drop_first='Female') ---
        "customer_gender_Male": 1 if gender == "Masculino" else 0,
        # --- One-Hot: phone_multiplelines (drop_first='No') ---
        "phone_multiplelines_No phone service": 1 if multiple_lines == "Sin servicio telefónico" else 0,
        "phone_multiplelines_Yes": 1 if multiple_lines == "Sí" else 0,
        # --- One-Hot: internet_internetservice (drop_first='DSL') ---
        "internet_internetservice_Fiber optic": 1 if internet_service == "Fibra óptica" else 0,
        "internet_internetservice_No":          1 if internet_service == "Sin internet" else 0,
    }

    # --- One-Hot: servicios de internet (drop_first='No') ---
    internet_services = {
        "internet_onlinesecurity":    online_security,
        "internet_onlinebackup":      online_backup,
        "internet_deviceprotection":  device_protection,
        "internet_techsupport":       tech_support,
        "internet_streamingtv":       streaming_tv,
        "internet_streamingmovies":   streaming_movies,
    }
    for prefix, value in internet_services.items():
        row[f"{prefix}_No internet service"] = 1 if value == "Sin internet" else 0
        row[f"{prefix}_Yes"]                 = 1 if value == "Sí" else 0

    # --- One-Hot: account_contract (drop_first='Month-to-month') ---
    row["account_contract_One year"]  = 1 if contract == "Un año" else 0
    row["account_contract_Two year"]  = 1 if contract == "Dos años" else 0

    # --- One-Hot: account_paymentmethod (drop_first='Bank transfer') ---
    row["account_paymentmethod_Credit card (automatic)"] = 1 if payment_method == "Tarjeta de crédito" else 0
    row["account_paymentmethod_Electronic check"]        = 1 if payment_method == "Cheque electrónico" else 0
    row["account_paymentmethod_Mailed check"]            = 1 if payment_method == "Cheque por correo" else 0

    # Construir DataFrame con el orden exacto del entrenamiento
    df = pd.DataFrame([row], columns=ALL_FEATURES)

    # Escalar con el mismo StandardScaler usado en entrenamiento
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=ALL_FEATURES,
    )
    return df_scaled


# ---------------------------------------------------------------------------
# Funciones de predicción
# ---------------------------------------------------------------------------
def predict_logistic_regression(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
) -> str:
    """Predice churn con Regresión Logística (20 features seleccionadas)."""
    df = _build_features(
        gender, senior_citizen, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges,
    )
    X = df[LR_FEATURES]
    pred = lr_model.predict(X)[0]
    proba = lr_model.predict_proba(X)[0]

    label = "⚠️ CANCELA" if pred == 1 else "✅ NO cancela"
    return (
        f"Predicción: {label}\n\n"
        f"Probabilidad de cancelación: {proba[1]:.1%}\n"
        f"Probabilidad de permanencia: {proba[0]:.1%}"
    )


def predict_random_forest(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
) -> str:
    """Predice churn con Random Forest (30 features completas)."""
    df = _build_features(
        gender, senior_citizen, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges,
    )
    X = df[ALL_FEATURES]
    pred = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0]

    label = "⚠️ CANCELA" if pred == 1 else "✅ NO cancela"
    return (
        f"Predicción: {label}\n\n"
        f"Probabilidad de cancelación: {proba[1]:.1%}\n"
        f"Probabilidad de permanencia: {proba[0]:.1%}"
    )


# ---------------------------------------------------------------------------
# Componentes de entrada reutilizables (una función por pestaña)
# ---------------------------------------------------------------------------
def _create_inputs() -> list:
    """Crea y devuelve la lista de componentes de entrada para una pestaña."""

    gr.Markdown("### 👤 Datos del Cliente")
    with gr.Row():
        gender = gr.Radio(
            ["Femenino", "Masculino"],
            label="Género", value="Femenino",
        )
        senior_citizen = gr.Radio(
            ["No", "Sí"],
            label="Adulto mayor (+65)", value="No",
        )
    with gr.Row():
        partner = gr.Radio(
            ["No", "Sí"], label="Tiene pareja", value="No",
        )
        dependents = gr.Radio(
            ["No", "Sí"], label="Tiene dependientes", value="No",
        )
    tenure = gr.Slider(
        minimum=0, maximum=72, step=1, value=12,
        label="Antigüedad (meses)",
    )

    gr.Markdown("### 📞 Servicio Telefónico")
    with gr.Row():
        phone_service = gr.Radio(
            ["No", "Sí"], label="Servicio telefónico", value="Sí",
        )
        multiple_lines = gr.Radio(
            ["No", "Sí", "Sin servicio telefónico"],
            label="Múltiples líneas", value="No",
        )

    gr.Markdown("### 🌐 Servicio de Internet")
    internet_service = gr.Radio(
        ["DSL", "Fibra óptica", "Sin internet"],
        label="Tipo de servicio de internet", value="DSL",
    )
    with gr.Row():
        online_security = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Seguridad en línea", value="No",
        )
        online_backup = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Respaldo en línea", value="No",
        )
    with gr.Row():
        device_protection = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Protección de dispositivos", value="No",
        )
        tech_support = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Soporte técnico", value="No",
        )
    with gr.Row():
        streaming_tv = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Streaming de TV", value="No",
        )
        streaming_movies = gr.Radio(
            ["No", "Sí", "Sin internet"],
            label="Streaming de películas", value="No",
        )

    gr.Markdown("### 💳 Cuenta")
    contract = gr.Radio(
        ["Mensual", "Un año", "Dos años"],
        label="Tipo de contrato", value="Mensual",
    )
    with gr.Row():
        paperless_billing = gr.Radio(
            ["No", "Sí"], label="Facturación digital", value="No",
        )
        payment_method = gr.Dropdown(
            [
                "Transferencia bancaria",
                "Tarjeta de crédito",
                "Cheque electrónico",
                "Cheque por correo",
            ],
            label="Método de pago", value="Transferencia bancaria",
        )
    with gr.Row():
        monthly_charges = gr.Number(
            label="Cargo mensual (USD)", value=50.0,
            minimum=18.25, maximum=118.75,
        )
        total_charges = gr.Number(
            label="Cargo total acumulado (USD)", value=600.0,
            minimum=0.0, maximum=8685.0,
        )

    return [
        gender, senior_citizen, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges,
    ]


# ---------------------------------------------------------------------------
# Interfaz Gradio con Tabs
# ---------------------------------------------------------------------------
with gr.Blocks(title="Telecom X — Predicción de Churn") as demo:

    gr.Markdown(
        "# 📡 Telecom X — Predicción de Cancelación de Clientes\n"
        "Ingrese los datos del cliente para predecir si cancelará su servicio. "
        "Puede elegir entre dos modelos de Machine Learning entrenados sobre "
        "datos históricos de Telecom X."
    )

    # ── Tab 1: Regresión Logística ──────────────────────────────────────
    with gr.Tab("🔵 Regresión Logística"):
        gr.Markdown(
            "> **Modelo lineal** entrenado con las **20 features más relevantes** "
            "seleccionadas por correlación de Pearson ($|r| \\geq 0.15$). "
            "Ofrece alta interpretabilidad y buen recall para detectar churners."
        )
        inputs_lr = _create_inputs()
        btn_lr = gr.Button("Predecir con Regresión Logística", variant="primary")
        output_lr = gr.Textbox(label="Resultado de la predicción", lines=4)
        btn_lr.click(fn=predict_logistic_regression, inputs=inputs_lr, outputs=output_lr)

    # ── Tab 2: Random Forest ────────────────────────────────────────────
    with gr.Tab("🌲 Random Forest"):
        gr.Markdown(
            "> **Ensemble de 100 árboles de decisión** entrenado con las "
            "**30 features completas**. Captura relaciones no lineales e "
            "interacciones entre variables."
        )
        inputs_rf = _create_inputs()
        btn_rf = gr.Button("Predecir con Random Forest", variant="primary")
        output_rf = gr.Textbox(label="Resultado de la predicción", lines=4)
        btn_rf.click(fn=predict_random_forest, inputs=inputs_rf, outputs=output_rf)

# ---------------------------------------------------------------------------
# Lanzamiento
# ---------------------------------------------------------------------------
demo.launch(theme=gr.themes.Soft())
