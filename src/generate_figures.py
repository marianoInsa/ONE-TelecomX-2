"""
Generador de figuras para el README.

Genera las 5 visualizaciones clave del proyecto como archivos PNG
en ``docs/images/``, listas para ser referenciadas desde el README.md.

Ejecutar desde la raíz del proyecto:
    python src/generate_figures.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Asegurar que src/ es importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BACKGROUND_COLOR, setup_plot_style
from src.visualization import (
    plot_charges_analysis,
    plot_confusion_matrix,
    plot_correlation_bars,
    plot_importance_comparison,
    plot_metrics_comparison,
)

# ── Configuración ──────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"
DPI = 150
SAVE_KWARGS = dict(
    dpi=DPI,
    bbox_inches="tight",
    facecolor=BACKGROUND_COLOR,
    edgecolor="none",
    pad_inches=0.3,
)


def _load_eval_results() -> list[dict]:
    """Carga los resultados de evaluación y devuelve la lista de modelos."""
    path = PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["models"], data.get("feature_config", {})


def generate_correlation_bars() -> None:
    """Fig 1 — Barras horizontales de correlación con Churn."""
    print("  [1/5] Generando barras de correlación...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "telecom_encoded.csv")
    corr = df.corr(numeric_only=True)["Churn"].drop("Churn")

    fig = plot_correlation_bars(corr, top_n=15, show=False)
    fig.savefig(OUTPUT_DIR / "01_correlation_bars.png", **SAVE_KWARGS)
    plt.close(fig)


def generate_metrics_comparison(eval_results: list[dict]) -> None:
    """Fig 2 — Barras agrupadas comparando métricas de test."""
    print("  [2/5] Generando comparación de métricas...")
    fig = plot_metrics_comparison(eval_results, show=False)
    fig.savefig(OUTPUT_DIR / "02_metrics_comparison.png", **SAVE_KWARGS)
    plt.close(fig)


def generate_confusion_matrices(eval_results: list[dict]) -> None:
    """Fig 3 — Matrices de confusión lado a lado."""
    print("  [3/5] Generando matrices de confusión...")
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Matrices de Confusión — Test Set",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, result in zip(axes, eval_results):
        cm = np.array(result["test"]["cm"])
        plot_confusion_matrix(cm, result["model"], ax=ax, show=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_confusion_matrices.png", **SAVE_KWARGS)
    plt.close(fig)


def generate_importance_comparison(feature_config: dict) -> None:
    """Fig 4 — Dot-chart comparando rankings de importancia LR vs RF."""
    print("  [4/5] Generando comparación de importancia...")

    lr_features = feature_config["lr_features"]
    rf_features = feature_config["rf_features"]

    # Cargar modelos para extraer importancias
    lr_model = joblib.load(PROJECT_ROOT / "models" / "logistic_regression.pkl")
    rf_model = joblib.load(PROJECT_ROOT / "models" / "random_forest.pkl")

    lr_coefs = lr_model.coef_[0]
    rf_importances = rf_model.feature_importances_

    fig = plot_importance_comparison(
        lr_coefs,
        lr_features,
        rf_importances,
        rf_features,
        top_n=15,
        show=False,
    )
    fig.savefig(OUTPUT_DIR / "04_importance_comparison.png", **SAVE_KWARGS)
    plt.close(fig)


def generate_charges_analysis() -> None:
    """Fig 5 — Boxplot + Scatter de cargos por clase."""
    print("  [5/5] Generando análisis de cargos...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "telecom_encoded.csv")
    fig = plot_charges_analysis(df, show=False)
    fig.savefig(OUTPUT_DIR / "05_charges_analysis.png", **SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    """Genera las 5 figuras clave del proyecto."""
    print(f"Generando figuras en {OUTPUT_DIR}/\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    eval_results, feature_config = _load_eval_results()

    generate_correlation_bars()
    generate_metrics_comparison(eval_results)
    generate_confusion_matrices(eval_results)
    generate_importance_comparison(feature_config)
    generate_charges_analysis()

    # Verificar resultados
    generated = list(OUTPUT_DIR.glob("*.png"))
    print(f"\n✅ {len(generated)} figuras generadas:")
    for p in sorted(generated):
        size_kb = p.stat().st_size / 1024
        print(f"   {p.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
