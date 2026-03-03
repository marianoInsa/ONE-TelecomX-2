"""
Módulo de visualización.

Todas las funciones de gráficos del proyecto, con estilo cyberpunk
consistente. Cada función genera una figura completa lista para mostrar.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mplcyberpunk
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import CMAP_CYBER, COLOR_PALETTE, BACKGROUND_COLOR


# Helpers internos
def _annotate_bars(
    ax: plt.Axes,
    bars,
    values,
    fmt: str = "{:,}",
    offset_pct: float = 0.02,
    **text_kwargs,
) -> None:
    """Añade etiquetas de valor sobre cada barra."""
    defaults = dict(ha="center", va="bottom", fontweight="bold", fontsize=12, color="white")
    defaults.update(text_kwargs)
    y_max = max(abs(v) for v in values) if values is not None else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * offset_pct,
            fmt.format(val),
            **defaults,
        )


# Distribución de clases
def plot_class_distribution(
    class_counts: pd.Series,
    class_pct: pd.Series,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Gráfico de barras + pie chart de la distribución de la variable objetivo.

    Parameters
    ----------
    class_counts : pd.Series
        Conteo absoluto por clase.
    class_pct : pd.Series
        Porcentaje por clase.
    colors : list[str] | None
        Dos colores [clase_0, clase_1]. Por defecto los primeros de la paleta.
    labels : list[str] | None
        Etiquetas para cada clase.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1]]
    labels = labels or ["No canceló (0)", "Canceló (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Distribución de Clases | Variable Objetivo: Churn",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Barras con porcentaje
    bars = axes[0].bar(labels, class_pct.values, color=colors, width=0.5, edgecolor="none", zorder=2)
    axes[0].set_title("Proporción por clase", fontweight="bold")
    axes[0].set_ylabel("Porcentaje de clientes (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(0, 90)
    for bar, pct, cnt in zip(bars, class_pct.values, class_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%\n({cnt:,})",
            ha="center", va="bottom", fontweight="bold", fontsize=11, color="white",
        )

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        class_counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": BACKGROUND_COLOR, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    for t in texts:
        t.set_color("white")
    axes[1].set_title("Distribución relativa", fontweight="bold")

    mplcyberpunk.add_bar_gradient(bars=bars, ax=axes[0])
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Comparación antes/después de SMOTE
def plot_smote_comparison(
    y_before: pd.Series,
    y_after: pd.Series,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Barras comparando la distribución de clases antes y después de SMOTE.

    Parameters
    ----------
    y_before : pd.Series
        Variable objetivo antes del balanceo (train original).
    y_after : pd.Series
        Variable objetivo después de SMOTE.
    colors : list[str] | None
        Dos colores [clase_0, clase_1].
    labels : list[str] | None
        Etiquetas para cada clase.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1]]
    labels = labels or ["No canceló (0)", "Canceló (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Efecto del Balanceo | Antes vs Después de SMOTE (Train Set)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (counts, title) in zip(axes, [
        (y_before.value_counts(), "Antes del balanceo"),
        (pd.Series(y_after).value_counts(), "Después de SMOTE"),
    ]):
        bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="none", zorder=2)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Número de registros")
        ax.set_ylim(0, max(counts.values) * 1.25)
        _annotate_bars(ax, bars, counts.values)
        mplcyberpunk.add_bar_gradient(bars=bars, ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Comparación antes/después del escalado
def plot_scaling_comparison(
    X_original: pd.DataFrame,
    X_scaled: np.ndarray | pd.DataFrame,
    columns: list[str],
    col_indices: list[int] | None = None,
    colors: list[str] | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Histogramas 2×N comparando distribuciones antes y después del escalado.

    Incluye líneas verticales de media y ±1σ con anotaciones numéricas
    para que la diferencia de escala sea visualmente evidente.

    Parameters
    ----------
    X_original : pd.DataFrame
        Datos originales (sin escalar).
    X_scaled : np.ndarray | pd.DataFrame
        Datos escalados.
    columns : list[str]
        Nombres de las columnas a graficar.
    col_indices : list[int] | None
        Índices correspondientes en ``X_scaled``. Si ``X_scaled`` es un
        DataFrame, se ignoran y se usan los nombres de columna.
    colors : list[str] | None
        Un color por columna.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1], COLOR_PALETTE[2]]
    n = len(columns)

    fig, axes = plt.subplots(2, n, figsize=(16, 8))
    fig.suptitle(
        "Distribución de Variables Numéricas | Antes vs Después del Escalado",
        fontsize=13, fontweight="bold",
    )

    for i, (col, color) in enumerate(zip(columns, colors)):
        orig_col = X_original[col].values

        # Resolver datos escalados por nombre de columna o índice
        if isinstance(X_scaled, pd.DataFrame):
            scaled_col = X_scaled[col].values
        else:
            idx = col_indices[i] if col_indices is not None else i
            scaled_col = X_scaled[:, idx]

        # Estadísticas originales
        orig_mean, orig_std = orig_col.mean(), orig_col.std()
        # Estadísticas escaladas
        sc_mean, sc_std = scaled_col.mean(), scaled_col.std()

        # Fila superior: datos originales
        ax_before = axes[0, i]
        ax_before.hist(orig_col, bins=35, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        # Líneas de media y ±1σ
        ax_before.axvline(orig_mean, color=COLOR_PALETTE[3], ls="--", lw=1.8, label=f"μ = {orig_mean:,.1f}")
        ax_before.axvline(orig_mean - orig_std, color=COLOR_PALETTE[2], ls=":", lw=1.4, label=f"−1σ = {orig_mean - orig_std:,.1f}")
        ax_before.axvline(orig_mean + orig_std, color=COLOR_PALETTE[2], ls=":", lw=1.4, label=f"+1σ = {orig_mean + orig_std:,.1f}")
        ax_before.set_title(f"Antes | {col}", fontsize=10, fontweight="bold")
        ax_before.set_ylabel("Frecuencia" if i == 0 else "")
        ax_before.set_xlabel("Valor original")
        ax_before.legend(fontsize=7, loc="upper right", framealpha=0.6)

        # Fila inferior: datos escalados
        ax_after = axes[1, i]
        ax_after.hist(scaled_col, bins=35, color=color, alpha=0.35, edgecolor="none")
        ax_after.hist(scaled_col, bins=35, color=color, alpha=1.0, histtype="step", linewidth=1.8)
        # Líneas de media y ±1σ
        ax_after.axvline(sc_mean, color=COLOR_PALETTE[3], ls="--", lw=1.8, label=f"μ = {sc_mean:.2f}")
        ax_after.axvline(sc_mean - sc_std, color=COLOR_PALETTE[2], ls=":", lw=1.4, label=f"−1σ = {sc_mean - sc_std:.2f}")
        ax_after.axvline(sc_mean + sc_std, color=COLOR_PALETTE[2], ls=":", lw=1.4, label=f"+1σ = {sc_mean + sc_std:.2f}")
        ax_after.set_title(f"Después | {col}", fontsize=10, fontweight="bold")
        ax_after.set_ylabel("Frecuencia" if i == 0 else "")
        ax_after.set_xlabel("Valor estandarizado (z-score)")
        ax_after.legend(fontsize=7, loc="upper right", framealpha=0.6)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Heatmap de correlación
def plot_correlation_heatmap(
    df_encoded: pd.DataFrame,
    corr_with_target: pd.Series,
    top_n: int = 15,
    target: str = "Churn",
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Heatmap triangular de las Top N features más correlacionadas con el target.

    Parameters
    ----------
    df_encoded : pd.DataFrame
        Dataset codificado completo.
    corr_with_target : pd.Series
        Correlaciones de cada feature con el target.
    top_n : int
        Número de features a incluir.
    target : str
        Nombre de la columna objetivo.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    top_features = corr_with_target.abs().sort_values(ascending=False).head(top_n).index.tolist()
    features_with_target = top_features + [target]
    corr_top = df_encoded[features_with_target].corr()

    mask = np.zeros_like(corr_top, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        f"Correlación de Pearson | Top {top_n} Features + {target}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    sns.heatmap(
        corr_top,
        mask=mask,
        cmap=CMAP_CYBER,
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": "white"},
        linewidths=0.5,
        linecolor="#000000",
        cbar_kws={"shrink": 0.7, "label": "Coef. de Pearson"},
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Barras horizontales de correlación
def plot_correlation_bars(
    corr_series: pd.Series,
    top_n: int = 15,
    colors: list[str] | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Barras horizontales de las Top N features por correlación absoluta.

    Parameters
    ----------
    corr_series : pd.Series
        Correlaciones con el target.
    top_n : int
        Número de features a mostrar.
    colors : list[str] | None
        ``[color_positivo, color_negativo]``.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1]]

    top_abs_idx = corr_series.abs().sort_values(ascending=False).head(top_n).index
    top_corr_plot = corr_series.reindex(top_abs_idx).sort_values()

    colors_bar = [colors[0] if v > 0 else colors[1] for v in top_corr_plot.values]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(
        f"Top {top_n} Variables con Mayor Correlación con Churn",
        fontsize=14, fontweight="bold", y=1.01,
    )

    bars = ax.barh(
        top_corr_plot.index, top_corr_plot.values,
        color=colors_bar, edgecolor="none", height=0.65, zorder=2,
    )
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Coeficiente de Correlación de Pearson", fontsize=11)
    ax.set_xlim(-0.43, 0.43)

    for bar, val in zip(bars, top_corr_plot.values):
        if val >= 0:
            ax.text(
                val - 0.006, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="right",
                fontsize=9, fontweight="bold", color="white",
            )
        else:
            ax.text(
                val + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left",
                fontsize=9, fontweight="bold", color="white",
            )

    legend_elements = [
        Patch(facecolor=colors[0], label="Correlación positiva (favorece churn)"),
        Patch(facecolor=colors[1], label="Correlación negativa (protege contra churn)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    mplcyberpunk.add_bar_gradient(bars=bars, horizontal=True)
    mplcyberpunk.make_lines_glow(ax)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Boxplots por variable objetivo
def plot_boxplot_by_target(
    df: pd.DataFrame,
    columns: list[str],
    target: str = "Churn",
    colors: list[str] | None = None,
    ylabels: list[str] | None = None,
    suptitle: str | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Boxplots lado a lado comparando distribuciones por clase del target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con las columnas y el target.
    columns : list[str]
        Columnas numéricas a graficar (una por subplot).
    target : str
        Columna de la variable objetivo binaria (0/1).
    colors : list[str] | None
        ``[color_clase_0, color_clase_1]``.
    ylabels : list[str] | None
        Etiquetas para el eje Y de cada subplot.
    suptitle : str | None
        Título general de la figura.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1]]
    ylabels = ylabels or columns
    n = len(columns)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)

    for ax, col, ylabel in zip(axes, columns, ylabels):
        data_0 = df.loc[df[target] == 0, col]
        data_1 = df.loc[df[target] == 1, col]

        bp = ax.boxplot(
            [data_0, data_1],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color="white", linewidth=1.5),
            capprops=dict(color="white", linewidth=1.8),
            flierprops=dict(marker="o", color=COLOR_PALETTE[2], alpha=0.5, markersize=4),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
            patch.set_linewidth(1.5)

        # Anotación de medianas
        ref_range = data_0.max() - data_0.min()
        for med, x_pos in zip([data_0.median(), data_1.median()], [1, 2]):
            ax.text(
                x_pos, med + ref_range * 0.04,
                f"Med: {med:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="white",
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No canceló (0)", "Canceló (1)"], fontsize=10)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel)
        mplcyberpunk.make_lines_glow(ax)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# Boxplot + Scatter de cargos totales
def plot_charges_analysis(
    df: pd.DataFrame,
    target: str = "Churn",
    colors: list[str] | None = None,
    *,
    show: bool = True,
) -> plt.Figure | None:
    """Boxplot de gasto total + scatter mensual vs total, coloreado por Churn.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con ``account_charges_total``, ``account_charges_monthly`` y ``target``.
    target : str
        Columna objetivo.
    colors : list[str] | None
        ``[color_clase_0, color_clase_1]``.
    show : bool
        Si True, muestra la figura con ``plt.show()``.

    Returns
    -------
    plt.Figure | None
        La figura creada, o ``None`` si ``show=True`` (ya fue mostrada).
    """
    colors = colors or [COLOR_PALETTE[0], COLOR_PALETTE[1]]
    c0, c1 = colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gasto Total Acumulado × Cancelación", fontsize=14, fontweight="bold", y=1.01)

    # Boxplot
    data_0 = df.loc[df[target] == 0, "account_charges_total"]
    data_1 = df.loc[df[target] == 1, "account_charges_total"]

    bp = axes[0].boxplot(
        [data_0, data_1],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color="white", linewidth=1.5),
        capprops=dict(color="white", linewidth=1.8),
        flierprops=dict(marker="o", color=COLOR_PALETTE[2], alpha=0.5, markersize=4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.5)

    ref_range = data_0.max() - data_0.min()
    for med, x_pos in zip([data_0.median(), data_1.median()], [1, 2]):
        axes[0].text(
            x_pos, med + ref_range * 0.025,
            f"Med: ${med:,.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="white",
        )

    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["No canceló (0)", "Canceló (1)"], fontsize=10)
    axes[0].set_title("Gasto Total Acumulado (USD)", fontweight="bold")
    axes[0].set_ylabel("account_charges_total (USD)")
    mplcyberpunk.make_lines_glow(axes[0])

    # Scatter
    x_sc = df["account_charges_monthly"]
    y_sc = df["account_charges_total"]
    mask_0 = df[target] == 0
    mask_1 = df[target] == 1

    # Halo de glow manual
    for i in range(1, 8):
        axes[1].scatter(x_sc[mask_0], y_sc[mask_0], color=c0, alpha=0.03, s=12 * (1.3**i), edgecolors="none")
        axes[1].scatter(x_sc[mask_1], y_sc[mask_1], color=c1, alpha=0.03, s=12 * (1.3**i), edgecolors="none")

    # Puntos principales
    axes[1].scatter(x_sc[mask_0], y_sc[mask_0], color=c0, alpha=0.35, s=12, edgecolors="none", zorder=3)
    axes[1].scatter(x_sc[mask_1], y_sc[mask_1], color=c1, alpha=0.35, s=12, edgecolors="none", zorder=3)

    axes[1].set_xlabel("Cargo Mensual (USD)", fontsize=10)
    axes[1].set_ylabel("Gasto Total Acumulado (USD)", fontsize=10)
    axes[1].set_title("Cargo Mensual vs Gasto Total", fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c0, markersize=8, label="No canceló (0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c1, markersize=8, label="Canceló (1)"),
    ]
    axes[1].legend(handles=legend_handles, fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Funciones de evaluación de modelos
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm,
    model_name: str = "",
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """Heatmap anotado de una confusion matrix.

    Parameters
    ----------
    cm : array-like, shape (2, 2)
        Confusion matrix (sklearn format).
    model_name : str
        Nombre para el título del gráfico.
    ax : Axes, optional
        Axes sobre el que dibujar. Si es ``None`` se crea una figura nueva.
    show : bool
        Si ``True``, muestra y cierra la figura. Si ``False``, retorna el
        ``Figure``.
    """
    plt.style.use("cyberpunk")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    labels = np.array(cm)
    # Porcentajes por fila (clase real)
    row_sums = labels.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, labels / row_sums * 100, 0)

    annot_text = np.array(
        [
            [f"{labels[i, j]:,}\n({pct[i, j]:.1f}%)" for j in range(labels.shape[1])]
            for i in range(labels.shape[0])
        ]
    )

    sns.heatmap(
        labels,
        annot=annot_text,
        fmt="",
        cmap=CMAP_CYBER,
        cbar=True,
        linewidths=1.5,
        linecolor=BACKGROUND_COLOR,
        ax=ax,
        xticklabels=["No Churn (0)", "Churn (1)"],
        yticklabels=["No Churn (0)", "Churn (1)"],
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )

    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    title = "Matriz de Confusión"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)

    plt.tight_layout()
    if show and own_fig:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_metrics_comparison(
    eval_results: list[dict],
    *,
    show: bool = True,
):
    """Barras agrupadas comparando Accuracy/Precision/Recall/F1 (test).

    Parameters
    ----------
    eval_results : list[dict]
        Lista de dicts retornados por ``evaluate_model()``.
    show : bool
        Si ``True``, muestra la figura.
    """
    plt.style.use("cyberpunk")

    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    n_metrics = len(metrics_keys)
    n_models = len(eval_results)

    x = np.arange(n_metrics)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for idx, result in enumerate(eval_results):
        values = [result["test"][k] for k in metrics_keys]
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width * 0.9,
            label=result["model"],
            color=COLOR_PALETTE[idx],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Comparación de Métricas — Test Set", fontweight="bold", fontsize=14, pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    mplcyberpunk.add_glow_effects(ax)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_overfit_analysis(
    eval_results: list[dict],
    *,
    show: bool = True,
):
    """Gráfico de gap Train vs Test para diagnóstico de overfitting.

    Muestra las métricas de Train y Test side-by-side para cada modelo,
    facilitando la detección visual de overfitting (gap grande) o
    underfitting (ambos valores bajos).

    Parameters
    ----------
    eval_results : list[dict]
        Lista de dicts retornados por ``evaluate_model()`` **con datos
        de train incluidos**.
    show : bool
        Si ``True``, muestra la figura.
    """
    plt.style.use("cyberpunk")

    # Filtrar modelos que tengan métricas de train
    results_with_train = [r for r in eval_results if r.get("train") is not None]
    if not results_with_train:
        print("⚠️ No hay métricas de train disponibles para el análisis de overfitting.")
        return None

    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1"]
    n_models = len(results_with_train)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results_with_train):
        train_vals = [result["train"][k] for k in metrics_keys]
        test_vals = [result["test"][k] for k in metrics_keys]
        gaps = [t - s for t, s in zip(train_vals, test_vals)]

        x = np.arange(len(metrics_keys))
        w = 0.32

        bars_train = ax.bar(
            x - w / 2, train_vals, w, label="Train", color=COLOR_PALETTE[2],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        bars_test = ax.bar(
            x + w / 2, test_vals, w, label="Test", color=COLOR_PALETTE[3],
            edgecolor="white", linewidth=0.5, zorder=3,
        )

        # Anotar gaps
        for i, (tv, sv, gap) in enumerate(zip(train_vals, test_vals, gaps)):
            higher = max(tv, sv)
            sign = "+" if gap > 0 else ""
            color = "#ff4444" if abs(gap) > 0.05 else "#44ff44"
            ax.text(
                x[i], higher + 0.025,
                f"Δ {sign}{gap:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(result["model"], fontweight="bold", fontsize=13, pad=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Análisis de Overfitting — Train vs Test",
        fontweight="bold", fontsize=14, y=1.02,
    )
    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ===================================================================
# Importancia de variables
# ===================================================================


def plot_feature_importance(
    importances,
    feature_names: list[str],
    model_name: str = "",
    *,
    top_n: int = 15,
    show: bool = True,
):
    """Barras horizontales con las features más relevantes.

    Funciona con coeficientes de LR (pueden ser negativos) y con
    importancias de RF (siempre ≥ 0). Las features se ordenan por
    magnitud absoluta.

    Parameters
    ----------
    importances : array-like
        Vector de importancias o coeficientes (longitud = nº features).
    feature_names : list[str]
        Nombres de las features correspondientes.
    model_name : str
        Nombre del modelo para el título.
    top_n : int
        Cuántas features mostrar.
    show : bool
        Si ``True``, muestra la figura.
    """
    plt.style.use("cyberpunk")

    imp = np.asarray(importances)
    names = np.asarray(feature_names)

    # Ordenar por magnitud absoluta descendente y tomar top_n
    order = np.argsort(np.abs(imp))[::-1][:top_n]
    # Invertir para que la barra más importante quede arriba
    order = order[::-1]

    top_imp = imp[order]
    top_names = names[order]

    has_negative = np.any(top_imp < 0)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))

    if has_negative:
        # Coeficientes: color diferente para positivo / negativo
        colors = [COLOR_PALETTE[4] if v >= 0 else COLOR_PALETTE[0] for v in top_imp]
    else:
        colors = [COLOR_PALETTE[0]] * len(top_imp)

    bars = ax.barh(range(len(top_imp)), top_imp, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Anotar valores
    for bar, val in zip(bars, top_imp):
        offset = 0.002 * np.sign(val) if has_negative else 0.002
        ha = "left" if val >= 0 else "right"
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha=ha, fontsize=9, fontweight="bold", color="white",
        )

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Coeficiente" if has_negative else "Importancia", fontsize=11)

    title = f"Top {len(top_imp)} Features"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)

    if has_negative:
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        legend_handles = [
            Patch(facecolor=COLOR_PALETTE[4], label="↑ Aumenta prob. churn"),
            Patch(facecolor=COLOR_PALETTE[0], label="↓ Reduce prob. churn"),
        ]
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_importance_comparison(
    lr_importances,
    lr_features: list[str],
    rf_importances,
    rf_features: list[str],
    *,
    top_n: int = 15,
    show: bool = True,
):
    """Dot-chart comparando los rankings de importancia de dos modelos.

    Muestra las top-N features de cada modelo en un gráfico unificado,
    resaltando las que son relevantes para **ambos** modelos.

    Parameters
    ----------
    lr_importances : array-like
        Coeficientes (magnitud absoluta) de LR.
    lr_features : list[str]
        Nombres de las features de LR.
    rf_importances : array-like
        Importancias de RF.
    rf_features : list[str]
        Nombres de las features de RF.
    top_n : int
        Cuántas features tomar de cada modelo.
    show : bool
        Si ``True``, muestra la figura.
    """
    plt.style.use("cyberpunk")

    # Construir rankings
    lr_imp = np.abs(np.asarray(lr_importances))
    rf_imp = np.asarray(rf_importances)

    lr_order = np.argsort(lr_imp)[::-1][:top_n]
    rf_order = np.argsort(rf_imp)[::-1][:top_n]

    lr_top_set = set(np.asarray(lr_features)[lr_order])
    rf_top_set = set(np.asarray(rf_features)[rf_order])

    common = sorted(lr_top_set & rf_top_set)
    only_lr = sorted(lr_top_set - rf_top_set)
    only_rf = sorted(rf_top_set - lr_top_set)

    # Unificar en un solo set ordenado
    all_features = common + only_lr + only_rf

    # Normalizar importancias a [0, 1] para comparabilidad
    lr_dict = dict(zip(np.asarray(lr_features), lr_imp / lr_imp.max() if lr_imp.max() > 0 else lr_imp))
    rf_dict = dict(zip(np.asarray(rf_features), rf_imp / rf_imp.max() if rf_imp.max() > 0 else rf_imp))

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_features) * 0.4)))

    y_pos = np.arange(len(all_features))

    for i, feat in enumerate(reversed(all_features)):
        lr_val = lr_dict.get(feat, 0)
        rf_val = rf_dict.get(feat, 0)

        # Determinar color según pertenencia
        if feat in common:
            marker_color_lr = COLOR_PALETTE[5]  # amarillo
            marker_color_rf = COLOR_PALETTE[5]
        else:
            marker_color_lr = COLOR_PALETTE[0]  # cyan
            marker_color_rf = COLOR_PALETTE[4]  # magenta

        ax.scatter(lr_val, i, color=COLOR_PALETTE[0], s=80, zorder=4, marker="o")
        ax.scatter(rf_val, i, color=COLOR_PALETTE[4], s=80, zorder=4, marker="s")

        # Línea conectora
        ax.plot([lr_val, rf_val], [i, i], color="white", alpha=0.3, linewidth=1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(reversed(all_features)), fontsize=9)
    ax.set_xlabel("Importancia Normalizada (0-1)", fontsize=11)
    ax.set_title("Comparación de Importancia — LR vs RF", fontweight="bold", fontsize=13, pad=12)

    # Leyenda
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PALETTE[0], markersize=9, label="Reg. Logística"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_PALETTE[4], markersize=9, label="Random Forest"),
        Patch(facecolor=COLOR_PALETTE[5], label=f"En ambos top-{top_n} ({len(common)})", alpha=0.3),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    # Resaltar features comunes con fondo
    for i, feat in enumerate(reversed(all_features)):
        if feat in common:
            ax.axhspan(i - 0.4, i + 0.4, color=COLOR_PALETTE[5], alpha=0.08, zorder=1)

    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig
