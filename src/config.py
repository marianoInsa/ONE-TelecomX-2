"""
Configuración global del proyecto.

Centraliza constantes, paleta de colores y configuración de estilo
para matplotlib. Importar este módulo garantiza consistencia visual
en todos los notebooks y scripts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Constantes de modelado
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.15

# Paleta de colores
COLOR_PALETTE = [
    "#37EBF3", # 1. Electric Blue
    "#E455AE", # 2. Frostbite
    "#1AC5B0", # 3. Keppel
    "#FDF500", # 4. Rich Lemon
    "#710000", # 5. Blood Red
    "#272932", # 6. Raisin Black
    "#9370DB", # 7. Blushing Purple
    "#D1C5CC", # 8. Pale Silver
    "#CB1DCD", # 9. Steel Pink
]
BACKGROUND_COLOR = "#212946"

# Colormap cyberpunk divergente
CMAP_CYBER = LinearSegmentedColormap.from_list(
    "cyber_div", [COLOR_PALETTE[1], BACKGROUND_COLOR, COLOR_PALETTE[0]]
)


def setup_plot_style() -> None:
    """Aplica el estilo cyberpunk y la configuración global de matplotlib."""
    plt.style.use("cyberpunk")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["font.family"] = "sans-serif"
