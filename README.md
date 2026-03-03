# Predicción de Cancelación (Churn) | Telecom X – Parte 2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Cyberpunk-blueviolet)
![Status](https://img.shields.io/badge/Status-Complete-success)

</div>

---

## Descripción del Proyecto

Este proyecto constituye la **Parte 2** del análisis de cancelación de clientes (*churn*) para **Telecom X**, una empresa de telecomunicaciones. Mientras que la [Parte 1](https://github.com/marianoInsa/ONE-TelecomX) se enfocó en la exploración y el análisis descriptivo de los datos, esta segunda fase implementa un **pipeline completo de Machine Learning** para predecir qué clientes tienen mayor probabilidad de cancelar sus servicios.

El proyecto abarca desde la preparación de los datos para el modelado (codificación, normalización, balanceo de clases) hasta la interpretación de los resultados con importancia de variables, culminando en recomendaciones estratégicas accionables para el negocio.

### Características Principales

- ✅ **Preprocesamiento robusto**: One-Hot Encoding con `sklearn`, eliminación de variables redundantes, estandarización con `StandardScaler`
- ✅ **Balanceo de clases con SMOTE**: Generación de ejemplos sintéticos de la clase minoritaria para mejorar la detección de churn
- ✅ **Prevención de data leakage**: Pipeline riguroso que escala y balancea exclusivamente sobre el conjunto de entrenamiento
- ✅ **Selección de variables multi-criterio**: Correlación de Pearson, Información Mutua y análisis de multicolinealidad (VIF)
- ✅ **Dos modelos complementarios**: Regresión Logística (interpretable) y Random Forest (captura no linealidades)
- ✅ **Evaluación exhaustiva**: Accuracy, Precision, Recall, F1-score, matrices de confusión y diagnóstico de overfitting
- ✅ **Interpretación de importancia de variables**: Coeficientes logísticos + importancia Gini, con comparación cruzada entre modelos
- ✅ **Código modular**: Lógica encapsulada en `src/` con 6 módulos especializados y estilo visual cyberpunk consistente
- ✅ **Reproducibilidad total**: Semilla fija (`RANDOM_STATE = 42`), artefactos serializados y notebooks encadenados

---

## Contexto: Parte 1 – Análisis Exploratorio

Este proyecto es la continuación directa del **análisis exploratorio de datos (EDA)** realizado en la Parte 1. Allí se implementó un pipeline ETL completo que extrajo datos desde una API externa, los limpió y transformó, y se realizó un análisis descriptivo profundo de la base de clientes.

| Aspecto | Parte 1 | Parte 2 |
|---|---|---|
| **Enfoque** | ETL + Análisis Exploratorio (EDA) | Machine Learning Predictivo |
| **Entregable** | Insights descriptivos + visualizaciones estratégicas | Modelos predictivos + factores de churn + recomendaciones |
| **Dataset** | 7,043 clientes × 21 variables (crudo desde API) | 7,043 clientes × 31 variables (codificado para ML) |
| **Hallazgos** | Churn Rate: 26.54%, variables críticas identificadas, segmentación de riesgo | Modelo recomendado (LR, Recall 79.4%), factores de churn robustos |

### Hallazgos Clave de la Parte 1

La exploración inicial reveló patrones fundamentales que guían el modelado:

- **Tasa de churn general**: 26.54% (1,869 clientes perdidos de 7,043)
- **Adultos mayores (Senior)**: Tasa de churn del 41.68% — casi el doble del promedio
- **Contratos mensuales**: Tasa de churn significativamente mayor que contratos anuales o bianuales
- **Fibra óptica**: Paradójicamente, clientes con fibra óptica presentan mayor cancelación
- **Impacto económico**: Pérdida estimada de $139,129.83 en cargos mensuales recurrentes
- **Género**: Sin impacto significativo — tasa de churn idéntica para ambos géneros (26.92%)

> 📌 El dataset utilizado en esta Parte 2 (`telecom_data_processed.csv`) es el resultado del pipeline ETL de la Parte 1. Se encuentra en `data/raw/` como dato inmutable de entrada.

---

## Problema de Negocio

Telecom X enfrenta una **tasa de cancelación del 26.54%**, lo que se traduce en pérdidas significativas de ingresos recurrentes. El costo de adquirir un nuevo cliente es muy superior al de retener uno existente, por lo que la empresa necesita:

1. **Predecir** qué clientes están en riesgo de cancelar *antes* de que lo hagan.
2. **Comprender** qué factores impulsan la cancelación para diseñar intervenciones efectivas.
3. **Priorizar** recursos de retención hacia los clientes con mayor probabilidad de churn.

> **Enfoque elegido**: En un problema de retención de clientes, el **Recall** (proporción de churners detectados) es la métrica más valiosa — es preferible contactar a un cliente que no iba a cancelar (falso positivo) que perder a uno que sí iba a hacerlo (falso negativo).

---

## Objetivos del Desafío

1. **Preparar los datos para el modelado**: Tratamiento, codificación, normalización y balanceo de clases.
2. **Realizar análisis de correlación y selección de variables**: Identificar las features más relevantes para predecir churn.
3. **Entrenar dos o más modelos de clasificación**: Regresión Logística y Random Forest con estrategias diferenciadas.
4. **Evaluar el rendimiento con métricas estándar**: Accuracy, Precision, Recall, F1-score y matrices de confusión.
5. **Interpretar los resultados**: Importancia de variables desde ambas perspectivas (lineal y ensemble).
6. **Crear una conclusión estratégica**: Identificar los principales factores de cancelación y proponer estrategias de retención.

---

## Tecnologías y Herramientas

### Lenguajes y Plataformas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | ≥ 3.13 | Lenguaje principal |
| Jupyter Notebook | 1.1+ | Entorno de desarrollo interactivo |
| Git | 2.25+ | Control de versiones |
| Google Colab | — | Ejecución alternativa en la nube |

### Librerías

| Librería | Versión | Propósito |
|---|---|---|
| `pandas` | ≥ 3.0.1 | Manipulación y análisis de datos |
| `numpy` | ≥ 2.4.2 | Operaciones numéricas y vectoriales |
| `scikit-learn` | ≥ 1.8.0 | Modelos de ML, métricas, preprocesamiento |
| `imbalanced-learn` | ≥ 0.14.1 | SMOTE para balanceo de clases |
| `matplotlib` | ≥ 3.10.8 | Visualizaciones estáticas |
| `mplcyberpunk` | ≥ 0.7.6 | Estilo visual cyberpunk para gráficos |
| `seaborn` | ≥ 0.13.2 | Visualizaciones estadísticas avanzadas |
| `statsmodels` | ≥ 0.14.4 | Análisis estadístico (VIF) |
| `ipykernel` | ≥ 7.2.0 | Kernel de Jupyter para entorno virtual |

---

## Estructura del Proyecto

```
ONE-TelecomX-2/
│
├── data/
│   ├── raw/                              # Datos originales inmutables
│   │   └── telecom_data_processed.csv    # Dataset limpio de la Parte 1 (7,043 × 22)
│   └── processed/                        # Artefactos intermedios y finales
│       ├── telecom_encoded.csv           # Dataset codificado (7,043 × 31)
│       ├── selected_features.json        # 20 features seleccionadas + metadata
│       ├── evaluation_results.json       # Métricas de ambos modelos
│       ├── X_train_bal.csv               # Features de train (escalado + SMOTE)
│       ├── y_train_bal.csv               # Target de train (balanceado 50/50)
│       ├── X_test.csv                    # Features de test (escalado)
│       └── y_test.csv                    # Target de test (distribución real)
│
├── models/                               # Modelos serializados
│   ├── logistic_regression.pkl           # Regresión Logística entrenada
│   ├── random_forest.pkl                 # Random Forest entrenado
│   └── scaler_modelado.pkl               # StandardScaler ajustado en train
│
├── notebooks/                            # Pipeline secuencial de notebooks
│   ├── 01_preparacion_datos.ipynb        # Carga, encoding, SMOTE, escalado
│   ├── 02_correlacion_seleccion.ipynb    # Correlación, MI, VIF, selección
│   ├── 03_modelado_predictivo.ipynb      # Train/test, entrenamiento, evaluación
│   └── 04_interpretacion_conclusiones.ipynb  # Importancia, conclusión estratégica
│
├── src/                                  # Módulos Python reutilizables
│   ├── __init__.py                       # Paquete: "Telecom X – Churn Prediction"
│   ├── config.py                         # Paleta, estilos, constantes, rutas
│   ├── data_loader.py                    # Carga local + fallback URL
│   ├── preprocessing.py                  # Encoding, split, SMOTE, scaling
│   ├── analysis.py                       # Correlación, MI, VIF, selección
│   ├── modeling.py                       # Train, save, load, evaluate
│   └── visualization.py                  # 1,000+ líneas de gráficos cyberpunk
│
├── CHALLENGE.md                          # Enunciado del desafío
├── pyproject.toml                        # Configuración del proyecto y dependencias
```

### Convenciones

- **Notebooks numerados**: Se ejecutan en orden secuencial (01 → 02 → 03 → 04). Cada notebook consume artefactos del anterior.
- **Código modular**: La lógica pesada reside en `src/`; los notebooks se enfocan en narrativa y visualización.
- **Datos inmutables**: `data/raw/` nunca se modifica programáticamente. Los resultados se guardan en `data/processed/`.
- **Configuración centralizada**: Todas las constantes, colores y estilos se importan desde `src/config.py`.

---

## Pipeline de Machine Learning

El proyecto implementa un pipeline de ML en 4 etapas, cada una correspondiente a un notebook:

```
   telecom_data_processed.csv (Parte 1)
              │
              ▼
   ┌──────────────────────────┐
   │  01_preparacion_datos    │   Encoding + Escalado + SMOTE
   │  telecom_encoded.csv     │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  02_correlacion_seleccion│   Pearson + MI + VIF → 20 features
   │  selected_features.json  │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  03_modelado_predictivo  │   LR (20 feats) + RF (30 feats)
   │  *.pkl + evaluation.json │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  04_interpretacion       │   Importancia + Recomendaciones
   │  Conclusión estratégica  │
   └──────────────────────────┘
```

### Etapa 1: Preparación de Datos

**Notebook**: `01_preparacion_datos.ipynb`

Transforma el dataset crudo de la Parte 1 al formato que los algoritmos de ML requieren.

| Paso | Acción | Detalle |
|---|---|---|
| **1. Eliminación de columnas** | `drop_non_predictive()` | Elimina `customerID` (ID administrativo) y `cuentas_diarias` (redundante, $r = 1.0$ con `account_charges_monthly`) |
| **2. Codificación** | `encode_features()` | Variables binarias (Yes/No → 1/0) + One-Hot Encoding con `drop_first=True` para multi-clase. Usa `sklearn.OneHotEncoder` para prevenir leakage en producción |
| **3. Análisis de desbalance** | Distribución de Churn | Ratio 2.77:1 (73.5% No Churn / 26.5% Churn). Suficiente para sesgar modelos hacia clase mayoritaria |
| **4. Split + Scale + SMOTE** | `split_and_balance()` | Split 80/20 estratificado → `StandardScaler` (fit solo en train) → SMOTE (solo en train). Pipeline anti-leakage |
| **5. Verificación de escalado** | Estadísticas de train/test | Confirma media ≈ 0, std ≈ 1 en train tras estandarización |

**Artefacto generado**: `data/processed/telecom_encoded.csv` — Dataset completo codificado (7,043 × 31).

### Etapa 2: Correlación y Selección de Variables

**Notebook**: `02_correlacion_seleccion.ipynb`

Análisis multi-criterio para identificar las variables con mayor poder predictivo.

| Método | Propósito | Hallazgo Principal |
|---|---|---|
| **Correlación de Pearson** | Dependencia lineal con Churn | `customer_tenure` ($r = -0.35$) es el predictor más fuerte. Fibra óptica y pago electrónico son los mayores factores de riesgo |
| **Información Mutua** | Dependencia no lineal (cualquier tipo) | Revela variables cuyo poder predictivo no es capturado por correlación lineal |
| **VIF** | Multicolinealidad entre features | Identifica redundancias que inflan la varianza de los coeficientes en modelos lineales |

**Estrategia dual de selección**:
- **Modelos lineales (LR)**: 20 features con $|r| \geq 0.15$ — reduce ruido y multicolinealidad.
- **Modelos de árboles (RF)**: 30 features completas — los árboles ignoran automáticamente variables irrelevantes y capturan interacciones.

**Variables seleccionadas (umbral $|r| \geq 0.15$)**:

| Categoría | Features |
|---|---|
| **Contractuales** | `account_contract_Two year`, `account_contract_One year`, `account_paperlessbilling` |
| **Temporales** | `customer_tenure` |
| **Económicas** | `account_charges_monthly`, `account_charges_total` |
| **Método de pago** | `account_paymentmethod_Electronic check` |
| **Servicios de Internet** | `internet_internetservice_Fiber optic`, `internet_internetservice_No`, `internet_onlinesecurity_Yes`, `internet_techsupport_Yes` |
| **Sin servicio de Internet** | 6 variables indicadoras (`_No internet service`) |
| **Demográficas** | `customer_seniorcitizen`, `customer_partner`, `customer_dependents` |

**Artefacto generado**: `data/processed/selected_features.json`

### Etapa 3: Modelado Predictivo

**Notebook**: `03_modelado_predictivo.ipynb`

Entrenamiento y evaluación de dos modelos complementarios.

| Aspecto | Regresión Logística | Random Forest |
|---|---|---|
| **Familia** | Modelo lineal | Ensemble de árboles (100 estimadores) |
| **Features utilizadas** | 20 seleccionadas | 30 completas |
| **Normalización** | Requerida (ya aplicada) | Invariante a escala |
| **Hiperparámetros** | `max_iter=1000` | `n_estimators=100` |
| **Fortaleza** | Coeficientes interpretables, baseline robusto | Relaciones no lineales, interacciones entre variables |

**Artefactos generados**: `models/logistic_regression.pkl`, `models/random_forest.pkl`, `data/processed/evaluation_results.json`

### Etapa 4: Interpretación y Conclusiones

**Notebook**: `04_interpretacion_conclusiones.ipynb`

Análisis de importancia de variables desde dos perspectivas complementarias:

- **Regresión Logística**: Coeficientes con signo ($\beta_i$). Positivo → aumenta la probabilidad de churn; negativo → factor de retención. Comparables entre features gracias a la estandarización.
- **Random Forest**: Importancia basada en impureza de Gini. Indica relevancia para la clasificación pero no dirección del efecto.

La **comparación cruzada** identifica los factores de churn más **robustos**: aquellos que aparecen como relevantes en ambos modelos, independientemente de las suposiciones algorítmicas.

---

## Resultados del Modelado

### Métricas de Evaluación — Test Set

| Métrica | Regresión Logística | Random Forest |
|---|---|---|
| **Accuracy** | 0.7473 | **0.7807** |
| **Precision** | 0.5156 | **0.5876** |
| **Recall** | **0.7941** | 0.5829 |
| **F1-score** | **0.6253** | 0.5852 |

### Matriz de Confusión — Test Set

| | | **Regresión Logística** | | **Random Forest** | |
|---|---|---|---|---|---|
| | | Pred: No Churn | Pred: Churn | Pred: No Churn | Pred: Churn |
| **Real** | No Churn | 756 | 279 | 882 | 153 |
| | Churn | 77 | **297** | 156 | **218** |

### Diagnóstico de Generalización

| Modelo | F1 Train | F1 Test | Gap (Δ) | Diagnóstico |
|---|---|---|---|---|
| **Regresión Logística** | 0.7854 | 0.6253 | +0.1601 | Overfitting moderado |
| **Random Forest** | 0.9984 | 0.5852 | +0.4132 | Overfitting severo |

### Modelo Recomendado: Regresión Logística

Para el objetivo de **retención de clientes**, la Regresión Logística es el modelo preferible por:

1. **Recall superior (79.4%)**: Detecta casi 8 de cada 10 clientes que efectivamente cancelarán. El Random Forest solo detecta 5.8 de cada 10.
2. **Mejor F1-score (0.6253)**: Mejor equilibrio entre precision y recall.
3. **Interpretabilidad directa**: Los coeficientes indican exactamente qué variables influyen y en qué dirección.
4. **Menor overfitting**: Gap de 0.16 en F1 vs 0.41 del Random Forest — generaliza significativamente mejor.

> En un escenario de negocio, preferimos un modelo que contacte a clientes que no iban a cancelar (falsos positivos = 279) a perder clientes que sí iban a cancelar (falsos negativos = 77). La Regresión Logística minimiza estos últimos.

---

## Factores Clave de Cancelación

El análisis de importancia de variables, validado por **consenso entre ambos modelos**, identifica los siguientes drivers de churn:

### Factores que Impulsan la Cancelación

| # | Factor | Score Combinado | Perspectiva LR (coeficiente) | Perspectiva RF (Gini) | Interpretación |
|---|---|---|---|---|---|
| 1 | **`customer_tenure`** (baja antigüedad) | Alto | Coeficiente negativo fuerte | Top importancia | Los clientes nuevos (< 12 meses) tienen la mayor probabilidad de cancelar. El riesgo disminuye drásticamente con el tiempo |
| 2 | **`internet_internetservice_Fiber optic`** | Alto | Coeficiente positivo alto | Relevante | Clientes con fibra óptica cancelan más — posiblemente por insatisfacción con la relación precio/calidad |
| 3 | **`account_paymentmethod_Electronic check`** | Alto | Coeficiente positivo alto | Relevante | El pago por cheque electrónico está asociado a clientes menos comprometidos y con mayor churn |
| 4 | **`account_charges_monthly`** | Alto | Coeficiente positivo | Top importancia | Cargos mensuales altos impulsan la cancelación — los churners tienen una mediana de $80 vs $64 de los retenidos |
| 5 | **`account_paperlessbilling`** | Medio-Alto | Coeficiente positivo | Relevante | La facturación sin papel se asocia a mayor churn — posible correlación con perfiles digitales más propensos a cambiar de proveedor |
| 6 | **`customer_seniorcitizen`** | Medio | Coeficiente positivo | Relevante | Adultos mayores tienen tasa de churn del 41.68%, casi el doble del promedio (26.54%) |

### Factores de Retención

| # | Factor | Efecto | Interpretación |
|---|---|---|---|
| 1 | **`account_contract_Two year`** | ↓ Churn (fuerte) | Los contratos bianuales son el ancla de retención más poderosa. El compromiso a largo plazo reduce drásticamente la probabilidad de cancelación |
| 2 | **`account_contract_One year`** | ↓ Churn (moderado) | Los contratos anuales también reducen el churn, aunque en menor medida que los bianuales |
| 3 | **`internet_onlinesecurity_Yes`** | ↓ Churn | El servicio de seguridad en línea actúa como ancla — los clientes que lo tienen cancelan significativamente menos |
| 4 | **`internet_techsupport_Yes`** | ↓ Churn | El soporte técnico tiene un efecto protector similar al de seguridad en línea |
| 5 | **`customer_tenure`** (alta antigüedad) | ↓ Churn (fuerte) | Mediana de 38 meses para retenidos vs 10 meses para churners. La lealtad se consolida después del primer año |
| 6 | **`account_charges_total`** | ↓ Churn | Clientes con mayor gasto acumulado ($1,680 mediana retenidos vs $704 churners) son más leales — reflejo directo del tenure alto |

### La "Ventana Crítica" y la "Paradoja de la Fibra Óptica"

El análisis dirigido del Notebook 02 reveló dos patrones fundamentales:

1. **Ventana Crítica (0–12 meses)**: La mayoría de las cancelaciones ocurren durante el primer año de contrato. Los clientes que sobreviven esta ventana tienen alta probabilidad de permanecer largo tiempo. Las estrategias de retención deben concentrar recursos en este período.

2. **Paradoja de la Fibra Óptica**: A pesar de ser un servicio premium, los clientes con fibra óptica presentan mayor probabilidad de cancelación. Esto sugiere que el problema no es el servicio en sí, sino la **relación precio/calidad percibida** — estos clientes pagan más ($80 mediana) y tienen expectativas más altas que no se están cumpliendo.

---

## Estrategias de Retención Propuestas

Basándose en los factores de churn identificados y validados por ambos modelos, se proponen las siguientes estrategias:

### 1. Programa de Onboarding Intensivo (0–12 meses)

**Factor abordado**: `customer_tenure` baja + ventana crítica

- Implementar un programa de bienvenida con seguimiento personalizado durante los primeros 12 meses.
- Asignar un agente de éxito al cliente durante el período crítico.
- Ofrecer beneficios exclusivos por permanencia (descuentos progresivos, upgrades gratuitos).
- Establecer puntos de contacto proactivos a los meses 1, 3, 6 y 9.

### 2. Migración a Contratos de Largo Plazo

**Factor abordado**: `account_contract_*` como ancla de retención

- Diseñar incentivos atractivos para migrar clientes mensuales a contratos anuales o bianuales.
- Ofrecer descuentos significativos (10-20%) por compromiso de permanencia.
- Crear paquetes "lock-in" que combinen contrato largo + servicios adicionales a precio reducido.

### 3. Revisión del Servicio de Fibra Óptica

**Factor abordado**: `internet_internetservice_Fiber optic` + paradoja precio/calidad

- Auditar la satisfacción de clientes de fibra óptica vs DSL.
- Evaluar si la velocidad y estabilidad están a la altura del precio premium.
- Considerar planes de fibra óptica con diferentes niveles de precio/velocidad.
- Implementar encuestas NPS específicas para este segmento.

### 4. Promoción de Servicios Protectores

**Factor abordado**: `internet_onlinesecurity_Yes` y `internet_techsupport_Yes`

- Ofrecer paquetes que incluyan seguridad en línea y soporte técnico como estándar.
- Crear bundles atractivos que integren estos servicios con descuento.
- Comunicar el valor agregado de estos servicios en las campañas de retención.

### 5. Programa Senior Care

**Factor abordado**: `customer_seniorcitizen` (41.68% de churn)

- Crear un programa especializado para adultos mayores con atención personalizada.
- Simplificar planes y facturación para este segmento.
- Ofrecer soporte técnico dedicado con horarios extendidos y atención presencial.
- Evaluar planes con precios ajustados para clientes senior.

### 6. Optimización del Método de Pago

**Factor abordado**: `account_paymentmethod_Electronic check`

- Incentivar la migración a métodos de pago automáticos (tarjeta de crédito, débito automático).
- Ofrecer descuentos por domiciliación bancaria.
- Simplificar el proceso de cambio de método de pago.

### 7. Sistema de Alerta Temprana

**Factor abordado**: Todos los drivers combinados

- Implementar el modelo de Regresión Logística en producción para scoring mensual de riesgo.
- Generar alertas automáticas cuando un cliente supere un umbral de probabilidad de churn.
- Priorizar la intervención del equipo de retención según el score de riesgo.
- Monitorear la efectividad de las intervenciones y retroalimentar el modelo.

---

## Instalación y Despliegue Local

### Requisitos Previos

- **Python 3.13** o superior
- **pip** (incluido con Python) o **uv** (gestor de paquetes moderno)
- **Git** 2.25+

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/marianoInsa/ONE-TelecomX-2.git
cd ONE-TelecomX-2
```

### Paso 2: Crear Entorno Virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

**Con pip:**
```bash
pip install -e .
```

**Con uv (alternativa más rápida):**
```bash
pip install uv
uv sync
```

Esto instalará todas las dependencias definidas en `pyproject.toml`:

```
pandas, numpy, scikit-learn, imbalanced-learn, matplotlib,
mplcyberpunk, seaborn, statsmodels, ipykernel, jupyter
```

### Paso 4: Obtener los Datos

El dataset de entrada proviene de la Parte 1 del proyecto. Si la carpeta `data/raw/` no contiene `telecom_data_processed.csv`, el notebook 01 lo descargará automáticamente desde la URL de GitHub como fallback.

```bash
mkdir -p data/raw
# Opcionalmente, descargar manualmente:
curl -o data/raw/telecom_data_processed.csv https://raw.githubusercontent.com/marianoInsa/ONE-TelecomX/243bebbeb92b071c5d05ed0bf47f1fb9fe25ca2c/data/telecom_data_processed.csv
```

---

### Opción A: VS Code con Extensiones

#### Extensiones Requeridas

| Extensión | ID | Propósito |
|---|---|---|
| **Python** | `ms-python.python` | Soporte de lenguaje Python |
| **Jupyter** | `ms-toolsai.jupyter` | Ejecución de notebooks `.ipynb` |
| **Pylance** | `ms-python.vscode-pylance` | IntelliSense y análisis estático |

#### Pasos

1. Abrir el proyecto en VS Code:
   ```bash
   code ONE-TelecomX-2
   ```

2. Instalar las extensiones recomendadas (VS Code las sugerirá automáticamente si hay un `.vscode/extensions.json`), o instalarlas manualmente desde la barra lateral de extensiones.

3. Seleccionar el intérprete Python del entorno virtual:
   - `Ctrl + Shift + P` → "Python: Select Interpreter"
   - Elegir `.venv` → `Python 3.13.x`

4. Abrir y ejecutar los notebooks en orden:
   - Abrir `notebooks/01_preparacion_datos.ipynb`
   - Seleccionar el kernel `.venv (Python 3.13.x)` cuando VS Code lo solicite
   - Ejecutar todas las celdas con `Ctrl + Shift + Enter` o el botón "Run All"
   - Repetir para los notebooks 02, 03 y 04 en orden secuencial

> **Tip**: VS Code permite ejecutar celdas individuales con `Shift + Enter` y visualizar outputs inline, incluyendo gráficos con estilo cyberpunk.

---

### Opción B: Jupyter Notebook Clásico

1. Asegurarse de que el entorno virtual está activado y ejecutar:
   ```bash
   jupyter notebook
   ```

2. El navegador se abrirá automáticamente en `http://localhost:8888`.

3. Navegar a la carpeta `notebooks/` y abrir los notebooks en orden (01 → 02 → 03 → 04).

4. Ejecutar cada notebook secuencialmente:
   - `Kernel` → `Restart & Run All`
   - O ejecutar celda por celda con `Shift + Enter`

> **Nota**: Si el kernel no aparece automáticamente, registrarlo manualmente:
> ```bash
> python -m ipykernel install --user --name=telecomx2 --display-name "TelecomX-2"
> ```

---

### Opción C: JupyterLab

1. Instalar JupyterLab (si no está incluido):
   ```bash
   pip install jupyterlab
   ```

2. Iniciar el servidor:
   ```bash
   jupyter lab
   ```

3. El navegador se abrirá en `http://localhost:8888/lab`.

4. En el explorador de archivos lateral, navegar a `notebooks/` y abrir cada notebook en orden.

5. Ejecutar con `Shift + Enter` celda por celda, o `Run` → `Run All Cells`.

> **Ventaja de JupyterLab**: Permite abrir múltiples notebooks en pestañas, terminal integrada y explorador de archivos — ideal para inspeccionar los artefactos en `data/processed/` y `models/` mientras se ejecutan los notebooks.

---

### Opción D: Google Colab

Cada notebook incluye un badge de **"Abrir en Colab"** en su primera celda. Simplemente hacer clic para abrirlo directamente en Google Colab sin instalación local.

| Notebook | Abrir en Colab |
|---|---|
| **01** — Preparación de Datos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marianoInsa/ONE-TelecomX-2/blob/main/notebooks/01_preparacion_datos.ipynb) |
| **02** — Correlación y Selección | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marianoInsa/ONE-TelecomX-2/blob/main/notebooks/02_correlacion_seleccion.ipynb) |
| **03** — Modelado Predictivo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marianoInsa/ONE-TelecomX-2/blob/main/notebooks/03_modelado_predictivo.ipynb) |
| **04** — Interpretación y Conclusiones | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marianoInsa/ONE-TelecomX-2/blob/main/notebooks/04_interpretacion_conclusiones.ipynb) |

#### Configuración en Colab

Al ejecutar en Colab, es necesario instalar las dependencias del proyecto y clonar el repositorio para acceder al paquete `src/`:

```python
# Celda 0 — Ejecutar primero en Colab
!git clone https://github.com/marianoInsa/ONE-TelecomX-2.git
%cd ONE-TelecomX-2
!pip install -e . -q
```

Luego ejecutar cada notebook normalmente. Los datos se descargarán automáticamente gracias al fallback URL implementado en `src/data_loader.py`.

> **Nota**: Google Colab no preserva archivos entre sesiones. Si la sesión se reinicia, es necesario volver a ejecutar la celda de configuración y los notebooks previos para regenerar los artefactos.

---

Proyecto desarrollado como parte del Programa Oracle Next Education (ONE) × Alura Latam por **Mariano Insaurralde**

---