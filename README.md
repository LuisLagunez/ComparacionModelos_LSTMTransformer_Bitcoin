# Predicción del Precio de Bitcoin (BTC-USD) usando LSTM y Transformer con Walk-Forward

Este repositorio contiene un estudio completo sobre la predicción del precio de **Bitcoin (BTC-USD)** utilizando dos arquitecturas profundas:

- **LSTM (Long Short-Term Memory)**
- **Transformer Encoder**

Ambos modelos se entrenan mediante un esquema **Walk-Forward paralelo**, con recalibración completa en cada ventana temporal, lo que simula un entorno real de predicción sin fuga de datos (*data leakage*).

Todo el proyecto fue desarrollado y ejecutado en **Google Colab**, por lo que se consideran también sus limitantes (tiempo, RAM, GPU compartida, sesiones desconectadas, etc.).

---

## 📌 Objetivos del proyecto

1. **Evaluar si los modelos secuenciales (LSTM y Transformer) pueden predecir el precio del Bitcoin utilizando el conjunto de variables OHLCV (Open, High, Low, Close, Volume).**
2. **Probar dos escenarios temporales:**
   - **Caso 2024:** Entrenamiento con 9 meses, predicción semanal (horizonte H=7).
   - **Caso Histórico (2010–2024):** Entrenamiento con 14 años de datos, predicción mensual (horizonte H=30).
3. **Comparar ambas arquitecturas bajo el mismo esquema Walk-Forward.**
4. **Evitar completamente el data leakage**, usando:
   - Escaladores entrenados *solo* con datos previos.
   - Secuencias generadas correctamente hasta cada ventana.
   - Reentrenamiento modelo por ventana.

---

# 🧠 Arquitecturas utilizadas

### 🔹 LSTM
- 2 capas
- 64 unidades
- Dropout 0.2
- Entrenamiento por ventana con Adam, LR=1e-3

### 🔹 Transformer Encoder
- d_model = 64
- nheads = 4
- 2 capas encoder
- Activación GELU
- AdamW, LR=1e-4
- Positional Encoding sinusoidal propio

Ambos modelos fueron entrenados entre **4 y 16 epochs por ventana**, con **repeticiones por ventana (ensembling)** para reducir la varianza de la predicción.

---

# 📊 Esquemas temporales del estudio

## 🟦 Caso 1 — Año 2024
- Datos: 1 ene 2024 → 31 dic 2024  
- Entrenamiento: 1 ene → 30 sep  
- Prueba: 1 oct → 31 dic  
- Horizonte: **7 días (predicción semanal)**  
- Ventana: **60 días**

## 🟩 Caso 2 — Historia completa (2010–2024)
- Datos: jul 2010 → dic 2024  
- Entrenamiento: todo hasta dic 2023  
- Prueba: año 2024  
- Horizonte: **30 días (predicción mensual)**  
- Ventana: **60 días**

---

# 🔄 Walk-Forward Rolling Training

Ambos modelos utilizan un esquema **realista y estricto**:

Para cada nuevo punto en el set de prueba:

1. Se toman **todas las secuencias anteriores como entrenamiento**.  
2. Se reescala únicamente usando valores previos.  
3. Se entrena el modelo **desde cero** para esa ventana.  
4. Se predice solo 1 punto futuro.  
5. Se pasa a la siguiente ventana.

**Esto simula un trader o sistema automatizado real**, utilizando solo información disponible hasta ese momento.

---

# 🧪 Métricas utilizadas

- **MAE** — Error absoluto medio  
- **RMSE** — Raíz del error cuadrático medio  
- **MAPE** — Error porcentual absoluto medio  
- **sMAPE** — Error porcentual simétrico  
- **Precisión promedio (%)**  
- **Precisión punto a punto (%)**  
- **Directional Accuracy (DA)**  
- **Correlación de Pearson**

Todas las métricas están implementadas manualmente en el repositorio.

---

# 📁 Estructura del código

El código puede visualizarse directamente [aquí](Experimentacion_LSTMTransformer_Bitcoin.ipynb) o directamente en el notebook de [Google_Colab](https://colab.research.google.com/drive/1mSflyyC4mRUskUfOUhZFd-WnHKwKqr2x#scrollTo=Ph_Ts1AwjbuN).

https://colab.research.google.com/drive/1mSflyyC4mRUskUfOUhZFd-WnHKwKqr2x

El notebook contiene estos bloques principales:

### **Bloque 1 — Descarga y limpieza de datos**
- Obtención desde **Yahoo Finance** usando *yfinance*  
- Limpieza, orden temporal, splits y etiquetado

### **Bloque 2 — Escalado y generación de secuencias**
- MinMaxScaler entrenado **solo con entrenamiento**
- Secuencias multivariadas para cada horizonte
- Generador Walk-Forward parametrizable

### **Bloque 3 — LSTM Walk-Forward**
- Implementación desde cero en PyTorch
- Entrenamiento por ventana
- Ensemble de repeticiones

### **Bloque 4 — Transformer Walk-Forward**
- Positional Encoding propio
- Encoder con GELU + LayerNorm
- AdamW y dropout
- Ensemble por ventana

### **Bloque 5 — Evaluación y visualización**
- Desescalado real
- Cálculo de métricas
- Gráficos de comportamiento predictivo
- Exportación de tabla consolidada

---

# 📈 Resultados principales

El proyecto genera automáticamente:

- Métricas comparativas completas en CSV.
- Gráficos:
  - Serie Real vs LSTM vs Transformer.
  - Error absoluto comparado.
- Tabla final con todas las métricas para los 4 escenarios:
  - LSTM 2024  
  - Transformer 2024  
  - LSTM Histórico  
  - Transformer Histórico  

**Los resultados pueden visualizarse directamente en Colab.**

---

# ⚙️ Dependencias principales

```text
Python 3.10+
PyTorch
yfinance
scikit-learn
numpy
pandas
matplotlib
tqdm
scipy