import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

# --- CONFIGURACIÓN ---
DATA_PATH = "data/landmarks.csv"
OUTPUT_DIR = "data_prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. CARGAR EL CSV ---
print(f"📂 Cargando dataset desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, header=None)

# Última columna = etiqueta (letra)
# Todas las demás = coordenadas de landmarks
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Total de muestras: {len(df)}")
print(f"Letras detectadas: {y.unique()}")

# --- 2. LIMPIEZA DE DATOS ---
# Eliminar filas con valores nulos o incompletos
mask_invalid = X.isnull().any(axis=1)
if mask_invalid.any():
    print(f"⚠️ Filas inválidas detectadas: {mask_invalid.sum()}")
    df = df[~mask_invalid]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# Eliminar duplicados (por seguridad)
df = df.drop_duplicates()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"✅ Dataset limpio: {len(df)} muestras válidas")

# --- 3. ENCODIFICAR ETIQUETAS ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# También podríamos guardar el mapeo para usar en inferencia
label_map = {label: idx for idx, label in enumerate(label_encoder.classes_)}
print(f"🔠 Mapeo de letras: {label_map}")

# Guardar mapeo
np.save(os.path.join(OUTPUT_DIR, "label_map.npy"), label_map)

# --- 4. DIVISIÓN TRAIN / TEST / VALIDACIÓN ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"📊 División:")
print(f"  Entrenamiento: {len(X_train)} muestras")
print(f"  Validación:    {len(X_val)} muestras")
print(f"  Prueba:        {len(X_test)} muestras")

# --- 5. GUARDAR ARCHIVOS LIMPIOS ---
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print(f"💾 Archivos guardados en: {OUTPUT_DIR}")
print("✅ Dataset preparado correctamente.")
