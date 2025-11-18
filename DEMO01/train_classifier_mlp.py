import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
DATA_DIR = "data_prepared"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. CARGAR DATOS ---
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
label_map = np.load(os.path.join(DATA_DIR, "label_map.npy"), allow_pickle=True).item()

# --- 2. NORMALIZACIÓN ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --- 3. DEFINICIÓN DEL MODELO ---
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=800,
    alpha=0.0005,
    learning_rate_init=0.001,
    batch_size=32,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

# --- 4. ENTRENAMIENTO ---
print("🚀 Entrenando MLPClassifier...")
mlp.fit(X_train, y_train)
print("✅ Entrenamiento finalizado.\n")

# --- 5. VALIDACIÓN ---
y_pred = mlp.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"🎯 Precisión en validación: {acc*100:.2f}%")

# --- 6. REPORTE DE CLASIFICACIÓN ---
print("\n📋 Reporte de clasificación:")
print(classification_report(y_val, y_pred, target_names=label_map.keys()))

# --- 7. MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_val, y_pred)
labels = list(label_map.keys())
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)
plt.colorbar()

# Etiquetas en cada celda
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.tight_layout()
plt.show()

# --- 8. GUARDAR MODELO Y SCALER ---
joblib.dump(mlp, os.path.join(MODEL_DIR, "hand_sign_mlp.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print(f"💾 Modelo guardado en: {MODEL_DIR}/hand_sign_mlp.pkl")
print(f"💾 Scaler guardado en: {MODEL_DIR}/scaler.pkl")
