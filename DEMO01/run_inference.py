import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os

# --- CONFIGURACIÓN ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_mlp.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_MAP_PATH = "data_prepared/label_map.npy"

# --- CARGAR MODELO Y SCALER ---
print("📦 Cargando modelo y scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
idx_to_label = {v: k for k, v in label_map.items()}

# --- CONFIGURAR MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- FUNCIÓN DE PREPROCESAMIENTO ---
def preprocess_landmarks(landmarks):
    """Convierte coordenadas absolutas a relativas y normalizadas."""
    base_x, base_y, base_z = landmarks[0]
    rel = np.array([[x - base_x, y - base_y, z - base_z] for x, y, z in landmarks])
    max_val = np.max(np.abs(rel))
    if max_val > 0:
        rel /= max_val
    return rel.flatten()

# --- INICIO DE CÁMARA ---
cap = cv2.VideoCapture(0)
prev_label = ""
prev_time = time.time()
stable_label = ""
stable_count = 0

print("🎥 Reconociendo lenguaje de señas en tiempo real...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extraer y procesar landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        processed = preprocess_landmarks(landmarks).reshape(1, -1)
        processed = scaler.transform(processed)

        # Predicción
        pred_idx = model.predict(processed)[0]
        label = idx_to_label[pred_idx]
        probas = model.predict_proba(processed)[0]
        confidence = np.max(probas)

        # Filtrar resultados para estabilidad visual
        if label == prev_label:
            stable_count += 1
        else:
            stable_count = 0
        prev_label = label

        if stable_count > 3:  # mostrar solo si se mantiene unos frames
            stable_label = label

        # Mostrar resultado
        cv2.putText(frame, f"Letra: {stable_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confianza: {confidence*100:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        stable_label = ""
        prev_label = ""

    cv2.imshow("ASL en tiempo real", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("👋 Finalizado.")
