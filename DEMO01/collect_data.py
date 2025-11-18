import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# --- CONFIGURACIÓN ---
LETRA = "Z"           # Letra actual que estás grabando
N_MUESTRAS = 300      # Cuántas muestras grabar
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")  # archivo general

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- Captura ---
cap = cv2.VideoCapture(0)
contador = 0
print(f"🎬 Grabando datos para la letra '{LETRA}' ({N_MUESTRAS} muestras). Presiona 'q' para salir.")

# Abrir CSV global (append)
csv_file = open(CSV_PATH, mode='a', newline='')
csv_writer = csv.writer(csv_file)

def preprocess_landmarks(landmarks):
    # Convertir a coordenadas relativas a la muñeca
    base_x, base_y, base_z = landmarks[0]
    rel = np.array([[x - base_x, y - base_y, z - base_z] for x, y, z in landmarks])
    # Normalizar
    max_val = np.max(np.abs(rel))
    if max_val > 0:
        rel /= max_val
    return rel.flatten()

while cap.isOpened() and contador < N_MUESTRAS:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    color = (0, 0, 255)  # 🔴 Por defecto: rojo (no grabando)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # --- Criterio de calidad: tamaño del bounding box ---
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))

        # Si el área es suficientemente grande, se asume que la mano está bien visible
        if area > 0.02:  
            norm_landmarks = preprocess_landmarks(landmarks)
            csv_writer.writerow(np.concatenate((norm_landmarks, [LETRA])))
            contador += 1
            color = (0, 255, 0)  # 🟢 Verde = muestra válida
            time.sleep(0.1)  # pausa ligera (~10 FPS)

        # Dibujar la mano en pantalla
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar texto de estado
    cv2.putText(frame, f"Letra: {LETRA} | Muestras: {contador}/{N_MUESTRAS}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Captura de datos LSM", frame)

    # Salida con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ Captura finalizada.")
cap.release()
hands.close()
csv_file.close()
cv2.destroyAllWindows()
