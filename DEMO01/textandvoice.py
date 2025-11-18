import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from gtts import gTTS
try:
    from playsound import playsound  # prefer playsound if available
    _audio_player = "playsound"
except Exception:
    # fallback to python-vlc on Linux if playsound not installable
    try:
        import vlc
        _audio_player = "vlc"
    except Exception:
        _audio_player = None

# --- CONFIGURACIÓN ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_mlp.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_MAP_PATH = "data_prepared/label_map.npy"
AUDIO_DIR = "audio_cache"

# Crear carpeta para cachear audios
os.makedirs(AUDIO_DIR, exist_ok=True)

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
    base_x, base_y, base_z = landmarks[0]
    rel = np.array([[x - base_x, y - base_y, z - base_z] for x, y, z in landmarks])
    max_val = np.max(np.abs(rel))
    if max_val > 0:
        rel /= max_val
    return rel.flatten()

# --- FUNCIÓN PARA REPRODUCIR VOZ ---
def speak_letter(letter):
    audio_path = os.path.join(AUDIO_DIR, f"{letter}.mp3")
    if not os.path.exists(audio_path):
        tts = gTTS(letter, lang='es')  # puedes usar 'en' si las letras son del ASL
        tts.save(audio_path)
    if _audio_player == "playsound":
        playsound(audio_path)
    elif _audio_player == "vlc":
        player = vlc.MediaPlayer(audio_path)
        player.play()
        # wait until playback starts
        time.sleep(0.1)
        # block until playback finishes
        while player.is_playing():
            time.sleep(0.1)
    else:
        print(f"[Aviso] No hay reproductor de audio disponible para {audio_path}")

# --- INICIO DE CÁMARA ---
cap = cv2.VideoCapture(0)
prev_label = ""
stable_label = ""
stable_count = 0

print("🎥 Reconociendo lenguaje de señas en tiempo real... (presiona 'q' para salir)")
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

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        processed = preprocess_landmarks(landmarks).reshape(1, -1)
        processed = scaler.transform(processed)

        pred_idx = model.predict(processed)[0]
        label = idx_to_label[pred_idx]
        probas = model.predict_proba(processed)[0]
        confidence = np.max(probas)

        # Estabilización
        if label == prev_label:
            stable_count += 1
        else:
            stable_count = 0
        prev_label = label

        if stable_count > 3:  # estable por varios frames
            if label != stable_label:  # si cambió de letra
                stable_label = label
                print(f"🔤 Letra detectada: {stable_label}")
                speak_letter(stable_label)

        # Mostrar en pantalla
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
