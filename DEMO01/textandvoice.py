import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from gtts import gTTS
try:
    from playsound import playsound
    _audio_player = "playsound"
except:
    try:
        import vlc
        _audio_player = "vlc"
    except:
        _audio_player = None

<<<<<<< HEAD
# --- PRONUNCIACIÓN PERSONALIZADA (opcional) ---
=======
# ---------------- CONFIG ----------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_mlp.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_MAP_PATH = "data_prepared/label_map.npy"
AUDIO_DIR = "audio_cache"

os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------- PRONUNCIACIÓN ----------------
>>>>>>> 22f356b (Ignore virtualenv)
PRONUNCIATION_MAP = {
    "A": "aaa",
    "B": "beh",
    "C": "seh",
    "D": "deh",
    "E": "eh",
    "F": "fff",
    "G": "gueh",
    "H": "ahh",
    "I": "iii",
    "J": "jhh",
    "K": "kah",
    "L": "lll",
    "M": "mmm",
    "N": "nnn",
    "O": "ooo",
    "P": "peh",
    "Q": "kuu",
    "R": "rrr",
    "S": "sss",
    "T": "teh",
    "U": "uuu",
    "V": "veh",
    "W": "uve doble",
    "X": "ekis",
    "Y": "ye",
    "Z": "zzzz"
}

<<<<<<< HEAD
# --- CONFIGURACIÓN ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_mlp.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_MAP_PATH = "data_prepared/label_map.npy"
AUDIO_DIR = "audio_cache"

os.makedirs(AUDIO_DIR, exist_ok=True)

print("�� Cargando modelo y scaler...")
=======
# ---------------- Cargar Modelo ----------------
print("📦 Cargando modelo y scaler...")
>>>>>>> 22f356b (Ignore virtualenv)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
idx_to_label = {v: k for k, v in label_map.items()}

<<<<<<< HEAD
# --- CONFIGURAR MEDIAPIPE ---
=======
# ---------------- Mediapipe ----------------
>>>>>>> 22f356b (Ignore virtualenv)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

<<<<<<< HEAD
# --- FUNCIÓN DE PREPROCESAMIENTO ---
=======
# ---------------- Preprocesamiento ----------------
>>>>>>> 22f356b (Ignore virtualenv)
def preprocess_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    rel = np.array([[x - base_x, y - base_y, z - base_z] for x, y, z in landmarks])
    max_val = np.max(np.abs(rel))
    if max_val > 0:
        rel /= max_val
    return rel.flatten()

<<<<<<< HEAD
# --- FUNCIÓN PARA REPRODUCIR VOZ (usa PRONUNCIATION_MAP si está disponible) ---
=======
# ---------------- Sonido ----------------
>>>>>>> 22f356b (Ignore virtualenv)
def speak_letter(letter):
    sound = PRONUNCIATION_MAP.get(letter.upper(), letter)
    audio_path = os.path.join(AUDIO_DIR, f"{letter}.mp3")

    if not os.path.exists(audio_path):
<<<<<<< HEAD
=======
        print(f"🔊 Generando sonido para {letter}: {sound}")
>>>>>>> 22f356b (Ignore virtualenv)
        tts = gTTS(sound, lang='es')
        tts.save(audio_path)

    if _audio_player == "playsound":
        playsound(audio_path)
    elif _audio_player == "vlc":
        player = vlc.MediaPlayer(audio_path)
        player.play()
        time.sleep(0.1)
        while player.is_playing():
            time.sleep(0.1)
<<<<<<< HEAD

# --- INICIO DE CÁMARA ---
cap = cv2.VideoCapture(0)

# Reducir resolución para más velocidad
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

=======
    else:
        print(f"[Aviso] No hay reproductor de audio disponible para {audio_path}")

# ---------------- Cámara ----------------
cap = cv2.VideoCapture(0)
>>>>>>> 22f356b (Ignore virtualenv)
prev_label = ""
stable_label = ""
stable_count = 0

<<<<<<< HEAD
# para no predecir en todos los frames
frame_counter = 0

=======
>>>>>>> 22f356b (Ignore virtualenv)
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

<<<<<<< HEAD
        # Estabilización leve para evitar cambios bruscos
=======
        # Estabilizar detección
>>>>>>> 22f356b (Ignore virtualenv)
        if label == prev_label:
            stable_count += 1
        else:
            stable_count = 0
        prev_label = label

        if stable_count > 3:
            if label != stable_label:
                stable_label = label
                print(f"🔤 Letra detectada: {stable_label}")
                speak_letter(stable_label)

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
