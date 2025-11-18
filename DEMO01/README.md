# DEMO01 — Text and Voice

Este directorio contiene un demo que captura la cámara, reconoce lenguaje de señas (con MediaPipe + un modelo entrenado) y reproduce la letra detectada en voz usando `gTTS`.

Archivos importantes:
- `textandvoice.py`: script principal que abre la cámara, detecta manos y reproduce audio.
- `models/hand_sign_mlp.pkl` y `models/scaler.pkl`: modelo y scaler usados por el script.
- `data_prepared/label_map.npy`: mapa de etiquetas.
- `audio_cache/`: carpeta donde se guardan los mp3 generados.

Pasos para ejecutar (Linux)

1) Abrir terminal y ubicarse en el directorio:

```bash
cd /workspaces/ProyectoTraductor/DEMO01
```

2) Crear y activar un entorno virtual (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) Ejecutar el demo:

```bash
python textandvoice.py
```

5) Uso y salida:
- Se abre una ventana con la cámara. Mantén la letra con la mano hasta que el script estime una predicción estable.
- El audio se guarda en `audio_cache/` como `<letra>.mp3` y se reproduce automáticamente.
- Presiona `q` en la ventana para finalizar.

Notas y resolución de problemas
- Si `playsound` no reproduce el audio en tu sistema, prueba reproducir manualmente con `mpg123`:

```bash
sudo apt update && sudo apt install -y mpg123
mpg123 audio_cache/A.mp3
```
- Alternativamente `python-vlc` está incluido en `requirements.txt` y puedes editar `textandvoice.py` para usarlo si `playsound` falla.
- Si `mediapipe` falla al instalar, revisa los logs; en algunas distribuciones puede necesitar paquetes del sistema o una rueda precompilada.
- Asegúrate de que la cámara esté disponible y no esté en uso por otra aplicación.

Si quieres, puedo:
- Ejecutar el script aquí para verificar (si el contenedor tiene acceso a una cámara), o
- Modificar `textandvoice.py` para usar `python-vlc` como respaldo en Linux.
