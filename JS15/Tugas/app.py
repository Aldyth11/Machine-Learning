import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, render_template_string
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# Load Model, Scaler, dan Label
MODEL_PATH = 'bisindo_model.h5'
SCALER_PATH = 'scaler.pkl'
LABELS_PATH = 'labels.pkl'
LANDMARKER_PATH = 'hand_landmarker.task'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABELS_PATH, 'rb') as f:
        classes = pickle.load(f)
    print("System Loaded Successfully")
except Exception as e:
    model = None
    scaler = None
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    print(f"Error loading system: {e}")

# Inisialisasi MediaPipe HandLandmarker
detector = None
try:
    if os.path.exists(LANDMARKER_PATH):
        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.15
        )
        detector = vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error MediaPipe: {e}")

def preprocess_image(image_bytes):
    # Decode gambar
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or detector is None:
        raise ValueError("Gambar tidak valid atau detektor belum siap.")
    
    # Deteksi landmark tangan
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    res = detector.detect(mp_img)
    
    if not res.hand_landmarks:
        raise ValueError("Tidak ada tangan yang terdeteksi pada citra.")
    
    # Ekstraksi koordinat tangan primer dan sekunder
    raw_hands = [np.array([[lm.x, lm.y, lm.z] for lm in h], dtype=np.float32) for h in res.hand_landmarks]
    primary_hand = raw_hands[0]
    secondary_hand = raw_hands[1] if len(raw_hands) > 1 else None
    is_two_handed = 1.0 if len(raw_hands) > 1 else 0.0
    
    # Normalisasi translasi & skala telapak
    origin = primary_hand[0].copy()
    scale = np.linalg.norm(primary_hand[9] - origin) or 1.0
    
    feat_primary = ((primary_hand - origin) / scale).flatten()
    feat_secondary = ((secondary_hand - origin) / scale).flatten() if secondary_hand is not None else np.zeros(63, dtype=np.float32)
    
    # 127 fitur geometris
    feature_vector = np.concatenate([feat_primary, feat_secondary, [is_two_handed]])
    return scaler.transform(feature_vector.reshape(1, -1))

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <div style="text-align:center; padding:50px; font-family:sans-serif;">
        <h1>BISINDO Sign Classifier</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <button type="submit">Prediksi</button>
        </form>
    </div>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = preprocess_image(file.read())
        probs = model.predict(data, verbose=0)[0]
        
        pred_idx = int(np.argmax(probs))
        label = classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)
        
        return f"<h2 style='text-align:center; font-family:sans-serif;'>Hasil: Huruf {label} ({confidence}%)</h2><center><a href='/'>Kembali</a></center>"
    except Exception as e:
        return f"<div style='text-align:center; font-family:sans-serif;'><h3>Error: {e}</h3><br><a href='/'>Kembali</a></div>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
