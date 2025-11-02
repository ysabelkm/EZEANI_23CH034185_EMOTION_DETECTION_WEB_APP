from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
try:
    model = load_model('emotion_detector_v1.h5')
except Exception as e:
    print("Error loading model:", e)
    raise

with open('emotions_classes.txt', 'r') as f:
    EMOTIONS = f.read().strip().split(',')

# Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Database setup
DB_NAME = 'user_emotions.db'
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY, username TEXT, image_path TEXT, emotion TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_detection(username, image_path, emotion):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO detections (username, image_path, emotion, timestamp) VALUES (?, ?, ?, ?)",
              (username, image_path, emotion, timestamp))
    conn.commit()
    conn.close()

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    # First face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype('float') / 255.0
    roi_gray = np.expand_dims(np.expand_dims(roi_gray, axis=0), axis=3)
    prediction = model.predict(roi_gray, verbose=0)[0]
    return EMOTIONS[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    username = request.form.get('username', 'Anonymous')
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Secure filename and prepare paths
    original_name = secure_filename(file.filename)
    timestamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_name}"
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', timestamped_name)

    # Read bytes, save, and decode for OpenCV
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({'error': 'Empty file'}), 400
    try:
        with open(filepath, 'wb') as f:
            f.write(file_bytes)
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Unable to decode image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    emotion = detect_emotion(image)

    if emotion:
        save_detection(username, filepath, emotion)
        # Save to datasets for retraining
        datasets_dir = os.path.join('datasets', emotion)
        os.makedirs(datasets_dir, exist_ok=True)
        datasets_path = os.path.join(datasets_dir, os.path.basename(filepath))
        cv2.imwrite(datasets_path, image)
        return jsonify({'emotion': emotion, 'image_path': filepath})
    return jsonify({'error': 'No face detected'}), 400

@app.route('/capture', methods=['POST'])
def capture_image():
    username = request.form.get('username', 'Anonymous')
    image_data = request.form.get('image')
    if not image_data:
        return jsonify({'error': 'No image data'}), 400

    # Decode base64 (handle data URLs)
    try:
        if ',' in image_data:
            header, b64 = image_data.split(',', 1)
        else:
            b64 = image_data
        image_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Unable to decode image'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {e}'}), 400

    # Save
    os.makedirs('uploads', exist_ok=True)
    timestamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_webcam.jpg"
    filepath = os.path.join('uploads', timestamped_name)
    cv2.imwrite(filepath, image)

    emotion = detect_emotion(image)
    if emotion:
        save_detection(username, filepath, emotion)
        datasets_dir = os.path.join('datasets', emotion)
        os.makedirs(datasets_dir, exist_ok=True)
        datasets_path = os.path.join(datasets_dir, os.path.basename(filepath))
        cv2.imwrite(datasets_path, image)
        return jsonify({'emotion': emotion, 'image_path': filepath})
    return jsonify({'error': 'No face detected'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/history')
def history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, emotion, timestamp FROM detections ORDER BY timestamp DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', detections=rows)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    app.run(debug=True)