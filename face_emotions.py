import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and face cascade
model = load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Take the first face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype('float') / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = np.expand_dims(roi_gray, axis=3)
    prediction = model.predict(roi_gray)[0]
    label = classes[np.argmax(prediction)]
    return label