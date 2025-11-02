import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sqlite3

# Emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_data(data_dir='datasets'):
    images = []
    labels = []
    for emotion in EMOTIONS:
        folder = os.path.join(data_dir, emotion)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(folder, file)
                    try:
                        img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(emotion)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    if len(images) == 0:
        raise ValueError("No images found in datasets/. Add labeled images (48x48 grayscale).")
    return np.array(images), np.array(labels)

# Load and prepare data
X, y = load_data()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = pd.get_dummies(y_encoded).values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape for grayscale
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Save model
model.save('emotion_detector_v1.h5')
print("Model saved as emotion_detector_v1.h5")

# Save label encoder (simple pickle alternative - save classes)
with open('emotions_classes.txt', 'w') as f:
    f.write(','.join(EMOTIONS))