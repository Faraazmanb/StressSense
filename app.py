from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import torch
import numpy as np
from timm import create_model
from torchvision import transforms
from PIL import Image

import threading
import os
import requests

app = Flask(__name__)
app.secret_key = "supersecretkey"

# MongoDB Setup
MONGO_URI = "mongodb+srv://MohamedFaraazman:mongoaania@cluster0.nha3a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["StressDetectionDB"]
users_collection = db["users"]

# Download model files from Azure if not present
KERAS_MODEL_PATH = "best_model.keras"
KERAS_MODEL_URL = "https://stresssensemodels.blob.core.windows.net/models/best_model.keras?sp=r&st=2025-05-09T04:43:58Z&se=2025-05-09T12:43:58Z&spr=https&sv=2024-11-04&sr=b&sig=lDjZazmclfcupwSYFG92HyciX0EjGa5sX7s1WFlR%2FWM%3D"

if not os.path.exists(KERAS_MODEL_PATH):
    print("Downloading best_model.keras from Azure Blob Storage...")
    r = requests.get(KERAS_MODEL_URL)
    if r.status_code == 200:
        with open(KERAS_MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"Failed to download file: HTTP {r.status_code}")
        exit(1)
from keras.models import load_model
# Then load the model
emotion_model = load_model(KERAS_MODEL_PATH)

PTH_MODEL_URL = "https://stresssensemodels.blob.core.windows.net/models/stress_detection_vit.pth?sp=r&st=2025-05-08T15:26:58Z&se=2025-06-18T23:26:58Z&spr=https&sv=2024-11-04&sr=b&sig=%2Bd%2F1JDijtGeTNhnoDpl4uSpt72VSf6Y%2BZh4S0Fqd7os%3D"
PTH_MODEL_PATH = "stress_detection_vit.pth"

if not os.path.exists(PTH_MODEL_PATH):
    response = requests.get(PTH_MODEL_URL)
    with open(PTH_MODEL_PATH, "wb") as f:
        f.write(response.content)

# Models
device = torch.device("cpu")
stress_model = create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
stress_model.load_state_dict(torch.load(PTH_MODEL_PATH, map_location=device))
stress_model.to(device)
stress_model.eval()

emotion_model = load_model(KERAS_MODEL_PATH)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE = 96

# State
current_mode = "stress"  # Default mode
current_stress_value = 0
current_no_stress_value = 0
current_emotion = "neutral"
current_emotion_confidence = 0.0

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Frame Generator
def generate_frames():
    global current_stress_value, current_no_stress_value, current_emotion, current_emotion_confidence

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if current_mode == "stress":
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = stress_model(face_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
                    stress_percentage = probabilities[1].item()
                    no_stress_percentage = probabilities[0].item()

                current_stress_value = stress_percentage
                current_no_stress_value = no_stress_percentage

                label = f"Stress: {stress_percentage:.1f}%" if stress_percentage > no_stress_percentage else f"No Stress: {no_stress_percentage:.1f}%"
                color = (0, 0, 255) if stress_percentage > no_stress_percentage else (0, 255, 0)

            else:
                gray_face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE)) / 255.0
                reshaped = np.reshape(gray_face, (1, IMG_SIZE, IMG_SIZE, 1))

                preds = emotion_model.predict(reshaped, verbose=0)[0]
                max_idx = int(np.argmax(preds))
                current_emotion = EMOTIONS[max_idx]
                current_emotion_confidence = float(preds[max_idx]) * 100

                label = f"{current_emotion.capitalize()} ({current_emotion_confidence:.1f}%)"
                color = (255, 215, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------------- Routes ----------------

@app.route('/')
def index():
    session.clear()
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        return "Invalid credentials!"
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({"username": username}):
            return render_template('register.html', alert=True)
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({"username": username, "password": hashed_password})
        return redirect(url_for('index'))
    return render_template('register.html', alert=False)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stress')
def get_stress():
    return jsonify({
        "stress": round(current_stress_value, 2),
        "no_stress": round(current_no_stress_value, 2)
    })

@app.route('/get_emotion')
def get_emotion():
    return jsonify({
        "emotion": current_emotion,
        "confidence": round(current_emotion_confidence, 2)
    })

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    if mode in ["stress", "emotion"]:
        current_mode = mode
        return jsonify({"status": "success", "mode": mode})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

if __name__ == '__main__':
    app.run(debug=True)
