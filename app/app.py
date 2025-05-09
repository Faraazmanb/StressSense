from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import torch
import numpy as np
from timm import create_model
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change if needed

# -------------------------------
# MongoDB Atlas Connection            
# -------------------------------
MONGO_URI = "mongodb+srv://MohamedFaraazmn:mongoaania@cluster0.nha3a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_URI = "mongodb+srv://aania:aania@cluster0.rnrj5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["StressDetectionDB"]
users_collection = db["users"]

# -------------------------------
# Load Stress Detection Model
# -------------------------------
device = torch.device("cpu")
model = create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("stress_detection_vit.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Face Detection & Image Processing
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

current_stress_value = 0
current_no_stress_value = 0

# -------------------------------
# Function to generate video frames
# -------------------------------

def generate_frames():
    global current_stress_value, current_no_stress_value
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Extract the face region
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Perform stress classification
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
                stress_percentage = probabilities[1].item()
                no_stress_percentage = probabilities[0].item()

            # Update global stress values
            current_stress_value = stress_percentage
            current_no_stress_value = no_stress_percentage

            # Determine box color and label
            if stress_percentage > no_stress_percentage:
                label = f"Stress: {stress_percentage:.2f}%"
                color = (0, 0, 255)  # Red for stress
            else:
                label = f"No Stress: {no_stress_percentage:.2f}%"
                color = (0, 255, 0)  # Green for no stress

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Put text label above the bounding box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
# -------------------------------
# ROUTE: Landing Page
# -------------------------------
@app.route('/')
def index():
    session.clear()
    return render_template('landing.html')

# -------------------------------
# ROUTE: Login Page
# -------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials!"

    return render_template('index.html')

# -------------------------------
# ROUTE: Register Page
# -------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        # Check if username already exists
        existing_user = users_collection.find_one({"username": username})

        if existing_user:
            # If username exists but password is different, allow registration
            if not check_password_hash(existing_user["password"], password):
                users_collection.insert_one({"username": username, "password": hashed_password})
                return redirect(url_for('index'))
            else:
                return render_template('register.html', alert=True)

        # If username doesn't exist, allow registration
        users_collection.insert_one({"username": username, "password": hashed_password})
        return redirect(url_for('index'))

    return render_template('register.html', alert=False)

# -------------------------------
# ROUTE: Dashboard
# -------------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session['username'])

# -------------------------------
# ROUTE: Logout
# -------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# -------------------------------
# ROUTE: Video Feed
# -------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stress')
def get_stress():
    return jsonify({"stress": round(current_stress_value, 2), "no_stress": round(current_no_stress_value, 2)})

@app.route('/get_emotion')
def get_emotion():
    # Your existing code
    
    # If you only have one emotion, still format it as required
    top_emotions = [
        {"emotion": current_emotion, "confidence": confidence_value}
    ]
    
    return jsonify({
        "emotion": current_emotion,
        "confidence": confidence_value,
        "top_emotions": top_emotions
    })

# -------------------------------
# RUN Flask Application
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
