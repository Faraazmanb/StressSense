import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from timm import create_model



# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("stress_detection_vit.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
class_names = ["No Stress", "Stress"]

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        # Extract face
        face = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = transforms.functional.to_pil_image(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Model prediction
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Convert to probabilities
            stress_prob = probabilities[1].item() * 100  # Stress probability
            no_stress_prob = probabilities[0].item() * 100  # No Stress probability
            
            label = f"Stress: {stress_prob:.1f}% | No Stress: {no_stress_prob:.1f}%"
        
        # Display result
        color = (0, 255, 0) if no_stress_prob > stress_prob else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show frame
    cv2.imshow("Live Stress Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()