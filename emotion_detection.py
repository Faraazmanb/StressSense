# C:\Users\user\Desktop\MZ Code\emotion_detection.py
import cv2
import numpy as np
import tensorflow as tf
from keras import models

# Constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE = 96

def preprocess_image(img):
    """Apply image preprocessing"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img.astype(np.uint8))
    
    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    return img

def detect_emotions_realtime():
    # Load the trained model
    try:
        model = models.load_model('best_model.keras')
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Make sure 'best_model.keras' exists in the current directory.")
        return

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time emotion detection... Press 'q' to quit.")

    # Initialize variables for FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Create a copy for display
        display_frame = frame.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract and preprocess face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = preprocess_image(face_roi)
            face_roi = np.expand_dims(face_roi, axis=[0, -1])

            # Predict emotion
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = EMOTIONS[emotion_idx]
            confidence = prediction[0][emotion_idx]

            # Display emotion and confidence
            label = f"{emotion}: {confidence:.2%}"
            label_position = (x, y - 10)
            cv2.putText(display_frame, label, label_position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display emotion probabilities bar chart
            bar_width = 100
            bar_height = 15
            spacing = 20
            start_x = x + w + 10
            start_y = y

            for i, (emotion, prob) in enumerate(zip(EMOTIONS, prediction[0])):
                # Draw emotion label
                cv2.putText(display_frame, emotion, 
                           (start_x, start_y + i*spacing + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw probability bar
                bar_length = int(prob * bar_width)
                cv2.rectangle(display_frame,
                            (start_x + 60, start_y + i*spacing),
                            (start_x + 60 + bar_length, start_y + i*spacing + bar_height),
                            (0, 255, 0), -1)

        # Calculate and display FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps = 30 * cv2.getTickFrequency() / (fps_end_time - fps_start_time)
            fps_counter = 0
            fps_start_time = fps_end_time

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Emotion Detection', display_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_realtime()