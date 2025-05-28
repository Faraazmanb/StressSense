# StressSense

**StressSense** is a web-based application that detects **stress levels** and **emotions** from facial expressions using deep learning. It combines two models—one for stress detection and one for emotion classification—into a single intuitive dashboard built using Python Flask, HTML/CSS/JS, and Bootstrap. The application is deployed on Azure using GitHub Actions for CI/CD.

---

## 🔍 Features

- 🎭 Detects facial expressions from images
- 😫 Predicts stress levels (Stress / No Stress)
- 😀 Classifies emotions (Happy, Sad, Angry, Neutral, etc.)
- ☁️ To be Deployed on Azure App Service with GitHub Actions

---

## 🧠 Models Used

- **Stress Detection Model**
  - Architecture: Vision Transformer (ViT)
  - File: `stress_detection_model_vit.pth`
  - Dataset: Custom dataset trained
- **Emotion Detection Model**
  - Architecture: Convolutional Neural Network (CNN)
  - File: `best_model.keras`
  - Classes: Happy, Sad, Angry, Neutral, Surprised, Fearful, Disgusted

---

## 🗂️ Project Structure

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8 or later
- pip
- virtualenv (recommended)

### ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Faraazmanb/StressSense.git
   cd StressSense
2. **Create and activate a virtual environment**
    python -m venv venv
    source venv/bin/activate
3. **Install dependencies**
     pip install -r requirements.txt
4. **Run the Flask app**
     python app.py
