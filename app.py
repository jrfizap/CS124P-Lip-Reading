from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import json
import time

app = Flask(__name__)

# --- BULLETPROOF PATHING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
dataset_dir = os.path.join(script_dir, "dataset")

# --- 1. SETUP AI BRAIN (3 Neurons) ---
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((2, 4, 4))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 4, 4))
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 3) # 3 Words: Open, Close, Stop
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

words = ["open", "close", "stop"]
model = MiniLipNet()
try:
    model_path = os.path.join(models_dir, "lip_model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() 
    print("✅ AI Brain loaded successfully!")
except Exception as e: 
    print(f"⚠️ Could not load AI Brain. Did you train it yet? Error: {e}")

# --- 2. SETUP MEDIAPIPE (The Eyes) ---
landmarker_path = os.path.join(models_dir, 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=landmarker_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 3. GLOBAL BUFFER ---
SEQUENCE_LENGTH = 29
global_frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# --- 4. VIDEO GENERATOR (Handles both Collector and Translator UI) ---
def generate_frames(mode="translator"):
    cap = cv2.VideoCapture(0)
    try: 
        while True:
            success, frame = cap.read()
            if not success: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            if len(result.face_landmarks) > 0:
                face_landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape
                lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
                lip_points = np.array([[int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)] for idx in lip_indices])
                
                x_min, y_min = np.min(lip_points, axis=0)
                x_max, y_max = np.max(lip_points, axis=0)
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                lips_crop = frame[y_min:y_max, x_min:x_max]
                if lips_crop.size > 0:
                    lips_gray = cv2.cvtColor(lips_crop, cv2.COLOR_BGR2GRAY)
                    lips_resized = cv2.resize(lips_gray, (96, 96))
                    global_frame_buffer.append(lips_resized / 255.0)

            # Draw different UI text depending on which webpage is watching
            if mode == "collector":
                buffer_status = f"Buffer: {len(global_frame_buffer)}/{SEQUENCE_LENGTH}"
                color = (0, 255, 0) if len(global_frame_buffer) == SEQUENCE_LENGTH else (0, 0, 255)
                cv2.putText(frame, buffer_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Send the frame to the webpage
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    finally:
        cap.release()

# --- 5. ROUTES ---

# THIS IS THE FIX: The root route now points to home.html
@app.route('/')
def home(): 
    return render_template('home.html') 

@app.route('/collector')
def collector(): 
    return render_template('collector.html')

@app.route('/results')
def results():
    metrics = None
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return render_template('results.html', metrics=metrics)

@app.route('/translator')
def translator(): 
    return render_template('translator.html')

# Separate video feeds so the camera behaves differently on each page
@app.route('/video_feed_collector')
def video_feed_collector(): 
    return Response(generate_frames(mode="collector"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_translator')
def video_feed_translator(): 
    return Response(generate_frames(mode="translator"), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 6. DATA COLLECTOR ENDPOINT ---
@app.route('/save_word/<word>', methods=['POST'])
def save_word(word):
    if len(global_frame_buffer) == SEQUENCE_LENGTH:
        word_dir = os.path.join(dataset_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        filename = os.path.join(word_dir, f"{word}_{int(time.time())}.npy")
        
        np.save(filename, np.array(global_frame_buffer))
        global_frame_buffer.clear()
        return jsonify({"message": f"✅ Saved 1 example for {word.upper()}!"})
        
    return jsonify({"message": "⚠️ Buffer not full yet! Wait a second."})

# --- 7. TRANSLATOR ENDPOINT (Triggered by HTML Spacebar) ---
@app.route('/manual_translate', methods=['POST'])
def manual_translate():
    if len(global_frame_buffer) == SEQUENCE_LENGTH:
        
        # Package the data for PyTorch
        input_data = np.array(global_frame_buffer)
        input_data = np.expand_dims(input_data, axis=0) 
        input_data = np.expand_dims(input_data, axis=0) 
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Ask the AI
        with torch.no_grad():
            prediction = model(input_tensor)
            best_guess_index = torch.argmax(prediction).item()
            predicted_word = words[best_guess_index]
                
        # Clear buffer so it starts fresh for the next word
        global_frame_buffer.clear()
        
        # Send the answer back to the HTML JavaScript
        return jsonify({"word": predicted_word})
    
    return jsonify({"word": "buffer_empty"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)