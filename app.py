from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import os
import json
import time
from collections import deque

app = Flask(__name__)

# --- 1. SETUP AI AND MEDIAPIPE ---
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((2, 4, 4))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 4, 4))
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4) 
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

words = ["thank you", "hello", "goodbye", "silence"]
model = MiniLipNet()
try:
    model.load_state_dict(torch.load("models/lip_model.pth", weights_only=True))
    model.eval() 
except: pass

base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 2. GLOBAL VARIABLES FOR WEB STREAMING ---
SEQUENCE_LENGTH = 29
global_frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# --- 3. VIDEO GENERATOR FUNCTION ---
def generate_frames(mode="translator"):
    cap = cv2.VideoCapture(0)
    current_prediction = "Listening..."
    display_timer = 0 

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

        # AI LOGIC (Only runs if we are on the Translator page)
        if mode == "translator" and len(global_frame_buffer) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.expand_dims(np.array(global_frame_buffer), axis=0), axis=0) 
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            
            with torch.no_grad():
                pred = model(input_tensor)
                predicted_word = words[torch.argmax(pred).item()]
                if predicted_word != "silence":
                    current_prediction = f"You said: {predicted_word.upper()}"
                    display_timer = 40 
            global_frame_buffer.clear() 

        # DRAW UI
        if mode == "translator":
            if display_timer > 0:
                cv2.putText(frame, current_prediction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                display_timer -= 1
            else:
                cv2.putText(frame, "Listening...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        elif mode == "collector":
            buffer_status = f"Buffer: {len(global_frame_buffer)}/{SEQUENCE_LENGTH}"
            color = (0, 255, 0) if len(global_frame_buffer) == SEQUENCE_LENGTH else (0, 0, 255)
            cv2.putText(frame, buffer_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    cap.release()

# --- 4. WEB ROUTES ---
@app.route('/')
def home(): return render_template('home.html')

@app.route('/collector')
def collector(): return render_template('collector.html')

@app.route('/results')
def results():
    metrics = None
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
    return render_template('results.html', metrics=metrics)

@app.route('/translator')
def translator(): return render_template('translator.html')

@app.route('/video_feed_collector')
def video_feed_collector(): return Response(generate_frames(mode="collector"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_translator')
def video_feed_translator(): return Response(generate_frames(mode="translator"), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. DATA SAVING ENDPOINT ---
@app.route('/save_word/<word>', methods=['POST'])
def save_word(word):
    if len(global_frame_buffer) == SEQUENCE_LENGTH:
        os.makedirs(f"dataset/{word}", exist_ok=True)
        filename = f"dataset/{word}/{word}_{int(time.time())}.npy"
        np.save(filename, np.array(global_frame_buffer))
        global_frame_buffer.clear()
        return jsonify({"message": f"✅ Saved 1 example for {word.upper()}!"})
    return jsonify({"message": "⚠️ Buffer not full yet! Wait a second."})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)