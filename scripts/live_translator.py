import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# 1. The Brain Structure (Updated to 4 Neurons!)
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((2, 4, 4))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 4, 4))
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4) # Now 4 classes: open, close, stop, silence
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

# 2. Wake up the AI (Updated to point to the 'models' folder)
print("Waking up the AI Brain...")
words = ["open", "close", "stop", "silence"]
model = MiniLipNet()
model.load_state_dict(torch.load("models/lip_model.pth", weights_only=True))
model.eval() 

# 3. Setup MediaPipe (Updated to point to the 'models' folder)
model_path = 'models/face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. Setup Live Stream Variables
SEQUENCE_LENGTH = 29
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

current_prediction = ""
display_timer = 0 

cap = cv2.VideoCapture(0)
print("\n🎤 HANDS-FREE LIVE LIP READER READY!")
print("Just look at the camera and mouth your words.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

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
            lips_normalized = lips_resized / 255.0
            frame_buffer.append(lips_normalized)
            cv2.imshow("AI Vision", lips_resized)

    # 5. The Automatic Trigger 
    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_data = np.array(frame_buffer)
        input_data = np.expand_dims(input_data, axis=0) 
        input_data = np.expand_dims(input_data, axis=0) 
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            best_guess_index = torch.argmax(prediction).item()
            predicted_word = words[best_guess_index]
            
            # Smart UI Logic: Handle the silence!
            if predicted_word == "silence":
                display_timer = 0 # Don't display a flashy alert
            else:
                current_prediction = f"AI Hears: {predicted_word.upper()}"
                display_timer = 40 # Keep the word on screen for a moment
            
        # Instantly clear the buffer to start listening for the next word
        frame_buffer.clear() 

    # 6. Draw the AI's guess on the screen
    if display_timer > 0:
        cv2.putText(frame, current_prediction, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        display_timer -= 1
    else:
        # Default state when sitting in silence
        cv2.putText(frame, "Listening...", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Main Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()