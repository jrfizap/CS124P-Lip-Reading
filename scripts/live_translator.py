import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# 1. We have to define the exact same Brain structure so PyTorch knows how to load the weights
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((2, 4, 4))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 4, 4))
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 3)
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

# 2. Wake up the AI and load your trained weights
print("Waking up the AI Brain...")
words = ["open", "close", "stop"]
model = MiniLipNet()
# Load the file you just created!
model.load_state_dict(torch.load("models/lip_model.pth", weights_only=True))
model.eval() # Tell the AI it is time to take the test, not study

# 3. Setup MediaPipe (The Eyes)
model_path = 'models/face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. Start the Live Stream
SEQUENCE_LENGTH = 29
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
print("\n🎤 LIVE LIP READER READY!")
print("Look at the camera, say ONE of your 3 words, and immediately hit SPACEBAR.")
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

    cv2.imshow("Main Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # 5. The Moment of Truth!
    if key == ord(' '):
        if len(frame_buffer) == SEQUENCE_LENGTH:
            # Package the 29 frames for PyTorch (Add Channel and Batch dimensions)
            # Shape becomes: (1 Batch, 1 Channel, 29 Frames, 96 Height, 96 Width)
            input_data = np.array(frame_buffer)
            input_data = np.expand_dims(input_data, axis=0) 
            input_data = np.expand_dims(input_data, axis=0) 
            
            # Convert to PyTorch Math
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            
            # Ask the AI to guess!
            with torch.no_grad(): # Don't train, just guess
                prediction = model(input_tensor)
                
                # Find the word with the highest score
                best_guess_index = torch.argmax(prediction).item()
                predicted_word = words[best_guess_index]
                
                print(f"\n🤖 AI Says you said: >>> {predicted_word.upper()} <<<")
                
            frame_buffer.clear() # Clear the buffer for the next word
        else:
            print("Buffer not full yet!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()