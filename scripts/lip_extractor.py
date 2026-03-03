import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque

# 1. Setup Folders for our 3 words
words = ["open", "close", "stop"]
for word in words:
    os.makedirs(f"dataset/{word}", exist_ok=True)

# 2. Setup Model (Updated to point to the models folder!)
model_path = 'models/face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Initialize Buffer
SEQUENCE_LENGTH = 29
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
print("\n--- LIP READING DATA COLLECTOR ---")
print("Look at the camera and say the word.")
print("Press '1' right AFTER saying 'Open'")
print("Press '2' right AFTER saying 'Close'")
print("Press '3' right AFTER saying 'Stop'")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    lips_resized = None

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
            cv2.imshow("Extracted Lips", lips_resized)

    cv2.imshow("Main Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # 4. Save the data if a key is pressed
    if key in [ord('1'), ord('2'), ord('3')]:
        if len(frame_buffer) == SEQUENCE_LENGTH:
            video_sequence = np.array(frame_buffer)
            
            # Figure out which word was pressed
            if key == ord('1'): word_idx = 0
            elif key == ord('2'): word_idx = 1
            elif key == ord('3'): word_idx = 2
            
            target_word = words[word_idx]
            
            # Create a unique filename based on the timestamp
            filename = f"dataset/{target_word}/{target_word}_{int(time.time())}.npy"
            np.save(filename, video_sequence)
            
            print(f"✅ Saved 1 example for: {target_word.upper()}")
            
            # Clear the buffer so we don't accidentally save the same video twice
            frame_buffer.clear() 
        else:
            print(f"Buffer not full! Wait a second before pressing.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()