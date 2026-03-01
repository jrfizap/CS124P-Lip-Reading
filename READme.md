# 👄 AI Lip Reading Dictation (Visual Speech Recognition)

A real-time, end-to-end Visual Speech Recognition (VSR) system built with Python, PyTorch, and Flask. 

This project captures a live webcam feed, isolates the user's mouth using Google's MediaPipe Face Landmarker, and feeds a 1-second rolling video buffer (29 frames) into a custom-trained **3D Convolutional Neural Network (3D CNN)**. The AI analyzes the spatio-temporal movement of the lips to predict spoken words without relying on any audio input.

## 🚀 Features
* **Real-Time Tracking:** Uses MediaPipe's modern Tasks API for sub-millisecond facial landmark detection.
* **Spatio-Temporal AI:** A lightweight PyTorch 3D CNN that analyzes the *movement* of lips over time, not just static images.
* **Live Web Interface:** Streams the processed video feed and the AI's predictions directly to a web browser using a Flask MJPEG generator.
* **Automated Data Collection:** Includes scripts to easily build your own custom video dataset and train the brain from scratch.

---

## 🧠 Architecture Overview

1. **The Extractor (MediaPipe + OpenCV):** Detects 468 3D facial landmarks, calculates the bounding box around the lips, and extracts a normalized 96x96 grayscale crop.
2. **The Buffer (Deque):** Maintains a rolling 29-frame memory (approx. 1 second of video) to capture the full sequence of a word.
3. **The Brain (PyTorch 3D CNN):** Processes the `(1, 1, 29, 96, 96)` video tensor through multiple 3D convolutional and max-pooling layers to classify the word.
4. **The Frontend (Flask):** Serves an HTML UI and streams the annotated video feed asynchronously.

---

## 🛠️ Installation & Setup

Because AI models are highly sensitive to individual facial structures and lighting, this repository does **not** include the raw dataset. You will need to collect your own data and train the model on your face using the provided scripts.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName