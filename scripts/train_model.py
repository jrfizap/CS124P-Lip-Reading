import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

print("Loading dataset...")

# 1. Load the Data
words = ["open", "close", "stop"]
X = []
y = []

for label, word in enumerate(words):
    folder = f"dataset/{word}"
    if not os.path.exists(folder):
        continue
        
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            # Load the 29-frame video block
            data = np.load(os.path.join(folder, file)) 
            
            # PyTorch expects a "Channel" dimension (like RGB). 
            # Since we use grayscale, we add 1 channel: (1, 29, 96, 96)
            data = np.expand_dims(data, axis=0) 
            X.append(data)
            y.append(label)

# Convert our lists to PyTorch math tensors
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

print(f"✅ Loaded {len(X)} total video examples!")
print(f"Data shape: {X.shape} -> (Batch, Channels, Frames, Height, Width)")

# 2. Build the 3D Neural Network
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First 3D Convolution Layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        # Pooling layer to compress the data and focus on the most important features
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 4, 4)) 
        
        # Second 3D Convolution Layer
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 4, 4))
        
        # Fully Connected (Dense) layers to make the final guess
        # After pooling, our 29x96x96 video is compressed down to 7x6x6
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 3) # 3 output neurons (for our 3 words)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1) # Flatten the 3D block into a 1D list
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MiniLipNet()

# 3. Setup the AI's "Teacher" (Loss function and Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the AI!
print("\n🧠 Starting Training...\n")
epochs = 25
train_losses = [] # NEW: We will store the error history here!

for epoch in range(epochs):
    optimizer.zero_grad()    # Clear old guesses
    outputs = model(X)       # Make a guess on the videos
    loss = criterion(outputs, y) # Grade the guess (How wrong was it?)
    
    loss.backward()          # Calculate how to fix the mistakes
    optimizer.step()         # Adjust the brain weights
    
    train_losses.append(loss.item()) # Save the error for our graph
    print(f"Epoch {epoch+1}/{epochs} | Error (Loss): {loss.item():.4f}")

# 5. Save the Graph
os.makedirs("static", exist_ok=True) 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='#3498db', linewidth=2)
plt.title('Training Loss Plateau')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('static/loss_curve.png')
plt.close()

# 6. Save the Brain
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lip_model.pth")
print("\n🎉 Training Complete! Model saved as 'models/lip_model.pth'")
print("📈 Graph saved as 'static/loss_curve.png'")

# 7. Extract Training Metrics for the Web Dashboard
with torch.no_grad():
    preds = torch.argmax(model(X), dim=1)
    correct_guesses = (preds == y).sum().item()
    training_accuracy = (correct_guesses / len(y)) * 100

metrics_data = {
    "accuracy": round(training_accuracy, 2),
    "config": {
        "split": "100% Training",
        "weights": "PyTorch Default",
        "epochs": epochs,
        "optimizer": "Adam (LR=0.001)"
    }
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics_data, f)