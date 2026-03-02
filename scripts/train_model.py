import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import json

# --- CONFIGURATION ---
WORDS = ["thank you", "hello", "goodbye", "silence"]
INIT_METHOD = "kaiming_heuristic" 
TEST_RATIO = 0.3 # 70/30 Split
MAX_EPOCHS = 100
PATIENCE = 15

print(f"Loading dataset for: {WORDS}...")

X, y = [], []
for label, word in enumerate(WORDS):
    folder = f"dataset/{word}"
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder '{word}' not found.")
        continue
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file)) 
            X.append(np.expand_dims(data, axis=0))
            y.append(label)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

# 70/30 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)

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

model = MiniLipNet()

def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

model.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training lists
train_losses, test_losses = [], []
best_loss = float('inf')
patience_counter = 0
actual_epochs = 0

print("\n🚀 Training with 70/30 Split...")

for epoch in range(MAX_EPOCHS):
    actual_epochs += 1
    # Training Phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation Phase (Testing)
    model.eval()
    with torch.no_grad():
        t_outputs = model(X_test)
        t_loss = criterion(t_outputs, y_test)
        test_losses.append(t_loss.item())

    # Early Stopping check on Validation Loss
    if t_loss.item() < best_loss:
        best_loss = t_loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), "models/lip_model.pth")
    else:
        patience_counter += 1

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Test Loss: {t_loss.item():.4f}")

    if patience_counter >= PATIENCE:
        print(f"🛑 Early Stopping at Epoch {epoch+1}")
        break

# --- SAVE COMPARISON GRAPH ---
os.makedirs("static", exist_ok=True) 

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='#3498db', linewidth=2)
plt.plot(test_losses, label='Testing Loss', color='#e74c3c', linewidth=2, linestyle='--')
plt.title('Training vs Testing Loss (Plateau Comparison)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('static/loss_curve.png')
plt.close()

# --- EVALUATE & SAVE METADATA ---
model.load_state_dict(torch.load("models/lip_model.pth", weights_only=True))
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1).numpy()
    true = y_test.numpy()

# Calculate all metrics including Kappa
accuracy = accuracy_score(true, preds)
precision = precision_score(true, preds, average='weighted', zero_division=0)
recall = recall_score(true, preds, average='weighted', zero_division=0)
f1 = f1_score(true, preds, average='weighted', zero_division=0)
kappa = cohen_kappa_score(true, preds)

metrics_data = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "kappa": round(kappa, 4),
    "config": {
        "split": "70% Training / 30% Testing",
        "weights": "Kaiming He (Heuristic)",
        "epochs": actual_epochs,
        "optimizer": "Adam (LR=0.001)"
    }
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics_data, f)