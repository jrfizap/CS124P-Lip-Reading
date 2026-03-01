import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

print("Loading dataset...")

# 1. Load the Data
words = ["open", "close", "stop", "silence"]
X = []
y = []

for label, word in enumerate(words):
    folder = f"dataset/{word}"
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Could not find folder for '{word}'.")
        continue
        
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file)) 
            data = np.expand_dims(data, axis=0) 
            X.append(data)
            y.append(label)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Definition
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

# 3. Initialization & Setup
def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

model.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop with Comparison
print("\n🚀 Starting Training...\n")
epochs = 100
patience = 15 
best_test_loss = float('inf')
patience_counter = 0

train_loss_history = []
test_loss_history = []

os.makedirs("models", exist_ok=True)
model_save_path = "models/lip_model.pth"

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train() 
    optimizer.zero_grad()    
    train_outputs = model(X_train)       
    train_loss = criterion(train_outputs, y_train) 
    train_loss.backward()          
    optimizer.step()         
    
    # --- TESTING/VALIDATION PHASE ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    
    # Track history
    train_loss_history.append(train_loss.item())
    test_loss_history.append(test_loss.item())
    
    # Early Stopping based on TEST loss (prevents overfitting)
    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
        
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1} | Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f} | P: {patience_counter}")
        
    if patience_counter >= patience:
        print(f"\n🛑 Early Stopping! Test loss stopped improving.")
        break

# --- GENERATE COMPARISON GRAPH ---
os.makedirs("static", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Loss (Learning)', color='#3498db', linewidth=2)
plt.plot(test_loss_history, label='Test Loss (Generalizing)', color='#e74c3c', linewidth=2, linestyle='--')
plt.title('AI Learning vs. Testing Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('static/loss_curve.png')
plt.close()
print("📈 Comparison graph saved to 'static/loss_curve.png'")

# 6. Final Evaluation
model.load_state_dict(torch.load(model_save_path, weights_only=True)) 
model.eval() 
with torch.no_grad():
    test_predictions = model(X_test)
    predicted_classes = torch.argmax(test_predictions, dim=1).numpy()
    true_classes = y_test.numpy()

accuracy = accuracy_score(true_classes, predicted_classes)
# (Rest of your metrics code remains the same...)
import json
metrics_data = {"accuracy": round(accuracy * 100, 2)} # Shortened for brevity
with open("models/metrics.json", "w") as f: json.dump(metrics_data, f)
print(f"\n🎉 Training Complete!")