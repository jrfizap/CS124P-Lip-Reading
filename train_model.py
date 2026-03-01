import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

print("Loading dataset...")

# 1. Load the Data (NOW WITH SILENCE!)
words = ["open", "close", "stop", "silence"]
X = []
y = []

for label, word in enumerate(words):
    folder = f"dataset/{word}"
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Could not find folder for '{word}'. Did you record data for it?")
        continue
        
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file)) 
            data = np.expand_dims(data, axis=0) 
            X.append(data)
            y.append(label)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

print(f"✅ Loaded {len(X)} total video examples!")

# Split data: 80% for studying (train), 20% for the final exam (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the 3D Neural Network (Updated for 4 words)
class MiniLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d((2, 4, 4)) 
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 4, 4))
        self.fc1 = nn.Linear(16 * 7 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4) # NOW 4 OUTPUT NEURONS
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

# 3. Weight Initialization
INIT_METHOD = "heuristic" # Change to "general", "xavier", or "heuristic"

def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        if INIT_METHOD == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif INIT_METHOD == "heuristic":
            # Kaiming (He) Initialization is best for ReLU activations
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # 'general' just leaves PyTorch's default alone
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
print(f"\n🧠 Applied '{INIT_METHOD.upper()}' weight initialization.")

# Sample the initial weights of the very first math filter
initial_weight_sample = model.conv1.weight[0][0][0][0].detach().numpy().copy()
print(f"Initial Weight Sample: {initial_weight_sample}")

# 4. Setup the AI's "Teacher"
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train with Early Stopping
print("\n🚀 Starting Training...\n")
epochs = 100
patience = 15 # Stop if it doesn't improve for 15 epochs
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train() # Study mode
    optimizer.zero_grad()    
    outputs = model(X_train)       
    loss = criterion(outputs, y_train) 
    
    loss.backward()          
    optimizer.step()         
    
    # Early Stopping Check
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        # Save the absolute best version of the brain
        torch.save(model.state_dict(), "lip_model.pth")
    else:
        patience_counter += 1
        
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Patience: {patience_counter}/{patience}")
        
    if patience_counter >= patience:
        print(f"\n🛑 Early Stopping triggered at Epoch {epoch+1}! The model stopped improving.")
        break

# 6. Evaluate the Model on the Test Set
print("\n📊 TAKING THE FINAL EXAM (Test Set)...")
model.load_state_dict(torch.load("lip_model.pth", weights_only=True)) # Load the best weights
model.eval() # Testing mode

with torch.no_grad():
    test_predictions = model(X_test)
    predicted_classes = torch.argmax(test_predictions, dim=1).numpy()
    true_classes = y_test.numpy()

# Calculate Metrics 
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
kappa = cohen_kappa_score(true_classes, predicted_classes)

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f} (When it guesses a word, how often is it right?)")
print(f"Recall:    {recall:.4f} (Out of all times you said a word, how many did it catch?)")
print(f"F1 Score:  {f1:.4f} (Balance between Precision and Recall)")
print(f"Kappa:     {kappa:.4f} (How much better the AI is than just randomly guessing)")

# Sample the final weights to see how they changed!
final_weight_sample = model.conv1.weight[0][0][0][0].detach().numpy()
print(f"\nFinal Weight Sample: {final_weight_sample}")
print("Notice how the math actually shifted to learn your face!")
print("\n🎉 Training Complete! Best model saved as 'lip_model.pth'")