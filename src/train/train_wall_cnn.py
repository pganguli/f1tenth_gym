import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load dataset
# -----------------------------
data = np.load('../data/dataset.npz')
X = data['lidar']        # shape = (num_samples, 1080)
y_left = data['left_wall_dist']
y_right = data['right_wall_dist']

print("Dataset loaded:", X.shape, y_left.shape)

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Standardize per scan
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True) + 1e-6
X = (X - X_mean) / X_std

# Stack left/right distances as 2D target
y = np.vstack([y_left, y_right]).T  # shape = (num_samples, 2)

# Optional: add small Gaussian noise for augmentation
noise_std = 0.01
X += np.random.normal(0, noise_std, X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (B,1,1080)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 3. Define improved 1D CNN
# -----------------------------
class Lidar1DCNN(nn.Module):
    def __init__(self, input_len=1080):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # predict left and right distances
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

model = Lidar1DCNN()
print(model)

# -----------------------------
# 4. Training setup
# -----------------------------
criterion = nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
num_epochs = 100

# -----------------------------
# 5. Training loop
# -----------------------------
best_test_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor).item()

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        torch.save(model.state_dict(), "../data/lidar_wall_cnn.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# -----------------------------
# 6. Quick prediction
# -----------------------------
model.eval()
with torch.no_grad():
    sample = X_test_tensor[0].unsqueeze(0)
    pred = model(sample).squeeze().numpy()
    print(f"Predicted distances: left={pred[0]:.3f} m, right={pred[1]:.3f} m")
    print(f"True distances: left={y_test[0,0]:.3f} m, right={y_test[0,1]:.3f} m")
