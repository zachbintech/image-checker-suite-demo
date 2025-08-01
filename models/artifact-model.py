import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score

# --- Define the CNN ---
class ArtifactNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 16, 64, 64]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 32, 32, 32]
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Use raw logits

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
IMAGE_SIZE = (128, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# --- Load Datasets ---
data_path = "/home/zach/Documents/dev/image-checker-suite-demo/data/artifact_for_model"
train_data = datasets.ImageFolder(f"{data_path}/train", transform=transform)
val_data   = datasets.ImageFolder(f"{data_path}/val", transform=transform)

# --- Create Weighted Sampler for class balancing ---
targets = [label for _, label in train_data.samples]
class_sample_counts = np.bincount(targets)
weights = 1. / class_sample_counts
sample_weights = [weights[label] for label in targets]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- Initialize model ---
model = ArtifactNet().to(DEVICE)

# --- Weighted Binary Cross Entropy Loss ---
# pos_weight = num_negative / num_positive
pos_weight = torch.tensor([class_sample_counts[0] / class_sample_counts[1]]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # --- Validation ---
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds.extend((probs > 0.5).cpu().numpy().flatten())
            truths.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(truths, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {acc:.3f}")

# --- Save Model ---
torch.save(model.state_dict(), "artifact_net.pth")
print("✅ Model saved to artifact_net.pth")
