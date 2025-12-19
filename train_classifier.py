import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------
# 1. DATASET DEFINITION
# ---------------------------------------------------------
class CelebACustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.attr_names = self.annotations.columns[1:].tolist()

        # Robust check for separators if needed
        if len(self.attr_names) < 2:
            # Fallback for whitespace separated files
            self.annotations = pd.read_csv(csv_file)
            self.attr_names = self.annotations.columns[1:].tolist()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])

        try:
            image = Image.open(img_name).convert('RGB')
        except (OSError, FileNotFoundError):
            # Skip corrupted images if necessary, or just fail
            print(f"Error loading {img_name}")
            # return a blank image to prevent crash (hacky fix)
            image = Image.new('RGB', (224, 224))

        labels = self.annotations.iloc[idx, 1:].values

        # Map -1 to 0, and 1 to 1
        labels_new = (labels.astype('float32') + 1) // 2
        labels_new = torch.tensor(labels_new, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels_new


# ---------------------------------------------------------
# 2. SETUP DATA & SPLIT
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CelebACustomDataset(
    csv_file='./archive/list_attr_celeba.csv',
    root_dir='./archive/img_align_celeba/img_align_celeba',
    transform=transform
)

# --- SPLIT LOGIC ---
# Calculate sizes
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% Training
val_size = total_size - train_size  # 20% Validation

# Perform the random split
train_set, val_set = random_split(dataset, [train_size, val_size])

print(f"Total Images: {total_size}")
print(f"Training Set: {train_size}")
print(f"Validation Set: {val_size}")

# Create TWO DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)  # Shuffle False for Val

attribute_names = dataset.attr_names

# ---------------------------------------------------------
# 3. MODEL SETUP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze layers
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

    # Modify Head
num_ftrs = model.fc.in_features
# We use Sigmoid + BCELoss as per your preference
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 40),
    nn.Sigmoid()
)
model = model.to(device)

optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

criterion = nn.BCELoss()

# ---------------------------------------------------------
# 4. TRAIN & VALIDATION LOOP
# ---------------------------------------------------------
num_epochs = 3

for epoch in range(num_epochs):
    # ==========================
    #      TRAINING PHASE
    # ==========================
    model.train()  # Enable Dropout/BN
    train_loss = 0.0
    train_correct = torch.zeros(40).to(device)
    train_total = 0

    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Metrics
        preds = (outputs > 0.5).float()
        train_correct += (preds == labels).sum(dim=0)
        train_total += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"[Train] Batch {batch_idx}: Loss = {loss.item():.4f}")

    # ==========================
    #     VALIDATION PHASE
    # ==========================
    model.eval()  # Disable Dropout/BN
    val_loss = 0.0
    val_correct = torch.zeros(40).to(device)
    val_total = 0

    # No gradients needed for validation (saves memory/time)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum(dim=0)
            val_total += labels.size(0)

    # ==========================
    #       EPOCH SUMMARY
    # ==========================
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_acc_per_attr = train_correct / train_total
    val_acc_per_attr = val_correct / val_total

    print(f"\nResults Epoch {epoch + 1}:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(
        f"Train Mean Acc: {train_acc_per_attr.mean() * 100:.2f}% | Val Mean Acc: {val_acc_per_attr.mean() * 100:.2f}%")

    print("\n--- Validation Accuracy Breakdown (Top 8) ---")
    # Just printing the first 8 for brevity, change range to 40 for all
    for i in range(min(8, 40)):
        name = attribute_names[i]
        print(f"{name}: {val_acc_per_attr[i].item() * 100:.1f}%", end=" | ")
    print("\n")

# ---------------------------------------------------------
# 5. SAVE
# ---------------------------------------------------------
print("Saving model...")
torch.save(model.state_dict(), 'celeba_resnet18_split.pth')
print("Saved.")