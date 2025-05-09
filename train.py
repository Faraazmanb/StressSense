import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model

# Dataset paths
train_dir = r"C:\Users\mdfar\Downloads\streenew10epoc-20250314T170136Z-001\streenew10epoc\archive (5)\facesData\train"
test_dir = r"C:\Users\mdfar\Downloads\streenew10epoc-20250314T170136Z-001\streenew10epoc\archive (5)\facesData\test"

# Image preprocessing
img_size = 224
batch_size = 32
num_epochs = 10  # 🔥 Increased from 5 to 10 for better performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load ViT model
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model = model.to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        val_acc, val_loss = evaluate_model(model, test_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
# Evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total, total_loss / total

# Train & Evaluate
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

# Save model
torch.save(model.state_dict(), "stress_detection_vit.pth")
print("✅ Model saved successfully.")