# 🧠 Quick Alzheimer's Detection System Demo
# ================================================================

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("🚀 Quick Alzheimer's Detection Demo")
print("=" * 60)

# Quick configuration
config = {
    'input_size': 224,
    'batch_size': 8,  # Smaller for faster demo
    'num_epochs': 5,  # Fewer epochs for demo
    'learning_rate': 0.001,
    'train_dir': './train',
    'test_dir': './test'
}

# Check if dataset exists
if not os.path.exists(config['train_dir']):
    print(f"❌ Dataset not found at {config['train_dir']}")
    exit(1)

print("✅ Dataset found!")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Simple transforms
transform = transforms.Compose([
    transforms.Resize((config['input_size'], config['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
try:
    train_dataset = datasets.ImageFolder(config['train_dir'], transform=transform)
    test_dataset = datasets.ImageFolder(config['test_dir'], transform=transform)
    
    print(f"✅ Loaded datasets:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    print(f"   - Classes: {train_dataset.classes}")
    
    num_classes = len(train_dataset.classes)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"✅ Data loaders created ({len(train_loader)} train batches, {len(test_loader)} test batches)")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Simple CNN model
class SimpleAlzheimerCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleAlzheimerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model
model = SimpleAlzheimerCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model initialized with {total_params:,} parameters")

# Training loop
print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")

train_losses = []
train_accuracies = []

for epoch in range(config['num_epochs']):
    print(f"\n📈 Epoch {epoch+1}/{config['num_epochs']}")
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if batch_idx % 20 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f"   ✅ Epoch {epoch+1} completed | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

print(f"\n🎉 Training completed!")

# Test evaluation
print(f"\n📊 Evaluating on test set...")

model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        test_total += targets.size(0)
        test_correct += (predicted == targets).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

test_accuracy = 100. * test_correct / test_total
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")

# Create simple visualization
plt.figure(figsize=(15, 5))

# Training curves
plt.subplot(1, 3, 1)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, 'g-', linewidth=2, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Confusion matrix
plt.subplot(1, 3, 3)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_targets, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_dataset.classes, 
            yticklabels=train_dataset.classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('quick_alzheimer_results.png', dpi=300, bbox_inches='tight')
print("✅ Results saved as 'quick_alzheimer_results.png'")

# Show class-wise performance
print(f"\n📋 Class-wise Performance:")
for i, class_name in enumerate(train_dataset.classes):
    class_mask = np.array(all_targets) == i
    if np.sum(class_mask) > 0:
        class_correct = np.sum(np.array(all_predictions)[class_mask] == np.array(all_targets)[class_mask])
        class_total = np.sum(class_mask)
        class_acc = 100. * class_correct / class_total
        print(f"   - {class_name}: {class_acc:.1f}% ({class_correct}/{class_total})")

plt.show()

print(f"\n🎯 Demo Summary:")
print(f"   - Dataset: {len(train_dataset)} training, {len(test_dataset)} test samples")
print(f"   - Classes: {num_classes} ({', '.join(train_dataset.classes)})")
print(f"   - Model: Simple CNN with {total_params:,} parameters")
print(f"   - Training: {config['num_epochs']} epochs")
print(f"   - Final Test Accuracy: {test_accuracy:.2f}%")

print(f"\n✅ Quick demo completed successfully!")
print("=" * 60)
