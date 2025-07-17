# 🧠 Complete Alzheimer's Detection System using HRNet
# ================================================================
# This script contains the complete end-to-end training and evaluation system

import os
import sys
import time
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add lib directory to path to use existing HRNet implementation
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
sys.path.insert(0, lib_path)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Import existing HRNet model
from models.hrnet import get_face_alignment_net

# Scikit-learn for metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

print("🚀 Starting Alzheimer's Detection System...")
print("=" * 80)

# ================================
# STEP 1: Configuration
# ================================
print("⚙️ Setting up configuration...")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("✅ Random seeds set for reproducibility")

# Configuration parameters
config = {
    'input_size': 224,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_classes': 4,  # NonDemented, VeryMildDemented, MildDemented, ModerateDemented
    'train_dir': './train',
    'test_dir': './test',
    'class_names': ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
}

print("✅ Configuration set:")
for key, value in config.items():
    print(f"   - {key}: {value}")

# ================================
# STEP 2: Data Transforms and Dataset
# ================================
print("\n🔄 Setting up data transforms and loading dataset...")

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.Resize((config['input_size'], config['input_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((config['input_size'], config['input_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ Data transforms configured")

# Load dataset
if os.path.exists(config['train_dir']) and os.path.exists(config['test_dir']):
    train_dataset = datasets.ImageFolder(config['train_dir'], transform=train_transforms)
    val_dataset = datasets.ImageFolder(config['test_dir'], transform=val_transforms)
    
    print("✅ Real dataset loaded successfully")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Classes found: {train_dataset.classes}")
    
    # Update config with actual class names
    config['class_names'] = train_dataset.classes
    config['num_classes'] = len(train_dataset.classes)
    
    # Class distribution
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_distribution = Counter(train_labels)
    print("   - Training distribution:")
    for class_idx, count in train_distribution.items():
        class_name = train_dataset.classes[class_idx]
        print(f"     * {class_name}: {count} samples")
        
else:
    print(f"❌ Dataset directories not found!")
    print(f"   - Looking for: {config['train_dir']} and {config['test_dir']}")
    exit(1)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=0,  # Set to 0 for Windows compatibility
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("✅ Data loaders created")
print(f"   - Training batches: {len(train_loader)}")
print(f"   - Validation batches: {len(val_loader)}")

# ================================
# STEP 3: Modified HRNet Model for Alzheimer's Detection
# ================================
print("\n🏗️ Setting up HRNet model for Alzheimer's detection...")

class AlzheimerHRNet(nn.Module):
    """Modified HRNet for Alzheimer's Detection"""
    
    def __init__(self, num_classes=4):
        super(AlzheimerHRNet, self).__init__()
        
        # Use existing HRNet backbone (modify for our use case)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Feature extraction layers
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, inplanes, planes, stride=1):
        """Create a simple conv layer"""
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output

# Initialize model
model = AlzheimerHRNet(num_classes=config['num_classes'])
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("✅ AlzheimerHRNet model initialized")
print(f"   - Total parameters: {total_params:,}")
print(f"   - Trainable parameters: {trainable_params:,}")

# ================================
# STEP 4: Loss, Optimizer, and Scheduler
# ================================
print("\n⚖️ Setting up loss function, optimizer, and scheduler...")

# Calculate class weights for imbalanced data
class_counts = Counter(train_labels)
class_weights = []
total_samples = len(train_dataset)

for i in range(config['num_classes']):
    weight = total_samples / (config['num_classes'] * class_counts.get(i, 1))
    class_weights.append(weight)

class_weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"✅ Weighted CrossEntropyLoss initialized")
print(f"   - Class weights: {[f'{w:.3f}' for w in class_weights.cpu().numpy()]}")

# Optimizer
optimizer = optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay']
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

print("✅ Optimizer and scheduler configured")

# ================================
# STEP 5: Training Functions
# ================================
print("\n🏋️ Setting up training functions...")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store for detailed metrics
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predicted, all_targets, all_probabilities

print("✅ Training functions ready")

# ================================
# STEP 6: Training Loop
# ================================
print(f"\n🚀 Starting training for {config['num_epochs']} epochs...")

# Initialize tracking variables
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0
best_model_state = None

start_time = time.time()

for epoch in range(config['num_epochs']):
    print(f"\n{'='*60}")
    print(f"🔄 EPOCH {epoch+1}/{config['num_epochs']}")
    print(f"{'='*60}")
    
    # Training
    print(f"📈 Training...")
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Validation
    print(f"\n📊 Validating...")
    val_loss, val_acc, val_predicted, val_targets, val_probabilities = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"   🎉 New best model! Validation accuracy: {val_acc:.2f}%")
    
    # Epoch summary
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"   - Learning Rate: {current_lr:.6f}")
    print(f"   - Best Val Acc: {best_val_acc:.2f}%")
    
    # Early stopping check (optional)
    if epoch > 10 and val_acc < best_val_acc - 15:
        print(f"⚠️ Early stopping triggered - validation accuracy hasn't improved significantly")
        break

total_time = time.time() - start_time
print(f"\n🎉 Training completed!")
print(f"   - Total time: {total_time/60:.2f} minutes")
print(f"   - Best validation accuracy: {best_val_acc:.2f}%")

# Load best model for evaluation
model.load_state_dict(best_model_state)
print("✅ Best model loaded for final evaluation")

# ================================
# STEP 7: Final Evaluation and Visualizations
# ================================
print(f"\n📊 Conducting final evaluation and creating visualizations...")

# Final evaluation on test set
print("🧪 Final model evaluation...")
final_loss, final_acc, final_predictions, final_targets, final_probabilities = validate_epoch(
    model, val_loader, criterion, device
)

print(f"✅ Final Test Accuracy: {final_acc:.2f}%")

# Convert to numpy arrays
final_predictions = np.array(final_predictions)
final_targets = np.array(final_targets)
final_probabilities = np.array(final_probabilities)

# Detailed classification report
class_report = classification_report(final_targets, final_predictions, target_names=config['class_names'])
print(f"\n📋 Classification Report:")
print(class_report)

# Precision, Recall, F1-score
precision, recall, f1, support = precision_recall_fscore_support(final_targets, final_predictions, average='weighted')
print(f"\n📈 Weighted Metrics:")
print(f"   - Precision: {precision:.4f}")
print(f"   - Recall: {recall:.4f}")
print(f"   - F1-Score: {f1:.4f}")

# ================================
# STEP 8: Create Comprehensive Visualizations
# ================================
print(f"\n🎨 Creating comprehensive visualizations...")

# Set up matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 16))

# 1. Training History
ax1 = plt.subplot(3, 3, 1)
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Accuracy History
ax2 = plt.subplot(3, 3, 2)
plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Confusion Matrix
ax3 = plt.subplot(3, 3, 3)
cm = confusion_matrix(final_targets, final_predictions)
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im)

classes = config['class_names']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{int(cm[i, j])}',
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 4. Class Distribution
ax4 = plt.subplot(3, 3, 4)
class_counts_test = Counter(final_targets)
counts = [class_counts_test.get(i, 0) for i in range(len(classes))]
colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
bars = plt.bar(classes, counts, color=colors)
plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f'{count}', ha='center', va='bottom')

# 5. Per-Class Accuracy
ax5 = plt.subplot(3, 3, 5)
class_accuracies = []
for i in range(len(classes)):
    class_mask = (final_targets == i)
    if np.sum(class_mask) > 0:
        class_acc = np.sum((final_predictions[class_mask] == final_targets[class_mask])) / np.sum(class_mask)
        class_accuracies.append(class_acc * 100)
    else:
        class_accuracies.append(0)

bars = plt.bar(classes, class_accuracies, color=colors)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.ylim(0, 100)

for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom')

# 6. ROC Curves (One vs Rest)
ax6 = plt.subplot(3, 3, 6)
for i in range(len(classes)):
    # Binary targets for class i vs rest
    binary_targets = (final_targets == i).astype(int)
    class_probabilities = final_probabilities[:, i]
    
    fpr, tpr, _ = roc_curve(binary_targets, class_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, linewidth=2, 
            label=f'{classes[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One vs Rest)')
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, alpha=0.3)

# 7. Model Architecture Summary
ax7 = plt.subplot(3, 3, 7)
ax7.axis('off')
arch_text = f"""
🏗️ Model Architecture

📊 AlzheimerHRNet:
• Input: {config['input_size']}×{config['input_size']} RGB images
• Backbone: Modified HRNet
• Features: 512 dimensional
• Classes: {config['num_classes']}
• Parameters: {total_params:,}

🔧 Training Setup:
• Optimizer: Adam (lr={config['learning_rate']})
• Loss: Weighted CrossEntropyLoss
• Batch Size: {config['batch_size']}
• Epochs: {len(train_losses)}
• Device: {device}
"""

ax7.text(0.05, 0.95, arch_text, transform=ax7.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# 8. Performance Summary
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')
perf_text = f"""
📈 Performance Results

🎯 Overall Metrics:
• Test Accuracy: {final_acc:.2f}%
• Best Val Accuracy: {best_val_acc:.2f}%
• Precision: {precision:.4f}
• Recall: {recall:.4f}
• F1-Score: {f1:.4f}

⏱️ Training Details:
• Training Time: {total_time/60:.1f} minutes
• Final Train Loss: {train_losses[-1]:.4f}
• Final Val Loss: {val_losses[-1]:.4f}
• Final Learning Rate: {current_lr:.2e}

🧠 Clinical Impact:
Automated assistance for:
• Early detection
• Disease progression monitoring
• Objective assessment
"""

ax8.text(0.05, 0.95, perf_text, transform=ax8.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 9. Dataset Information
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
dataset_text = f"""
📊 Dataset Information

🗂️ Training Set:
• Total Samples: {len(train_dataset)}
• Classes: {len(classes)}

🔍 Class Distribution:
"""

for i, class_name in enumerate(classes):
    count = class_counts.get(i, 0)
    percentage = (count / len(train_dataset)) * 100
    dataset_text += f"• {class_name}: {count} ({percentage:.1f}%)\n"

dataset_text += f"""
🧪 Test Set:
• Total Samples: {len(val_dataset)}
• Final Accuracy: {final_acc:.2f}%

⚠️ Medical Disclaimer:
This model is for research purposes only.
Always consult medical professionals.
"""

ax9.text(0.05, 0.95, dataset_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('alzheimer_detection_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✅ Comprehensive results saved as 'alzheimer_detection_comprehensive_results.png'")

# Save additional detailed plots
plt.figure(figsize=(12, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Detailed confusion matrix saved as 'detailed_confusion_matrix.png'")

plt.show()

# ================================
# STEP 9: Save Model and Results
# ================================
print(f"\n💾 Saving model and results...")

# Save the trained model
model_save_path = 'alzheimer_hrnet_model.pth'
torch.save({
    'model_state_dict': best_model_state,
    'config': config,
    'class_names': config['class_names'],
    'final_accuracy': final_acc,
    'best_val_accuracy': best_val_acc,
    'train_history': {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    },
    'class_weights': class_weights.cpu().numpy(),
    'total_training_time': total_time
}, model_save_path)

print(f"✅ Model saved as '{model_save_path}'")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})

results_df.to_csv('training_results.csv', index=False)
print("✅ Training results saved as 'training_results.csv'")

# Save classification report
with open('classification_report.txt', 'w') as f:
    f.write("Alzheimer's Detection Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Final Test Accuracy: {final_acc:.2f}%\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(class_report)
    f.write(f"\n\nWeighted Metrics:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

print("✅ Classification report saved as 'classification_report.txt'")

# ================================
# Final Summary
# ================================
print("\n" + "="*80)
print("🎉 ALZHEIMER'S DETECTION SYSTEM COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"🎯 Final Results Summary:")
print(f"   ✅ Test Accuracy: {final_acc:.2f}%")
print(f"   ✅ Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"   ✅ Model Parameters: {total_params:,}")
print(f"   ✅ Training Time: {total_time/60:.1f} minutes")
print(f"   ✅ Classes: {', '.join(config['class_names'])}")

print(f"\n📁 Generated Files:")
print(f"   📊 alzheimer_detection_comprehensive_results.png")
print(f"   🎯 detailed_confusion_matrix.png")
print(f"   💾 alzheimer_hrnet_model.pth")
print(f"   📈 training_results.csv")
print(f"   📋 classification_report.txt")

print(f"\n🏥 Clinical Applications:")
print(f"   • Early detection of cognitive decline")
print(f"   • Quantitative assessment of dementia severity")
print(f"   • Disease progression monitoring")
print(f"   • Supporting diagnostic decisions")
print(f"   • Research and clinical trials")

print(f"\n⚠️ Important Medical Disclaimer:")
print(f"   This model is designed for research and educational purposes only.")
print(f"   It should not be used as a substitute for professional medical diagnosis.")
print(f"   Always consult qualified healthcare professionals for medical decisions.")

print(f"\n🚀 Next Steps:")
print(f"   1. Review the comprehensive visualizations")
print(f"   2. Analyze per-class performance metrics")
print(f"   3. Consider model improvements (data augmentation, ensemble methods)")
print(f"   4. Validate on external datasets")
print(f"   5. Consult with medical professionals for clinical validation")

print("="*80)
print("🧠 Alzheimer's Detection System - Training Complete! 🧠")
print("="*80)
