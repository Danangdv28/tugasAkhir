"""
=============================================================================
STEP 3: LSTM TRAINING FOR CCS PREDICTION (PyTorch Version - FIXED)
=============================================================================
Tujuan: Prediksi Channel Condition State (CCS) berdasarkan 20 menit histori
Input: X (N, 20, num_features) - 20 timesteps histori kanal & lingkungan
Output: y (N,) - CCS ∈ {0: GOOD, 1: DEGRADED, 2: SEVERE}

Author: Channel Prediction System
Date: January 2026
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

print(f"✓ Data loaded successfully")
print(f"  X shape: {X.shape} (samples, timesteps, features)")
print(f"  y shape: {y.shape} (samples,)")
print(f"  Number of features: {X.shape[2]}")
print(f"  Sequence length: {X.shape[1]} timesteps")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n✓ Class distribution:")
class_names = ["GOOD", "DEGRADED", "SEVERE"]
for cls, count in zip(unique, counts):
    print(f"  Class {int(cls)} ({class_names[int(cls)]}): {count} samples ({count/len(y)*100:.1f}%)")

# =============================================================================
# 2. TIME-SERIES DATA SPLIT (NO RANDOM SHUFFLE!)
# =============================================================================
print("\n" + "="*80)
print("STEP 2: TIME-SERIES DATA SPLIT")
print("="*80)

N = len(X)
train_end = int(N * 0.7)  # 70% training
val_end = int(N * 0.85)   # 15% validation
# Remaining 15% for testing

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"✓ Data split completed (chronological order maintained)")
print(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/N*100:.1f}%)")
print(f"  Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/N*100:.1f}%)")
print(f"  Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/N*100:.1f}%)")

# Check class distribution in each set
for name, y_subset in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    unique_sub, counts_sub = np.unique(y_subset, return_counts=True)
    print(f"\n  {name} distribution:")
    for cls, count in zip(unique_sub, counts_sub):
        print(f"    {class_names[int(cls)]}: {count} ({count/len(y_subset)*100:.1f}%)")

# =============================================================================
# 3. COMPUTE CLASS WEIGHTS (HANDLE IMBALANCE)
# =============================================================================
print("\n" + "="*80)
print("STEP 3: COMPUTING CLASS WEIGHTS")
print("="*80)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"✓ Class weights computed (handles imbalanced data):")
for cls, weight in enumerate(class_weights):
    print(f"  {class_names[cls]}: {weight:.3f}")

# =============================================================================
# 3.5. SETUP TRAINING PARAMETERS (MOVED HERE - FIXED!)
# =============================================================================
print("\n" + "="*80)
print("STEP 3.5: SETTING UP TRAINING PARAMETERS")
print("="*80)

# Create models directory
os.makedirs("models", exist_ok=True)
model_path = "models/best_ccs_lstm_model.pth"

# Training parameters - DEFINED HERE FIRST!
epochs = 100
batch_size = 32
patience = 10
lr_patience = 5
lr_factor = 0.5
min_lr = 1e-6

NUM_FEATURES = X.shape[2]
NUM_CLASSES = 3
TIMESTEPS = X.shape[1]

print("✓ Training parameters configured:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Early stopping patience: {patience}")
print(f"  LR reduce patience: {lr_patience}")
print(f"  Model save path: {model_path}")
print(f"  Input features: {NUM_FEATURES}")
print(f"  Output classes: {NUM_CLASSES}")

# =============================================================================
# 4. BUILD LSTM MODEL (PRODUCTION-GRADE)
# =============================================================================
print("\n" + "="*80)
print("STEP 4: BUILDING LSTM MODEL")
print("="*80)

class CCS_LSTM_Predictor(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CCS_LSTM_Predictor, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(num_features, 64, batch_first=True)
        self.ln1 = nn.LayerNorm(64)  # LayerNorm instead of BatchNorm
        self.dropout1 = nn.Dropout(0.3)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.ln2 = nn.LayerNorm(32)  # LayerNorm instead of BatchNorm
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layers
        self.dense = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # First LSTM
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        # Second LSTM
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last timestep output
        x = self.ln2(x)
        x = self.dropout2(x)
        
        # Dense layers
        x = torch.relu(self.dense(x))
        x = self.dropout3(x)
        x = self.output(x)
        
        return x

# Create model
model = CCS_LSTM_Predictor(NUM_FEATURES, NUM_CLASSES)

# Loss and optimizer with class weights
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✓ Model architecture:")
print(model)
print(f"\n✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print(f"\n✓ Model compiled successfully")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: CrossEntropyLoss with class weights")

# =============================================================================
# 5. PREPARE DATA LOADERS
# =============================================================================
print("\n" + "="*80)
print("STEP 5: PREPARING DATA LOADERS")
print("="*80)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Create data loaders (NO SHUFFLE for time-series!)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ Data loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Batch size: {batch_size}")
print(f"  Shuffle: False (time-series preserved)")

# =============================================================================
# 6. TRAIN MODEL
# =============================================================================
print("\n" + "="*80)
print("STEP 6: TRAINING MODEL")
print("="*80)
print("Training started... This may take several minutes.")
print("-"*80)

# Training tracking
best_val_loss = float('inf')
epochs_no_improve = 0
current_lr = 0.001
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(epochs):
    # ==================== TRAINING ====================
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_dataset)
    train_acc = train_correct / train_total
    
    # ==================== VALIDATION ====================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataset)
    val_acc = val_correct / val_total
    
    # Save history
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch+1:3d}/{epochs} - "
          f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # ==================== MODEL CHECKPOINT ====================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        epochs_no_improve += 1
    
    # ==================== LEARNING RATE REDUCTION ====================
    if epochs_no_improve > 0 and epochs_no_improve % lr_patience == 0:
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = max(old_lr * lr_factor, min_lr)
        if new_lr < old_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            current_lr = new_lr
            print(f"  ⚠ Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    # ==================== EARLY STOPPING ====================
    if epochs_no_improve >= patience:
        print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
        print(f"  No improvement for {patience} epochs")
        break

print("-"*80)
print("✓ Training completed!")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Total epochs trained: {len(history['loss'])}")

# =============================================================================
# 7. PLOT TRAINING HISTORY
# =============================================================================
print("\n" + "="*80)
print("STEP 7: VISUALIZING TRAINING HISTORY")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history['loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot accuracy
axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training history plots saved to: training_history.png")

# =============================================================================
# 8. EVALUATE ON TEST SET
# =============================================================================
print("\n" + "="*80)
print("STEP 8: EVALUATING MODEL ON TEST SET")
print("="*80)

# Load best model
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"✓ Best model loaded from: {model_path}")

# Make predictions
print("\nGenerating predictions...")
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred_proba = torch.softmax(outputs, dim=1).numpy()
    y_pred = torch.argmax(outputs, dim=1).numpy()

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Set Accuracy: {test_accuracy*100:.2f}%")

# Detailed classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT (DETAILED)")
print("="*80)
print(classification_report(
    y_test, y_pred,
    target_names=class_names,
    digits=4
))

# =============================================================================
# 9. CONFUSION MATRIX
# =============================================================================
print("="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_test, y_pred)
print("\nRaw confusion matrix:")
print(cm)

# Calculate percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Absolute counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

# Percentages
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            cbar_kws={'label': 'Percentage (%)'})
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Confusion matrix saved to: confusion_matrix.png")

# =============================================================================
# 10. PER-CLASS ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("="*80)

for i, class_name in enumerate(class_names):
    true_positives = cm[i, i]
    false_positives = cm[:, i].sum() - cm[i, i]
    false_negatives = cm[i, :].sum() - cm[i, i]
    true_negatives = cm.sum() - (true_positives + false_positives + false_negatives)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")

# =============================================================================
# 11. PREDICTION CONFIDENCE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*80)

# Get max confidence for each prediction
confidence_scores = y_pred_proba.max(axis=1)

print(f"\nConfidence Statistics:")
print(f"  Mean confidence: {confidence_scores.mean()*100:.2f}%")
print(f"  Median confidence: {np.median(confidence_scores)*100:.2f}%")
print(f"  Min confidence: {confidence_scores.min()*100:.2f}%")
print(f"  Max confidence: {confidence_scores.max()*100:.2f}%")

# Confidence by class
fig, ax = plt.subplots(figsize=(12, 6))
for i, class_name in enumerate(class_names):
    class_mask = y_pred == i
    if class_mask.sum() > 0:
        class_confidences = confidence_scores[class_mask]
        ax.hist(class_confidences, bins=30, alpha=0.6, label=class_name, edgecolor='black')

ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Prediction Confidence Distribution by Class', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Confidence distribution saved to: confidence_distribution.png")

# =============================================================================
# 12. SAVE RESULTS SUMMARY
# =============================================================================
print("\n" + "="*80)
print("STEP 12: SAVING RESULTS SUMMARY")
print("="*80)

results_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'framework': 'PyTorch',
    'model_architecture': 'LSTM (64→32 units)',
    'total_samples': int(len(X)),
    'train_samples': int(len(X_train)),
    'val_samples': int(len(X_val)),
    'test_samples': int(len(X_test)),
    'test_accuracy': float(test_accuracy),
    'mean_confidence': float(confidence_scores.mean()),
    'epochs_trained': len(history['loss']),
    'final_train_loss': float(history['loss'][-1]),
    'final_val_loss': float(history['val_loss'][-1]),
    'best_val_loss': float(best_val_loss),
    'final_train_acc': float(history['accuracy'][-1]),
    'final_val_acc': float(history['val_accuracy'][-1]),
}

# Save to file
with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Results summary saved to: results_summary.json")
print("\nSummary:")
for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# =============================================================================
# 13. FINAL INTERPRETATION GUIDE
# =============================================================================
print("\n" + "="*80)
print("INTERPRETASI HASIL (PANDUAN UNTUK DOSEN)")
print("="*80)

print("""
✓ HASIL YANG BAIK & NORMAL:
  1. Test Accuracy > 80% → Model belajar dengan baik
  2. Precision tinggi untuk DEGRADED & SEVERE → Model bisa deteksi gangguan
  3. Recall tinggi untuk SEVERE → Tidak melewatkan kondisi kritis
  4. Sedikit confusion antara GOOD↔DEGRADED → WAJAR karena transisi gradual
  
✓ KENAPA MODEL INI VALID:
  1. Time-series split (NO random shuffle)
  2. Class weights untuk handle imbalance
  3. Early stopping untuk prevent overfitting
  4. Validation set untuk monitor generalization
  5. LayerNorm untuk stabilitas training sequence model
  
✓ NEXT STEPS (pilih salah satu):
  4 → Tambah SNR range prediction (multi-output LSTM)
  5 → Feature importance & ablation study
  6 → Real-time inference design
""")

print("\n" + "="*80)
print("TRAINING SELESAI! ✓ (PyTorch Version - ALL BUGS FIXED)")
print("="*80)
print(f"\nFiles generated:")
print(f"  1. {model_path}")
print(f"  2. training_history.png")
print(f"  3. confusion_matrix.png")
print(f"  4. confidence_distribution.png")
print(f"  5. results_summary.json")
print("\n" + "="*80)