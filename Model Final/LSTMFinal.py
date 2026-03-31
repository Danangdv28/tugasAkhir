# ==========================================================
# 1. IMPORT LIBRARY
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset_scaled.csv")

target = "snr_dB"

X = df.drop(columns=[target]).values
y = df[target].values


# ==========================================================
# 3. SEQUENCE ENGINEERING
# ==========================================================

def create_sequences(X, y, window):

    X_seq = []
    y_seq = []

    for i in range(len(X) - window):

        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])

    return np.array(X_seq), np.array(y_seq)


window_size = 20

X_seq, y_seq = create_sequences(X, y, window_size)


# ==========================================================
# 4. TRAIN / VAL / TEST SPLIT
# ==========================================================

train_split = int(len(X_seq) * 0.7)
val_split = int(len(X_seq) * 0.85)

X_train = X_seq[:train_split]
y_train = y_seq[:train_split]

X_val = X_seq[train_split:val_split]
y_val = y_seq[train_split:val_split]

X_test = X_seq[val_split:]
y_test = y_seq[val_split:]


# ==========================================================
# 5. DATASET CLASS
# ==========================================================

class ChannelDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


train_dataset = ChannelDataset(X_train, y_train)
val_dataset = ChannelDataset(X_val, y_val)
test_dataset = ChannelDataset(X_test, y_test)


# ==========================================================
# 6. DATALOADER
# ==========================================================

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==========================================================
# 7. LSTM MODEL
# ==========================================================

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.relu(self.fc1(out))

        out = self.fc2(out)

        return out.squeeze()


input_size = X_train.shape[2]

model = LSTMModel(input_size)


# ==========================================================
# 8. LOSS & OPTIMIZER
# ==========================================================

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ==========================================================
# 9. EARLY STOPPING SETUP
# ==========================================================

patience = 10
best_val_loss = np.inf
counter = 0


# ==========================================================
# 10. TRAINING LOOP
# ==========================================================

epochs = 100

train_losses = []
val_losses = []

for epoch in range(epochs):

    # TRAIN
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            output = model(X_batch)

            loss = criterion(output, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # EARLY STOPPING

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "best_lstm_model.pth")

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping triggered")
            break


# ==========================================================
# 11. LOAD BEST MODEL
# ==========================================================

model.load_state_dict(torch.load("best_lstm_model.pth"))


# ==========================================================
# 12. TEST PREDICTION
# ==========================================================

model.eval()

predictions = []

with torch.no_grad():

    for X_batch, _ in test_loader:

        output = model(X_batch)

        predictions.extend(output.numpy())

predictions = np.array(predictions)


# ==========================================================
# OVERFITTING CHECK (TRAIN PREDICTION)
# ==========================================================

train_predictions = []

with torch.no_grad():

    for X_batch, _ in train_loader:

        output = model(X_batch)

        train_predictions.extend(output.numpy())

train_predictions = np.array(train_predictions)


# ==========================================================
# 13. EVALUATION
# ==========================================================

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

train_mse = mean_squared_error(y_train[:len(train_predictions)], train_predictions)
train_r2 = r2_score(y_train[:len(train_predictions)], train_predictions)

print("\nMODEL PERFORMANCE")
print("Train MSE :", train_mse)
print("Train R2  :", train_r2)

print("\nTEST PERFORMANCE")
print("Test MSE :", mse)
print("Test R2  :", r2)


# ==========================================================
# OVERFITTING ANALYSIS
# ==========================================================

print("\nOVERFITTING CHECK")

if train_r2 - r2 > 0.2:
    print("WARNING: Model kemungkinan OVERFITTING")
else:
    print("Model generalization masih OK")


# ==========================================================
# 14. SAVE PREDICTION PLOT
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(y_test[:500], label="True SNR")
plt.plot(predictions[:500], label="Predicted SNR")

plt.legend()
plt.title("LSTM Channel Prediction")

plt.savefig("prediction_plot.png", dpi=300)

plt.close()


# ==========================================================
# TRAIN vs TEST COMPARISON PLOT
# ==========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(y_train[:500], label="True")
plt.plot(train_predictions[:500], label="Pred")
plt.title("Train Prediction")
plt.legend()

plt.subplot(1,2,2)
plt.plot(y_test[:500], label="True")
plt.plot(predictions[:500], label="Pred")
plt.title("Test Prediction")
plt.legend()

plt.savefig("overfitting_check.png", dpi=300)

plt.close()


# ==========================================================
# 15. SAVE TRAINING CURVE
# ==========================================================

plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.legend()
plt.title("Training Curve")

plt.savefig("training_curve.png", dpi=300)

plt.close()


torch.save(model.state_dict(), "lstm_channel_model.pth")