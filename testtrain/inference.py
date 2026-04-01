import numpy as np
import torch
import joblib

# =========================
# LOAD ARTIFACT
# =========================
scaler = joblib.load("scaler_140GHz.save")

checkpoint = torch.load("lstm_model_140GHz.pth")

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, 64, num_layers=2, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(64, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()

model = LSTMModel(checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

WINDOW_SIZE = checkpoint["window_size"]

# =========================
# FEATURE ORDER (KRITIS)
# =========================
FEATURE_ORDER = [
    "temperature_C",
    "humidity_percent",
    "rain_rate_mmhr",
    "wind_speed_ms",
    "pressure_hPa",
    "hour_of_day",
    "day_of_week",
    "month"
]

# =========================
# MAIN FUNCTION
# =========================
def predict_snr(input_sequence):
    """
    input_sequence: list of dict (panjang = window_size)
    """

    # 1. convert ke array
    X = []
    for row in input_sequence:
        X.append([row[f] for f in FEATURE_ORDER])

    X = np.array(X)

    # 2. scaling
    X_scaled = scaler.transform(X)

    # 3. reshape ke LSTM
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    # 4. predict
    with torch.no_grad():
        pred = model(X_tensor).item()

    return float(pred)