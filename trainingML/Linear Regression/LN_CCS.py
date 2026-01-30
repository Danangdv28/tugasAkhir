import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# =========================
# CONFIG
# =========================
RAW_FILE = "single_140GHz_14days_urban.csv"
CCS_FILE = "single_140GHz_14days_urban_CCS_FULL.csv"
TARGET = "snr_db"
SHIFT = 1

# Rasio pembagian data
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_FEATURES = [
    "snr_db",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m",
    "path_loss_db"
]

CCS_FEATURES = RAW_FEATURES + ["ccs"]

# =========================
# UTILS
# =========================
def prepare_data(df, features):
    """Prepare data dengan target shift dan split 70-15-15"""
    df = df.copy()
    df["target"] = df[TARGET].shift(-SHIFT)
    df = df.dropna().reset_index(drop=True)
    
    X = df[features].values
    y = df["target"].values
    
    # Split data: 70% train, 15% val, 15% test
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def print_metrics(name, metrics):
    """Print metrics in formatted way"""
    print(f"\n{name}")
    print(f"{'='*50}")
    print(f"MSE  : {metrics['MSE']:.4f} dB²")
    print(f"RMSE : {metrics['RMSE']:.4f} dB")
    print(f"MAE  : {metrics['MAE']:.4f} dB")
    print(f"R²   : {metrics['R2']:.4f}")

# =========================
# LOAD DATA
# =========================
print("Loading data...")
raw_df = pd.read_csv(RAW_FILE)
ccs_df = pd.read_csv(CCS_FILE)

# Handle missing values
raw_df["fog_visibility_m"] = raw_df["fog_visibility_m"].fillna(0)
ccs_df["fog_visibility_m"] = ccs_df["fog_visibility_m"].fillna(0)

print(f"RAW data shape: {raw_df.shape}")
print(f"CCS data shape: {ccs_df.shape}")

# =========================
# PREPARE DATA
# =========================
print("\n" + "="*60)
print("PREPARING DATA")
print("="*60)

Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = prepare_data(raw_df, RAW_FEATURES)
Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = prepare_data(ccs_df, CCS_FEATURES)

print(f"\nRAW Features: {len(RAW_FEATURES)}")
print(f"CCS Features: {len(CCS_FEATURES)}")
print(f"\nTrain set: {Xr_train.shape[0]} samples ({TRAIN_RATIO*100}%)")
print(f"Val set  : {Xr_val.shape[0]} samples ({VAL_RATIO*100}%)")
print(f"Test set : {Xr_test.shape[0]} samples ({TEST_RATIO*100}%)")

# Standardize data
scaler_raw = StandardScaler()
Xr_train_scaled = scaler_raw.fit_transform(Xr_train)
Xr_val_scaled = scaler_raw.transform(Xr_val)
Xr_test_scaled = scaler_raw.transform(Xr_test)

scaler_ccs = StandardScaler()
Xc_train_scaled = scaler_ccs.fit_transform(Xc_train)
Xc_val_scaled = scaler_ccs.transform(Xc_val)
Xc_test_scaled = scaler_ccs.transform(Xc_test)

# =========================
# TRAIN MODELS
# =========================
print("\n" + "="*60)
print("TRAINING LINEAR REGRESSION MODELS")
print("="*60)

# Model RAW
model_raw = LinearRegression()
model_raw.fit(Xr_train_scaled, yr_train)

# Model CCS
model_ccs = LinearRegression()
model_ccs.fit(Xc_train_scaled, yc_train)

# =========================
# EVALUATE MODELS
# =========================
results = {
    'RAW': {'train': {}, 'val': {}, 'test': {}},
    'CCS': {'train': {}, 'val': {}, 'test': {}}
}

# Predictions RAW
yr_train_pred = model_raw.predict(Xr_train_scaled)
yr_val_pred = model_raw.predict(Xr_val_scaled)
yr_test_pred = model_raw.predict(Xr_test_scaled)

results['RAW']['train'] = calculate_metrics(yr_train, yr_train_pred)
results['RAW']['val'] = calculate_metrics(yr_val, yr_val_pred)
results['RAW']['test'] = calculate_metrics(yr_test, yr_test_pred)

# Predictions CCS
yc_train_pred = model_ccs.predict(Xc_train_scaled)
yc_val_pred = model_ccs.predict(Xc_val_scaled)
yc_test_pred = model_ccs.predict(Xc_test_scaled)

results['CCS']['train'] = calculate_metrics(yc_train, yc_train_pred)
results['CCS']['val'] = calculate_metrics(yc_val, yc_val_pred)
results['CCS']['test'] = calculate_metrics(yc_test, yc_test_pred)

# Print results
print("\n" + "="*60)
print("LINEAR REGRESSION - RAW FEATURES")
print("="*60)
print_metrics("TRAIN SET", results['RAW']['train'])
print_metrics("VALIDATION SET", results['RAW']['val'])
print_metrics("TEST SET", results['RAW']['test'])

print("\n" + "="*60)
print("LINEAR REGRESSION - CCS FEATURES")
print("="*60)
print_metrics("TRAIN SET", results['CCS']['train'])
print_metrics("VALIDATION SET", results['CCS']['val'])
print_metrics("TEST SET", results['CCS']['test'])

# =========================
# CREATE COMPARISON TABLE
# =========================
print("\n" + "="*60)
print("METRICS COMPARISON TABLE")
print("="*60)

comparison_data = []
for model_name in ['RAW', 'CCS']:
    for dataset in ['train', 'val', 'test']:
        comparison_data.append({
            'Model': model_name,
            'Dataset': dataset.upper(),
            'MSE': results[model_name][dataset]['MSE'],
            'RMSE': results[model_name][dataset]['RMSE'],
            'MAE': results[model_name][dataset]['MAE'],
            'R²': results[model_name][dataset]['R2']
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# =========================
# VISUALIZATIONS
# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Linear Regression Model Evaluation', fontsize=16, fontweight='bold')

# 1. Metrics Comparison - RMSE
datasets = ['train', 'val', 'test']
raw_rmse = [results['RAW'][ds]['RMSE'] for ds in datasets]
ccs_rmse = [results['CCS'][ds]['RMSE'] for ds in datasets]

x = np.arange(len(datasets))
width = 0.35

axes[0, 0].bar(x - width/2, raw_rmse, width, label='RAW', alpha=0.8, color='#3498db')
axes[0, 0].bar(x + width/2, ccs_rmse, width, label='CCS', alpha=0.8, color='#e74c3c')
axes[0, 0].set_xlabel('Dataset')
axes[0, 0].set_ylabel('RMSE (dB)')
axes[0, 0].set_title('RMSE Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([d.upper() for d in datasets])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Metrics Comparison - R²
raw_r2 = [results['RAW'][ds]['R2'] for ds in datasets]
ccs_r2 = [results['CCS'][ds]['R2'] for ds in datasets]

axes[0, 1].bar(x - width/2, raw_r2, width, label='RAW', alpha=0.8, color='#3498db')
axes[0, 1].bar(x + width/2, ccs_r2, width, label='CCS', alpha=0.8, color='#e74c3c')
axes[0, 1].set_xlabel('Dataset')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('R² Score Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([d.upper() for d in datasets])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Metrics Comparison - MSE
raw_mse = [results['RAW'][ds]['MSE'] for ds in datasets]
ccs_mse = [results['CCS'][ds]['MSE'] for ds in datasets]

axes[0, 2].bar(x - width/2, raw_mse, width, label='RAW', alpha=0.8, color='#3498db')
axes[0, 2].bar(x + width/2, ccs_mse, width, label='CCS', alpha=0.8, color='#e74c3c')
axes[0, 2].set_xlabel('Dataset')
axes[0, 2].set_ylabel('MSE (dB²)')
axes[0, 2].set_title('MSE Comparison')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels([d.upper() for d in datasets])
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Prediction vs Actual - RAW (Test Set)
axes[1, 0].scatter(yr_test, yr_test_pred, alpha=0.5, s=20, color='#3498db')
axes[1, 0].plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual SNR (dB)')
axes[1, 0].set_ylabel('Predicted SNR (dB)')
axes[1, 0].set_title(f'RAW - Test Set\nR²={results["RAW"]["test"]["R2"]:.4f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Prediction vs Actual - CCS (Test Set)
axes[1, 1].scatter(yc_test, yc_test_pred, alpha=0.5, s=20, color='#e74c3c')
axes[1, 1].plot([yc_test.min(), yc_test.max()], [yc_test.min(), yc_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual SNR (dB)')
axes[1, 1].set_ylabel('Predicted SNR (dB)')
axes[1, 1].set_title(f'CCS - Test Set\nR²={results["CCS"]["test"]["R2"]:.4f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Residuals Plot
residuals_raw = yr_test - yr_test_pred
residuals_ccs = yc_test - yc_test_pred

axes[1, 2].scatter(yr_test_pred, residuals_raw, alpha=0.5, s=20, 
                   color='#3498db', label='RAW')
axes[1, 2].scatter(yc_test_pred, residuals_ccs, alpha=0.5, s=20, 
                   color='#e74c3c', label='CCS')
axes[1, 2].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1, 2].set_xlabel('Predicted SNR (dB)')
axes[1, 2].set_ylabel('Residuals (dB)')
axes[1, 2].set_title('Residuals Plot (Test Set)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_evaluation.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'linear_regression_evaluation.png'")
plt.show()

# =========================
# IMPROVEMENT ANALYSIS
# =========================
print("\n" + "="*60)
print("IMPROVEMENT ANALYSIS: CCS vs RAW (TEST SET)")
print("="*60)

rmse_improvement = ((results['RAW']['test']['RMSE'] - results['CCS']['test']['RMSE']) / 
                    results['RAW']['test']['RMSE']) * 100
r2_improvement = ((results['CCS']['test']['R2'] - results['RAW']['test']['R2']) / 
                  abs(results['RAW']['test']['R2'])) * 100
mse_improvement = ((results['RAW']['test']['MSE'] - results['CCS']['test']['MSE']) / 
                   results['RAW']['test']['MSE']) * 100

print(f"\nRMSE Reduction: {rmse_improvement:+.2f}%")
print(f"MSE Reduction : {mse_improvement:+.2f}%")
print(f"R² Improvement: {r2_improvement:+.2f}%")

if rmse_improvement > 0:
    print(f"\n✓ CCS model performs BETTER (lower RMSE by {rmse_improvement:.2f}%)")
else:
    print(f"\n✗ RAW model performs BETTER (lower RMSE by {abs(rmse_improvement):.2f}%)")