import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

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

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

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
print("TRAINING XGBOOST MODELS")
print("="*60)

# Model RAW
print("\nTraining RAW model...")
model_raw = XGBRegressor(**XGBOOST_PARAMS)
model_raw.fit(Xr_train_scaled, yr_train)
print("✓ RAW model training completed")

# Model CCS
print("\nTraining CCS model...")
model_ccs = XGBRegressor(**XGBOOST_PARAMS)
model_ccs.fit(Xc_train_scaled, yc_train)
print("✓ CCS model training completed")

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
print("XGBOOST - RAW FEATURES")
print("="*60)
print_metrics("TRAIN SET", results['RAW']['train'])
print_metrics("VALIDATION SET", results['RAW']['val'])
print_metrics("TEST SET", results['RAW']['test'])

print("\n" + "="*60)
print("XGBOOST - CCS FEATURES")
print("="*60)
print_metrics("TRAIN SET", results['CCS']['train'])
print_metrics("VALIDATION SET", results['CCS']['val'])
print_metrics("TEST SET", results['CCS']['test'])

# =========================
# FEATURE IMPORTANCE
# =========================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

feature_imp_raw = pd.DataFrame({
    'Feature': RAW_FEATURES,
    'Importance': model_raw.feature_importances_
}).sort_values('Importance', ascending=False)

feature_imp_ccs = pd.DataFrame({
    'Feature': CCS_FEATURES,
    'Importance': model_ccs.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRAW Features:")
print(feature_imp_raw.to_string(index=False))

print("\nCCS Features:")
print(feature_imp_ccs.to_string(index=False))

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
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('XGBoost Model Evaluation & Analysis', fontsize=18, fontweight='bold', y=0.995)

# 1. Metrics Comparison - RMSE
ax1 = fig.add_subplot(gs[0, 0])
datasets = ['train', 'val', 'test']
raw_rmse = [results['RAW'][ds]['RMSE'] for ds in datasets]
ccs_rmse = [results['CCS'][ds]['RMSE'] for ds in datasets]

x = np.arange(len(datasets))
width = 0.35

ax1.bar(x - width/2, raw_rmse, width, label='RAW', alpha=0.8, color='#3498db')
ax1.bar(x + width/2, ccs_rmse, width, label='CCS', alpha=0.8, color='#e74c3c')
ax1.set_xlabel('Dataset', fontweight='bold')
ax1.set_ylabel('RMSE (dB)', fontweight='bold')
ax1.set_title('RMSE Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([d.upper() for d in datasets])
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Metrics Comparison - R²
ax2 = fig.add_subplot(gs[0, 1])
raw_r2 = [results['RAW'][ds]['R2'] for ds in datasets]
ccs_r2 = [results['CCS'][ds]['R2'] for ds in datasets]

ax2.bar(x - width/2, raw_r2, width, label='RAW', alpha=0.8, color='#3498db')
ax2.bar(x + width/2, ccs_r2, width, label='CCS', alpha=0.8, color='#e74c3c')
ax2.set_xlabel('Dataset', fontweight='bold')
ax2.set_ylabel('R² Score', fontweight='bold')
ax2.set_title('R² Score Comparison', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([d.upper() for d in datasets])
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Metrics Comparison - MSE
ax3 = fig.add_subplot(gs[0, 2])
raw_mse = [results['RAW'][ds]['MSE'] for ds in datasets]
ccs_mse = [results['CCS'][ds]['MSE'] for ds in datasets]

ax3.bar(x - width/2, raw_mse, width, label='RAW', alpha=0.8, color='#3498db')
ax3.bar(x + width/2, ccs_mse, width, label='CCS', alpha=0.8, color='#e74c3c')
ax3.set_xlabel('Dataset', fontweight='bold')
ax3.set_ylabel('MSE (dB²)', fontweight='bold')
ax3.set_title('MSE Comparison', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([d.upper() for d in datasets])
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Feature Importance - RAW
ax4 = fig.add_subplot(gs[1, 0])
feature_imp_raw_sorted = feature_imp_raw.sort_values('Importance', ascending=True)
ax4.barh(range(len(feature_imp_raw_sorted)), feature_imp_raw_sorted['Importance'], 
         color='#3498db', alpha=0.8)
ax4.set_yticks(range(len(feature_imp_raw_sorted)))
ax4.set_yticklabels(feature_imp_raw_sorted['Feature'])
ax4.set_xlabel('Importance', fontweight='bold')
ax4.set_title('Feature Importance - RAW Model', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Feature Importance - CCS
ax5 = fig.add_subplot(gs[1, 1])
feature_imp_ccs_sorted = feature_imp_ccs.sort_values('Importance', ascending=True)
ax5.barh(range(len(feature_imp_ccs_sorted)), feature_imp_ccs_sorted['Importance'], 
         color='#e74c3c', alpha=0.8)
ax5.set_yticks(range(len(feature_imp_ccs_sorted)))
ax5.set_yticklabels(feature_imp_ccs_sorted['Feature'])
ax5.set_xlabel('Importance', fontweight='bold')
ax5.set_title('Feature Importance - CCS Model', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# 6. Feature Importance Comparison (CCS only)
ax6 = fig.add_subplot(gs[1, 2])
ccs_feature = feature_imp_ccs[feature_imp_ccs['Feature'] == 'ccs']['Importance'].values[0]
other_features = feature_imp_ccs[feature_imp_ccs['Feature'] != 'ccs']['Importance'].sum()
labels = ['CCS Feature', 'Other Features']
sizes = [ccs_feature, other_features]
colors = ['#e74c3c', '#95a5a6']
explode = (0.1, 0)

ax6.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax6.set_title('CCS Feature vs Others\nImportance Distribution', fontweight='bold')

# 7. Prediction vs Actual - RAW (Test Set)
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(yr_test, yr_test_pred, alpha=0.5, s=30, color='#3498db', edgecolors='black', linewidth=0.5)
ax7.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax7.set_xlabel('Actual SNR (dB)', fontweight='bold')
ax7.set_ylabel('Predicted SNR (dB)', fontweight='bold')
ax7.set_title(f'RAW - Test Set Predictions\nR²={results["RAW"]["test"]["R2"]:.4f}, RMSE={results["RAW"]["test"]["RMSE"]:.4f} dB', 
              fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Prediction vs Actual - CCS (Test Set)
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(yc_test, yc_test_pred, alpha=0.5, s=30, color='#e74c3c', edgecolors='black', linewidth=0.5)
ax8.plot([yc_test.min(), yc_test.max()], [yc_test.min(), yc_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax8.set_xlabel('Actual SNR (dB)', fontweight='bold')
ax8.set_ylabel('Predicted SNR (dB)', fontweight='bold')
ax8.set_title(f'CCS - Test Set Predictions\nR²={results["CCS"]["test"]["R2"]:.4f}, RMSE={results["CCS"]["test"]["RMSE"]:.4f} dB', 
              fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Residuals Plot
ax9 = fig.add_subplot(gs[2, 2])
residuals_raw = yr_test - yr_test_pred
residuals_ccs = yc_test - yc_test_pred

ax9.scatter(yr_test_pred, residuals_raw, alpha=0.5, s=30, 
            color='#3498db', label='RAW', edgecolors='black', linewidth=0.5)
ax9.scatter(yc_test_pred, residuals_ccs, alpha=0.5, s=30, 
            color='#e74c3c', label='CCS', edgecolors='black', linewidth=0.5)
ax9.axhline(y=0, color='black', linestyle='--', lw=2)
ax9.set_xlabel('Predicted SNR (dB)', fontweight='bold')
ax9.set_ylabel('Residuals (dB)', fontweight='bold')
ax9.set_title('Residuals Plot (Test Set)', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.savefig('xgboost_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'xgboost_evaluation.png'")
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
mae_improvement = ((results['RAW']['test']['MAE'] - results['CCS']['test']['MAE']) / 
                   results['RAW']['test']['MAE']) * 100

print(f"\nMSE Reduction : {mse_improvement:+.2f}%")
print(f"RMSE Reduction: {rmse_improvement:+.2f}%")
print(f"MAE Reduction : {mae_improvement:+.2f}%")
print(f"R² Improvement: {r2_improvement:+.2f}%")

if rmse_improvement > 0:
    print(f"\n✓ CCS model performs BETTER (lower RMSE by {rmse_improvement:.2f}%)")
else:
    print(f"\n✗ RAW model performs BETTER (lower RMSE by {abs(rmse_improvement):.2f}%)")

# Check for overfitting
print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)

print("\nRAW Model:")
train_test_gap_raw = results['RAW']['train']['RMSE'] - results['RAW']['test']['RMSE']
print(f"Train RMSE: {results['RAW']['train']['RMSE']:.4f} dB")
print(f"Test RMSE : {results['RAW']['test']['RMSE']:.4f} dB")
print(f"Gap       : {train_test_gap_raw:.4f} dB")
if abs(train_test_gap_raw) > 0.5:
    print("⚠ Potential overfitting detected!")
else:
    print("✓ Model generalizes well")

print("\nCCS Model:")
train_test_gap_ccs = results['CCS']['train']['RMSE'] - results['CCS']['test']['RMSE']
print(f"Train RMSE: {results['CCS']['train']['RMSE']:.4f} dB")
print(f"Test RMSE : {results['CCS']['test']['RMSE']:.4f} dB")
print(f"Gap       : {train_test_gap_ccs:.4f} dB")
if abs(train_test_gap_ccs) > 0.5:
    print("⚠ Potential overfitting detected!")
else:
    print("✓ Model generalizes well")

# Save models (optional)
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)
model_raw.save_model('xgboost_model_raw.json')
model_ccs.save_model('xgboost_model_ccs.json')
print("✓ Models saved successfully!")
print("  - xgboost_model_raw.json")
print("  - xgboost_model_ccs.json")

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)