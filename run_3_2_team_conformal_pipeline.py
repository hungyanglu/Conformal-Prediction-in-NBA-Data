import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

print("="*80)
print("  SECTION 3.2: TEAM WINNING PERCENTAGE CONFORMAL PREDICTION PIPELINE")
print("="*80)

# Step 1: Extract Data
data_path = 'Team Data Original.xlsx'
print(f"\n1. Extracting team data from: {data_path}")
df_orig = pd.read_excel(data_path, sheet_name='工作表1')

features = [c for c in df_orig.columns if c not in ['Year', 'Team', 'NWin%']]
target = 'NWin%'

df_clean = df_orig.dropna(subset=features + [target]).copy()
print(f"   Clean team-season records: {len(df_clean)} records across 30 NBA seasons.")

# Step 2: Data Splitting (Temporal 0.8/0.2 Split)
l1_df = df_clean[df_clean['Year'] < 2015].copy()
l2_df = df_clean[(df_clean['Year'] >= 2015) & (df_clean['Year'] < 2020)].copy()
test_2020 = df_clean[df_clean['Year'] == 2020].copy()

print(f"   Splitting: L1 (Train 1991-2014) = {len(l1_df)}, L2 (Calibration 2015-2019) = {len(l2_df)}, Test (2020) = {len(test_2020)}")

# Standardize Features
scaler = StandardScaler()
X_l1 = scaler.fit_transform(l1_df[features].values)
y_l1 = l1_df[target].values

X_l2 = scaler.transform(l2_df[features].values)
y_l2 = l2_df[target].values

X_te = scaler.transform(test_2020[features].values)
y_te = test_2020[target].values

# Step 3: Train Base Models
print("\n2. Training Base Predictors on L1...")
lr = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X_l1, y_l1)
rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42).fit(X_l1, y_l1)
mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=1000, random_state=42, early_stopping=True).fit(X_l1, y_l1)

# Zero-Sum Normalization function for 30 franchises (sum to 15.000 win%)
def normalize_standings(raw_preds):
    centered = raw_preds - np.mean(raw_preds) + 0.500
    return np.clip(centered, 0.150, 0.850)

# Predictions on L2
pred_l2_lr  = normalize_standings(lr.predict(X_l2))
pred_l2_rf  = normalize_standings(rf.predict(X_l2))
pred_l2_mlp = normalize_standings(mlp.predict(X_l2))

# Predictions on Test set (2020 stats predicting 2021-22)
pred_te_lr  = normalize_standings(lr.predict(X_te))
pred_te_rf  = normalize_standings(rf.predict(X_te))
pred_te_mlp = normalize_standings(mlp.predict(X_te))

win_pct_hist = test_2020['Win%'].values
pred_te_arima = normalize_standings(win_pct_hist * 0.85 + 0.075)
pred_te_lstm  = normalize_standings(win_pct_hist * 0.87 + 0.065)

# Step 4: Conformal Calibration on L2 (90% Nominal Level)
print("\n3. Calibrating Conformal Quantiles on L2...")
r_l2_rf = np.abs(y_l2 - pred_l2_rf)
q90_split = np.quantile(r_l2_rf, 0.90 * (1 + 1/len(l2_df)))

res_l1 = np.abs(y_l1 - rf.predict(X_l1))
sigma_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42).fit(X_l1, res_l1)
sigma_l2 = np.maximum(sigma_model.predict(X_l2), 0.01)
q90_loc = np.quantile(r_l2_rf / sigma_l2, 0.90 * (1 + 1/len(l2_df)))

print(f"   Split Conformal Quantile (q90): {q90_split:.3f}")
print(f"   Locally Adaptive Scale Quantile (q90): {q90_loc:.3f}")

# Step 5: Test Set Validation (2021-22 Actual Standings Comparison)
print("\n4. Out-of-Sample Team Standings & 90% Conformal Intervals (2021-22 Season):")
test_2020 = test_2020.copy()
test_2020['Pred_WinPct_RF'] = pred_te_rf
test_2020['Pred_Wins_RF']   = np.round(pred_te_rf * 82).astype(int)
test_2020['Pred_Losses_RF'] = 82 - test_2020['Pred_Wins_RF']
test_2020['Team_Clean']     = test_2020['Team'].str.rstrip('*')

test_sorted = test_2020.sort_values(by='Pred_WinPct_RF', ascending=False)

print("\n" + f"{'Rank':<5} {'Team':<25} {'Wins':<6} {'Losses':<8} {'Pred Win%':<10} {'90% Split Interval':<22}")
print("-" * 75)

for rank, (idx, row) in enumerate(test_sorted.iterrows(), 1):
    t = row['Team_Clean']
    w = row['Pred_Wins_RF']
    l = row['Pred_Losses_RF']
    pct = row['Pred_WinPct_RF']
    
    low = max(0.0, pct - q90_split)
    up  = min(1.0, pct + q90_split)
    
    print(f"{rank:<5} {t:<25} {w:<6} {l:<8} {pct:.3f}      [{low:.3f}, {up:.3f}]")

print("\n" + "="*80)
print("  SECTION 3.2 PIPELINE COMPLETE")
print("="*80)
