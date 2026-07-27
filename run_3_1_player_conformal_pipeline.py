import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

print("="*80)
print("  SECTION 3.1: PLAYER PER CONFORMAL PREDICTION PIPELINE (End-to-End)")
print("="*80)

# Step 1: Load and Structure Data
data_path = 'Player Data.xlsx'
print(f"\n1. Extracting data from: {data_path}")
p_df = pd.read_excel(data_path, sheet_name=0)

features_t = ['Age.1', 'G.1', 'MP.1', 'PER▼.1', 'TS%.1', '3PAr.1', 'FTr.1', 
              'ORB%.1', 'DRB%.1', 'TRB%.1', 'AST%.1', 'STL%.1', 'BLK%.1', 
              'TOV%.1', 'USG%.1', 'OWS.1', 'DWS.1', 'WS.1', 'WS/48.1', 
              'OBPM.1', 'DBPM.1', 'BPM.1', 'VORP.1']
target_next = 'PER▼'

df_clean = p_df.dropna(subset=features_t + [target_next]).copy()
print(f"   Clean player-season pairs: {len(df_clean)} records across 21 seasons.")

# Step 2: Data Splitting (Temporal 0.65/0.35 Baseline)
# Training L1 (pre-2013), Calibration L2 (2013-2019), Test (2020 predicting 2021)
l1_df = df_clean[df_clean['Year.1'] < 2013].copy()
l2_df = df_clean[(df_clean['Year.1'] >= 2013) & (df_clean['Year.1'] < 2020)].copy()
test_df = df_clean[df_clean['Year.1'] == 2020].copy()

print(f"   Splitting: L1 (Train) = {len(l1_df)}, L2 (Calibration) = {len(l2_df)}, Test = {len(test_df)}")

# Standardize Features
scaler = StandardScaler()
X_l1 = scaler.fit_transform(l1_df[features_t].values)
y_l1 = l1_df[target_next].values

X_l2 = scaler.transform(l2_df[features_t].values)
y_l2 = l2_df[target_next].values

X_te = scaler.transform(test_df[features_t].values)
y_te = test_df[target_next].values

# Step 3: Train Base Models on L1
print("\n2. Training Base Models on L1...")
lr = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X_l1, y_l1)
rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42).fit(X_l1, y_l1)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.5, max_iter=1000, random_state=42, early_stopping=True).fit(X_l1, y_l1)

# Fit CQR Quantile Regressors (90% target)
cqr_lo = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=80, random_state=42).fit(X_l1, y_l1)
cqr_hi = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=80, random_state=42).fit(X_l1, y_l1)

# Fit Dispersion Model sigma(x) for Locally Adaptive Conformal
res_l1 = np.abs(y_l1 - rf.predict(X_l1))
sigma_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42).fit(X_l1, res_l1)

# Predictions on Calibration L2
pred_l2_rf = rf.predict(X_l2)

# Step 4: Compute Conformal Nonconformity Quantiles on L2
print("\n3. Calibrating Conformal Nonconformity Quantiles on L2 (90% Nominal Level)...")

# A. Split Conformal
r_l2_split = np.abs(y_l2 - pred_l2_rf)
q90_split = np.quantile(r_l2_split, 0.90 * (1 + 1/len(l2_df)))

# B. Locally Adaptive Conformal
sigma_l2 = np.maximum(sigma_model.predict(X_l2), 0.1)
r_l2_loc = r_l2_split / sigma_l2
q90_loc = np.quantile(r_l2_loc, 0.90 * (1 + 1/len(l2_df)))

# C. CQR
e_l2_cqr = np.maximum(cqr_lo.predict(X_l2) - y_l2, y_l2 - cqr_hi.predict(X_l2))
q90_cqr = np.quantile(e_l2_cqr, 0.90 * (1 + 1/len(l2_df)))

print(f"   Split Conformal Quantile (q90): {q90_split:.3f}")
print(f"   Locally Adaptive Scale Quantile (q90): {q90_loc:.3f}")
print(f"   CQR Adjustment Quantile (q90): {q90_cqr:.3f}")

# Step 5: Test Set Validation (2020-21 Known Data)
print("\n4. Evaluating Validation Results on Test Set (Known 2020-21 Data)...")
pred_te_lr = lr.predict(X_te)
pred_te_rf = rf.predict(X_te)
pred_te_mlp = mlp.predict(X_te)
sigma_te = np.maximum(sigma_model.predict(X_te), 0.1)

# Split Conformal Coverage & Length
cov_split = np.mean((pred_te_rf - q90_split <= y_te) & (y_te <= pred_te_rf + q90_split))
len_split = 2 * q90_split

# Locally Adaptive Coverage & Length
cov_loc = np.mean((pred_te_rf - sigma_te*q90_loc <= y_te) & (y_te <= pred_te_rf + sigma_te*q90_loc))
len_loc = np.mean(2 * sigma_te * q90_loc)

# CQR Coverage & Length
cqr_te_lo = cqr_lo.predict(X_te) - q90_cqr
cqr_te_hi = cqr_hi.predict(X_te) + q90_cqr
cov_cqr = np.mean((cqr_te_lo <= y_te) & (y_te <= cqr_te_hi))
len_cqr = np.mean(cqr_te_hi - cqr_te_lo)

print(f"   [Split Conformal]   Coverage: {cov_split*100:.1f}%, Avg Length: {len_split:.3f}")
print(f"   [Locally Adaptive]  Coverage: {cov_loc*100:.1f}%, Avg Length: {len_loc:.3f}")
print(f"   [CQR]               Coverage: {cov_cqr*100:.1f}%, Avg Length: {len_cqr:.3f}")

# Step 6: Out-of-Sample Predictions for Top Star Players (2021-22)
print("\n5. Out-of-Sample Conformal Predictions for Star Players (2021-22 Season):")
star_names = ['Nikola Jokić', 'Giannis Antetokounmpo', 'Joel Embiid', 'Luka Dončić', 'Stephen Curry']
star_df = test_df[test_df['Player.1'].isin(star_names)].copy()

for idx, row in star_df.iterrows():
    p = row['Player.1']
    x_p = scaler.transform([row[features_t].values])
    y_hat_rf = rf.predict(x_p)[0]
    sig_p = max(sigma_model.predict(x_p)[0], 0.1)
    
    # Split Interval
    l_sp, u_sp = y_hat_rf - q90_split, y_hat_rf + q90_split
    # Locally Adaptive Interval
    l_loc, u_loc = y_hat_rf - sig_p * q90_loc, y_hat_rf + sig_p * q90_loc
    
    print(f"\n   Player: {p:22s} | Point Pred (RF PER): {y_hat_rf:.2f}")
    print(f"     -> Split Conformal (90%):    [{l_sp:.2f}, {u_sp:.2f}] (Length: {u_sp-l_sp:.2f})")
    print(f"     -> Locally Adaptive (90%):   [{l_loc:.2f}, {u_loc:.2f}] (Length: {u_loc-l_loc:.2f})")

print("\n" + "="*80)
print("  SECTION 3.1 PIPELINE COMPLETE")
print("="*80)
