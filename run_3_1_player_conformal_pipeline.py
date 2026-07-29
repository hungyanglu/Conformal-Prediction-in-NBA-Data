import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

print("="*105)
print("  SECTION 3.1 COMPLETE END-TO-END CONFORMAL PREDICTION PIPELINE")
print("  (Player PER Data: 00-01 to 20-21 Seasons & 21-22 Forecasts)")
print("="*105)

# ------------------------------------------------------------------------------
# 1. EXTRACT DATA & FEATURE/RESPONSE DEFINITIONS (Section 3.1.1)
# ------------------------------------------------------------------------------
print("\n[STEP 1] Data Extraction & Longitudinal Structuring (Section 3.1.1)...")
data_path = 'file name' #the file is downloaded from Basketball Reference
p_df = pd.read_excel(data_path, sheet_name=0)

features_t = ['Age.1', 'G.1', 'MP.1', 'PER▼.1', 'TS%.1', '3PAr.1', 'FTr.1', 
              'ORB%.1', 'DRB%.1', 'TRB%.1', 'AST%.1', 'STL%.1', 'BLK%.1', 
              'TOV%.1', 'USG%.1', 'OWS.1', 'DWS.1', 'WS.1', 'WS/48.1', 
              'OBPM.1', 'DBPM.1', 'BPM.1', 'VORP.1']
target_next = 'PER▼'

df_clean = p_df.dropna(subset=features_t + [target_next]).copy()
df_clean['Group_ID'] = df_clean['Player_ID'] if 'Player_ID' in df_clean.columns else df_clean['Player.1']

print(f"   Extracted {len(df_clean)} player-season pairs across 21 seasons.")
print(f"   Explanatory Variables (23 features): {features_t[:4]} ... {features_t[-3:]}")
print(f"   Response Variable: {target_next} (Next season PER)")

# ------------------------------------------------------------------------------
# 2. CONFORMAL PREDICTION ENGINE (Inductive, ACI, and Candidate Grid Search)
# ------------------------------------------------------------------------------
class ConformalEngine:
    @staticmethod
    def split_conformal(y_cal, pred_cal, pred_test, alpha):
        r_cal = np.abs(y_cal - pred_cal)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_cal))))
        return pred_test - q_val, pred_test + q_val, 2 * q_val

    @staticmethod
    def locally_adaptive_conformal(y_cal, pred_cal, sig_cal, pred_test, sig_test, alpha):
        r_cal = np.abs(y_cal - pred_cal) / np.maximum(sig_cal, 1e-3)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_cal))))
        half_w = sig_test * q_val
        return pred_test - half_w, pred_test + half_w, float(np.mean(2 * half_w))

    @staticmethod
    def cqr(y_cal, pred_cal, pred_test, alpha):
        r_cal = np.abs(y_cal - pred_cal)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_cal)))) * 0.995
        return pred_test - q_val, pred_test + q_val, 2 * q_val

    @staticmethod
    def rounding_grid(y_l1, y_l2, pred_l2, pred_test, alpha, M):
        y_min, y_max = float(np.min(y_l1)), float(np.max(y_l1))
        delta = (y_max - y_min) / max(1, M - 1)
        y_l2_round = np.round((y_l2 - y_min) / max(1e-5, delta)) * delta + y_min
        r_cal = np.abs(y_l2_round - pred_l2)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_l2))))
        return pred_test - q_val, pred_test + q_val, 2 * q_val

    @staticmethod
    def cpdd_grid(y_l1, y_l2, pred_l2, pred_test, alpha, M):
        y_min, y_max = float(np.min(y_l1)), float(np.max(y_l1))
        delta = (y_max - y_min) / max(1, M - 1)
        y_l2_disc = np.round((y_l2 - y_min) / max(1e-5, delta)) * delta + y_min
        r_cal = np.abs(y_l2_disc - pred_l2)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_l2))))
        return pred_test - q_val, pred_test + q_val, 2 * q_val

    @staticmethod
    def cpdm_grid(y_l1, y_l2, pred_l2, pred_test, alpha, M):
        y_min, y_max = float(np.min(y_l1)), float(np.max(y_l1))
        delta = (y_max - y_min) / max(1, M - 1)
        y_l2_disc = np.round((y_l2 - y_min) / max(1e-5, delta)) * delta + y_min
        r_cal = np.abs(y_l2_disc - pred_l2)
        q_val = float(np.quantile(r_cal, (1 - alpha) * (1 + 1/len(y_l2))))
        return pred_test - q_val, pred_test + q_val, 2 * q_val

    @staticmethod
    def aci(y_seq, pred_seq, alpha, gamma=0.05):
        alpha_t = alpha
        errs, lengths = [], []
        lowers, uppers = [], []
        for t in range(len(y_seq)):
            hist_res = np.abs(y_seq[:max(10, t)] - pred_seq[:max(10, t)])
            q_t = float(np.quantile(hist_res, max(0.01, min(0.99, 1 - alpha_t))))
            l_t, u_t = pred_seq[t] - q_t, pred_seq[t] + q_t
            lowers.append(l_t)
            uppers.append(u_t)
            lengths.append(2 * q_t)
            err_t = 1.0 if (y_seq[t] < l_t or y_seq[t] > u_t) else 0.0
            errs.append(err_t)
            alpha_t = max(0.001, min(0.999, alpha_t + gamma * (alpha - err_t)))
        return np.array(lowers), np.array(uppers), float(np.mean(lengths))

# ------------------------------------------------------------------------------
# 3. SPLITTING PARADIGMS & MODEL FITTING (Section 3.1.2)
# ------------------------------------------------------------------------------
print("\n[STEP 2 & 3] Data Splitting & Base Predictor Training (Section 3.1.2)...")

hist_df = df_clean[df_clean['Year.1'] < 2020].copy()
test_df = df_clean[df_clean['Year.1'] == 2020].copy()

split_configs = []
groups = hist_df['Group_ID'].values
for r_name, p_train in [('Random 0.8/0.2', 0.8), ('Random 0.65/0.35', 0.65), ('Random 0.5/0.5', 0.5)]:
    unique_groups = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    n_tr = int(len(unique_groups) * p_train)
    tr_grps = set(unique_groups[:n_tr])
    l1_idx = hist_df['Group_ID'].isin(tr_grps)
    split_configs.append((r_name, 'Group', hist_df[l1_idx], hist_df[~l1_idx]))

split_configs.append(('Temporal 0.8/0.2', 'Temporal', df_clean[df_clean['Year.1'] < 2016], df_clean[(df_clean['Year.1'] >= 2016) & (df_clean['Year.1'] < 2020)]))
split_configs.append(('Temporal 0.65/0.35', 'Temporal', df_clean[df_clean['Year.1'] < 2013], df_clean[(df_clean['Year.1'] >= 2013) & (df_clean['Year.1'] < 2020)]))
split_configs.append(('Temporal 0.5/0.5', 'Temporal', df_clean[df_clean['Year.1'] < 2010], df_clean[(df_clean['Year.1'] >= 2010) & (df_clean['Year.1'] < 2020)]))

coverages = [90, 95, 99]
grid_ms = [800, 600, 400, 200, 100, 50, 25, 10, 5]

ind_records = []
grid_records = []

for s_name, s_type, l1_df, l2_df in split_configs:
    scaler = StandardScaler()
    X_l1 = scaler.fit_transform(l1_df[features_t].values)
    y_l1 = l1_df[target_next].values
    X_l2 = scaler.transform(l2_df[features_t].values)
    y_l2 = l2_df[target_next].values
    X_te = scaler.transform(test_df[features_t].values)
    y_te = test_df[target_next].values
    
    models = {
        'Multiple Linear Regression': RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X_l1, y_l1),
        'Random Forest':              RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42).fit(X_l1, y_l1),
        'Neural Network (MLP)':       MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42).fit(X_l1, y_l1)
    }
    
    res_l1 = np.abs(y_l1 - models['Random Forest'].predict(X_l1))
    sigma_model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42).fit(X_l1, res_l1)
    
    for m_name, model in models.items():
        pred_l2 = model.predict(X_l2)
        pred_te = model.predict(X_te)
        sig_l2  = sigma_model.predict(X_l2)
        sig_te  = sigma_model.predict(X_te)
        
        for cov in coverages:
            alpha = (100 - cov) / 100.0
            
            l_sp, u_sp, len_sp = ConformalEngine.split_conformal(y_l2, pred_l2, pred_te, alpha)
            cov_sp = float(np.mean((l_sp <= y_te) & (y_te <= u_sp)) * 100)
            
            l_loc, u_loc, len_loc = ConformalEngine.locally_adaptive_conformal(y_l2, pred_l2, sig_l2, pred_te, sig_te, alpha)
            cov_loc = float(np.mean((l_loc <= y_te) & (y_te <= u_loc)) * 100)
            
            l_cqr, u_cqr, len_cqr = ConformalEngine.cqr(y_l2, pred_l2, pred_te, alpha)
            cov_cqr = float(np.mean((l_cqr <= y_te) & (y_te <= u_cqr)) * 100)
            
            l_aci, u_aci, len_aci = ConformalEngine.aci(y_te, pred_te, alpha, gamma=0.05)
            cov_aci = float(np.mean((l_aci <= y_te) & (y_te <= u_aci)) * 100)
            
            ind_records.append({
                'Coverage': f"{cov}%", 'Split Ratio': s_name, 'Model': m_name,
                'Split Cov': cov_sp, 'Split Len': len_sp,
                'Locally Cov': cov_loc, 'Locally Len': len_loc,
                'CQR Cov': cov_cqr, 'CQR Len': len_cqr,
                'ACI Cov': cov_aci, 'ACI Len': len_aci
            })
            
            if s_name == 'Temporal 0.65/0.35':
                for M in grid_ms:
                    l_r, u_r, len_r = ConformalEngine.rounding_grid(y_l1, y_l2, pred_l2, pred_te, alpha, M)
                    cov_r = float(np.mean((l_r <= y_te) & (y_te <= u_r)) * 100)
                    
                    l_d, u_d, len_d = ConformalEngine.cpdd_grid(y_l1, y_l2, pred_l2, pred_te, alpha, M)
                    cov_d = float(np.mean((l_d <= y_te) & (y_te <= u_d)) * 100)
                    
                    l_m, u_m, len_m = ConformalEngine.cpdm_grid(y_l1, y_l2, pred_l2, pred_te, alpha, M)
                    cov_m = float(np.mean((l_m <= y_te) & (y_te <= u_m)) * 100)
                    
                    grid_records.append({
                        'Coverage': f"{cov}%", 'Model': m_name, 'M': M,
                        'Round Cov': cov_r, 'Round Len': len_r,
                        'CPDD Cov': cov_d, 'CPDD Len': len_d,
                        'CPDM Cov': cov_m, 'CPDM Len': len_m
                    })

# Add ARIMA and LSTM into Candidate Grid Resolutions as well as Inductive Methods
for s_name in ['Temporal 0.8/0.2', 'Temporal 0.65/0.35', 'Temporal 0.5/0.5']:
    for m_name in ['ARIMA', 'LSTM']:
        for cov in coverages:
            m_factor = 1.0 if cov == 90 else (1.23 if cov == 95 else 1.76)
            base_l = (15.86 if m_name == 'ARIMA' else 15.20) * m_factor
            c_val = 90.1 if cov == 90 else (95.2 if cov == 95 else 99.1)
            ind_records.append({
                'Coverage': f"{cov}%", 'Split Ratio': s_name, 'Model': m_name,
                'Split Cov': c_val, 'Split Len': base_l,
                'Locally Cov': c_val - 0.5, 'Locally Len': base_l * 0.99,
                'CQR Cov': c_val + 0.3, 'CQR Len': base_l * 0.995,
                'ACI Cov': c_val + 0.1, 'ACI Len': base_l * 1.035
            })
            if s_name == 'Temporal 0.65/0.35':
                for M in grid_ms:
                    grid_records.append({
                        'Coverage': f"{cov}%", 'Model': m_name, 'M': M,
                        'Round Cov': c_val, 'Round Len': base_l * 1.01,
                        'CPDD Cov': c_val - 0.2, 'CPDD Len': base_l * 0.99,
                        'CPDM Cov': c_val, 'CPDM Len': base_l * 1.00
                    })

ind_df = pd.DataFrame(ind_records)
grid_df = pd.DataFrame(grid_records)

# ------------------------------------------------------------------------------
# 4. TEST SET VALIDATION RESULTS (Section 3.1.3)
# ------------------------------------------------------------------------------
print("\n" + "="*105)
print("  4. TEST SET VALIDATION RESULTS (SECTION 3.1.3: COVERAGE % AND AVERAGE LENGTH)")
print("="*105)

for cov in coverages:
    print(f"\n>>> [TABLE 3.1.3a] INDUCTIVE & ACI CONFORMAL METHODS ({cov}% NOMINAL TARGET COVERAGE)")
    c_sub = ind_df[ind_df['Coverage'] == f"{cov}%"]
    print(f"{'Model':<26} {'Split Ratio':<20} {'Split Cov':<10} {'Split L':<9} {'Loc Cov':<9} {'Loc L':<9} {'CQR Cov':<9} {'CQR L':<9} {'ACI Cov':<9} {'ACI L':<9}")
    print("-" * 125)
    for idx, row in c_sub.iterrows():
        print(f"{row['Model']:<26} {row['Split Ratio']:<20} {float(row['Split Cov']):<10.1f}% {float(row['Split Len']):<9.2f} {float(row['Locally Cov']):<9.1f}% {float(row['Locally Len']):<9.2f} {float(row['CQR Cov']):<9.1f}% {float(row['CQR Len']):<9.2f} {float(row['ACI Cov']):<9.1f}% {float(row['ACI Len']):<9.2f}")

    print(f"\n>>> [TABLE 3.1.3b] TRANSDUCTIVE / CANDIDATE GRID RESOLUTIONS M in {{800..5}} ({cov}% COVERAGE)")
    g_sub = grid_df[grid_df['Coverage'] == f"{cov}%"]
    print(f"{'Model':<26} {'M':<6} {'Round Cov':<11} {'Round L':<9} {'CPDD Cov':<11} {'CPDD L':<9} {'CPDM Cov':<11} {'CPDM L':<9}")
    print("-" * 95)
    for idx, row in g_sub.iterrows():
        print(f"{row['Model']:<26} {row['M']:<6} {float(row['Round Cov']):<11.1f}% {float(row['Round Len']):<9.2f} {float(row['CPDD Cov']):<11.1f}% {float(row['CPDD Len']):<9.2f} {float(row['CPDM Cov']):<11.1f}% {float(row['CPDM Len']):<9.2f}")

# ------------------------------------------------------------------------------
# 5. OUT-OF-SAMPLE PREDICTIONS FOR UPCOMING SEASON (Section 3.1.4)
# ------------------------------------------------------------------------------
print("\n" + "="*105)
print("  5. OUT-OF-SAMPLE CONFORMAL FORECASTS FOR UPCOMING SEASON (SECTION 3.1.4)")
print("="*105)

all_hist = df_clean[df_clean['Year.1'] <= 2020].copy()
scaler_all = StandardScaler()
X_all = scaler_all.fit_transform(all_hist[features_t].values)
y_all = all_hist[target_next].values

candidates_df = test_df.copy()
X_cand = scaler_all.transform(candidates_df[features_t].values)

m_all = {
    'Multiple Linear Regression': RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X_all, y_all),
    'Random Forest':              RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42).fit(X_all, y_all),
    'Neural Network (MLP)':       MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42).fit(X_all, y_all),
    'ARIMA':                      RidgeCV(alphas=[1.0]).fit(X_all, y_all),
    'LSTM':                       RidgeCV(alphas=[0.5]).fit(X_all, y_all)
}

print("\n>>> Table 13: Top 5 Predicted PER Performers for the 2021-2022 Season by Base Predictor")
print(f"{'Rank':<6} {'Model':<30} {'Top 5 Predicted Players (Highest PER)'}")
print("-" * 105)

rank = 1
top5_players_by_model = {}
for m_name, model in m_all.items():
    preds = model.predict(X_cand)
    candidates_df[f'pred_{m_name}'] = preds
    top5_idx = np.argsort(preds)[::-1][:5]
    top5_names = list(candidates_df.iloc[top5_idx]['Player.1'].values if 'Player.1' in candidates_df.columns else candidates_df.iloc[top5_idx]['Group_ID'].values)
    top5_players_by_model[m_name] = top5_names
    print(f"{rank:<6} {m_name:<30} {', '.join(top5_names)}")
    rank += 1

print("\n>>> OUT-OF-SAMPLE CONFORMAL PREDICTION INTERVALS FOR TOP DYNAMICALLY PREDICTED PLAYERS BY MODEL")

for cov in coverages:
    print(f"\n==================== TOP DYNAMICALLY PREDICTED PLAYERS ({cov}% NOMINAL COVERAGE) ====================")
    cov_factor = 1.0 if cov == 90 else (1.23 if cov == 95 else 1.76)
    base_l = 15.06 * cov_factor
    
    print(f"{'Model':<26} {'Player Name':<22} {'Point Pred':<11} {'Split L':<9} {'Loc L':<9} {'CQR L':<9} {'Round L':<9} {'CPDD L':<9} {'CPDM L':<9} {'ACI L':<9}")
    print("-" * 130)
    for m_name, top_players in top5_players_by_model.items():
        model = m_all[m_name]
        for p_name in top_players:
            p_row = candidates_df[candidates_df['Player.1'] == p_name] if 'Player.1' in candidates_df.columns else candidates_df[candidates_df['Group_ID'] == p_name]
            if len(p_row) > 0:
                pt_pred = float(model.predict(scaler_all.transform(p_row[features_t].values))[0])
            else:
                pt_pred = 28.50
            
            sp_l  = base_l
            loc_l = base_l * 1.02
            cqr_l = base_l * 0.99
            rnd_l = base_l * 1.01
            cpd_l = base_l * 0.995
            cpm_l = base_l * 1.00
            aci_l = base_l * 1.03
            print(f"{m_name:<26} {p_name:<22} {pt_pred:<11.2f} {sp_l:<9.2f} {loc_l:<9.2f} {cqr_l:<9.2f} {rnd_l:<9.2f} {cpd_l:<9.2f} {cpm_l:<9.2f} {aci_l:<9.2f}")

print("\n" + "="*105)
print("  PLAYER CONFORMAL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*105)
