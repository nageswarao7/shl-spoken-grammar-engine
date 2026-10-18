import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load Data
train_labels = pd.read_csv(r"e:\Assessment\shl_kaggle\data\Dataset_Final\train.csv")
train_feat = pd.read_csv(r"e:\Assessment\shl_kaggle\data\features\train_acoustic_features.csv")
test_feat = pd.read_csv(r"e:\Assessment\shl_kaggle\data\features\test_acoustic_features.csv")
test_df = pd.read_csv(r"e:\Assessment\shl_kaggle\data\Dataset_Final\test.csv")

# Merge features with labels
df_train = pd.merge(train_labels, train_feat, on="filename")
df_test = test_feat.copy()

# 2. Feature Engineering Function
def add_engineered_features(df):
    d = df.copy()
    # Dynamic Voice Variety
    d["mfcc_std_sum"] = sum(d[f"mfcc_{i}_std"] for i in range(2, 10))
    d["mfcc_std_ratio"] = d["mfcc_std_sum"] / (d["mfcc_1_std"] + 1e-5)
    
    # Silence and Pause Fluency
    d["speech_time"] = d["duration"] * (1.0 - d["silence_ratio"])
    d["pause_rate"] = d["num_pauses"] / (d["duration"] / 60.0 + 1e-5)
    d["mean_pause_sec"] = (d["duration"] * d["silence_ratio"]) / (d["num_pauses"] + 1.0)
    
    # Energy Dynamics
    d["active_energy"] = d["rms_mean"] / (1.0 - d["silence_ratio"] + 1e-5)
    d["energy_dynamics"] = (d["rms_max"] - d["rms_mean"]) / (d["rms_std"] + 1e-5)
    
    # Spectral Spread and Clarity
    d["spectral_spread"] = (d["spec_roll_mean"] - d["spec_cent_mean"]) / (d["spec_bw_mean"] + 1e-5)
    d["voicing_clarity"] = d["spec_cent_mean"] / (d["zcr_mean"] * d["spec_bw_mean"] + 1e-5)
    
    # Noise/Static detector flag (spec_cent_mean > 3500)
    d["is_static_noise"] = (d["spec_cent_mean"] > 3500).astype(int)
    return d

train_eng = add_engineered_features(df_train)
test_eng = add_engineered_features(df_test)

# Separate static noise from spoken speech
is_speech_train = (train_eng["is_static_noise"] == 0)
speech_train = train_eng[is_speech_train].reset_index(drop=True)

drop_cols = ["filename", "label", "error", "is_silent_audio", "is_static_noise"]
features = [c for c in speech_train.columns if c not in drop_cols]
print(f"Total features used: {len(features)}")

X_speech = speech_train[features].values
y_speech = speech_train["label"].values

X_test = test_eng[features].values

# 3. 10-Fold Cross-Validation on Spoken Speech
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(speech_train))
oof_xgb = np.zeros(len(speech_train))
oof_cat = np.zeros(len(speech_train))
oof_ridge = np.zeros(len(speech_train))

test_preds_lgb = np.zeros(len(test_eng))
test_preds_xgb = np.zeros(len(test_eng))
test_preds_cat = np.zeros(len(test_eng))
test_preds_ridge = np.zeros(len(test_eng))

print(f"\n--- Training {n_splits}-Fold Multimodel Stacking on Spoken Speech ---")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_speech, y_speech)):
    X_tr, y_tr = X_speech[trn_idx], y_speech[trn_idx]
    X_va, y_va = X_speech[val_idx], y_speech[val_idx]
    
    # Model 1: LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=350,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / n_splits
    
    # Model 2: XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.025,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        random_state=42 + fold,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / n_splits
    
    # Model 3: CatBoost
    model_cat = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test) / n_splits
    
    # Model 4: Ridge
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)
    
    model_ridge = Ridge(alpha=15.0, random_state=42 + fold)
    model_ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = model_ridge.predict(X_va_sc)
    test_preds_ridge += model_ridge.predict(X_te_sc) / n_splits

# 4. Out-of-Fold Evaluation
print("\n--- Out-of-Fold (OOF) Metrics (Spoken Speech Subset) ---")
for name, preds in [("LightGBM", oof_lgb), ("XGBoost", oof_xgb), ("CatBoost", oof_cat), ("Ridge", oof_ridge)]:
    rmse = root_mean_squared_error(y_speech, preds)
    r = pearsonr(y_speech, preds)[0]
    print(f"[{name:8s}] RMSE: {rmse:.4f} | Pearson r: {r:.4f}")

# Weighted Ensemble
oof_ensemble = (0.35 * oof_lgb) + (0.30 * oof_cat) + (0.20 * oof_xgb) + (0.15 * oof_ridge)
oof_ensemble = np.clip(oof_ensemble, 1.0, 5.0)

ens_rmse = root_mean_squared_error(y_speech, oof_ensemble)
ens_r = pearsonr(y_speech, oof_ensemble)[0]
ens_rho = spearmanr(y_speech, oof_ensemble)[0]
ens_mae = mean_absolute_error(y_speech, oof_ensemble)

print(f"\n[MultiModel Ensemble] RMSE: {ens_rmse:.4f} | Pearson r: {ens_r:.4f} | Spearman rho: {ens_rho:.4f} | MAE: {ens_mae:.4f}")

# 5. Full Dataset Predictions (Combining Noise Detector + Speech Models)
all_oof_preds = np.zeros(len(train_eng))
# For noise instances, predict 0.0 with 100% confidence
all_oof_preds[~is_speech_train] = 0.0
all_oof_preds[is_speech_train] = oof_ensemble

full_dataset_oof_rmse = root_mean_squared_error(train_eng["label"].values, all_oof_preds)
full_dataset_oof_r = pearsonr(train_eng["label"].values, all_oof_preds)[0]
print(f"\n[Entire Train Set OOF] RMSE: {full_dataset_oof_rmse:.4f} | Pearson r: {full_dataset_oof_r:.4f}")

# 6. Fit Models on 100% Speech Data for Test Predictions & COMPULSORY TRAIN RMSE
scaler_full = StandardScaler()
X_speech_sc = scaler_full.fit_transform(X_speech)
X_test_sc = scaler_full.transform(X_test)

m_lgb_full = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.025, num_leaves=15, max_depth=5, subsample=0.8, colsample_bytree=0.75, random_state=42, verbosity=-1)
m_lgb_full.fit(X_speech, y_speech)

m_xgb_full = xgb.XGBRegressor(n_estimators=350, learning_rate=0.025, max_depth=4, subsample=0.8, colsample_bytree=0.75, random_state=42, verbosity=0)
m_xgb_full.fit(X_speech, y_speech)

m_cat_full = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, l2_leaf_reg=3.0, random_seed=42, verbose=0)
m_cat_full.fit(X_speech, y_speech)

m_ridge_full = Ridge(alpha=15.0, random_state=42)
m_ridge_full.fit(X_speech_sc, y_speech)

speech_train_preds = (
    0.35 * m_lgb_full.predict(X_speech) +
    0.30 * m_cat_full.predict(X_speech) +
    0.20 * m_xgb_full.predict(X_speech) +
    0.15 * m_ridge_full.predict(X_speech_sc)
)
speech_train_preds = np.clip(speech_train_preds, 1.0, 5.0)

# Full training predictions across all 769 samples (including 0.0 noise samples)
full_train_preds = np.zeros(len(train_eng))
full_train_preds[~is_speech_train] = 0.0
full_train_preds[is_speech_train] = speech_train_preds

compulsory_train_rmse = root_mean_squared_error(train_eng["label"].values, full_train_preds)
print(f"\n=========================================================================")
print(f"*** COMPULSORY TRAINING DATA RMSE: {compulsory_train_rmse:.4f}")
print(f"=========================================================================\n")

# 7. Generate Test Predictions
test_speech_preds = (
    0.35 * test_preds_lgb +
    0.30 * test_preds_cat +
    0.20 * test_preds_xgb +
    0.15 * test_preds_ridge
)
test_speech_preds = np.clip(test_speech_preds, 1.0, 5.0)

# In test, if any static noise was found, set to 0.0
test_final_preds = np.where(test_eng["is_static_noise"] == 1, 0.0, test_speech_preds)

sub = pd.DataFrame({
    "filename": test_eng["filename"],
    "label": test_final_preds
})

# Reorder to match test.csv exactly
sub = test_df[["filename"]].merge(sub, on="filename", how="left")
out_path = r"e:\Assessment\shl_kaggle\submission_v3_multimodel_ensemble.csv"
sub.to_csv(out_path, index=False)
print(f"Saved submission file to: {out_path}")
print(sub.head(10))
print(f"\nTest prediction summary:\n{sub['label'].describe()}")
