import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import lightgbm as lgb
import os

train_labels = pd.read_csv(r"e:\Assessment\shl_kaggle\data\Dataset_Final\train.csv")
train_feat = pd.read_csv(r"e:\Assessment\shl_kaggle\data\features\train_acoustic_features.csv")
test_feat = pd.read_csv(r"e:\Assessment\shl_kaggle\data\features\test_acoustic_features.csv")
test_df = pd.read_csv(r"e:\Assessment\shl_kaggle\data\Dataset_Final\test.csv")

df = pd.merge(train_labels, train_feat, on="filename")
drop_cols = ["filename", "label", "error", "is_silent_audio"]
feature_cols = [c for c in train_feat.columns if c not in drop_cols]

print(f"Number of acoustic features: {len(feature_cols)}")
X = df[feature_cols].values
y = df["label"].values

X_test = test_feat[feature_cols].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_ridge = np.zeros(len(df))
test_preds_ridge = np.zeros(len(test_feat))

oof_preds_lgb = np.zeros(len(df))
test_preds_lgb = np.zeros(len(test_feat))

print("\n--- Training 5-Fold Cross-Validation ---")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X[trn_idx], y[trn_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # 1. Ridge Regressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    oof_preds_ridge[val_idx] = np.clip(ridge.predict(X_val_scaled), 0.0, 5.0)
    test_preds_ridge += np.clip(ridge.predict(X_test_scaled), 0.0, 5.0) / 5.0
    
    # 2. LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(X_train, y_train)
    oof_preds_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 5.0)
    test_preds_lgb += np.clip(model_lgb.predict(X_test), 0.0, 5.0) / 5.0

# Evaluate OOF
ridge_rmse = root_mean_squared_error(y, oof_preds_ridge)
ridge_pearson = pearsonr(y, oof_preds_ridge)[0]

lgb_rmse = root_mean_squared_error(y, oof_preds_lgb)
lgb_pearson = pearsonr(y, oof_preds_lgb)[0]

# Ensemble blend: 50% Ridge + 50% LightGBM
oof_blend = 0.5 * oof_preds_ridge + 0.5 * oof_preds_lgb
blend_rmse = root_mean_squared_error(y, oof_blend)
blend_pearson = pearsonr(y, oof_blend)[0]

print(f"\n[Ridge]    OOF RMSE: {ridge_rmse:.4f} | Pearson r: {ridge_pearson:.4f}")
print(f"[LightGBM] OOF RMSE: {lgb_rmse:.4f} | Pearson r: {lgb_pearson:.4f}")
print(f"[Ensemble] OOF RMSE: {blend_rmse:.4f} | Pearson r: {blend_pearson:.4f}")

# Train on FULL training data to compute COMPULSORY TRAIN RMSE
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)
X_test_scaled = scaler_full.transform(X_test)

ridge_full = Ridge(alpha=10.0, random_state=42)
ridge_full.fit(X_full_scaled, y)
train_preds_ridge = np.clip(ridge_full.predict(X_full_scaled), 0.0, 5.0)

lgb_full = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
lgb_full.fit(X, y)
train_preds_lgb = np.clip(lgb_full.predict(X), 0.0, 5.0)

train_preds_blend = 0.5 * train_preds_ridge + 0.5 * train_preds_lgb
compulsory_train_rmse = root_mean_squared_error(y, train_preds_blend)
print(f"\n========================================================")
print(f"COMPULSORY TRAINING DATA RMSE: {compulsory_train_rmse:.4f}")
print(f"========================================================\n")

# Generate test predictions
test_preds_blend = 0.5 * test_preds_ridge + 0.5 * test_preds_lgb
sub_df = pd.DataFrame({
    "filename": test_feat["filename"],
    "label": test_preds_blend
})

# Align with test.csv exact order
sub_df = test_df[["filename"]].merge(sub_df, on="filename", how="left")
out_sub_path = r"e:\Assessment\shl_kaggle\submission_acoustic_ensemble.csv"
sub_df.to_csv(out_sub_path, index=False)
print(f"Saved submission to {out_sub_path}")
print(sub_df.head(10))
print(f"Summary stats:\n{sub_df['label'].describe()}")
