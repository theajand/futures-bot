# src/ml_models.py - Refined with TimeSeries CV, Optuna tuning, feature selection, and data guard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import optuna

# Load features
df = pd.read_csv('data/features.csv', parse_dates=True, index_col='timestamp')

# Guard: Check for empty data
if len(df) == 0:
    print("Error: features.csv is empty. Rerun feature_eng.py.")
    exit()

# Clean inf/nan
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Classification (direction)
X = df.drop('target', axis=1)
y = df['target']

# Time-series CV for classification (5 splits, no shuffle)
tscv = TimeSeriesSplit(n_splits=5)
acc_lr, acc_xgb = [], []
for train_idx, test_idx in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_cv, y_train_cv)
    acc_lr.append(accuracy_score(y_test_cv, lr.predict(X_test_cv)))
    xgb = XGBClassifier()
    xgb.fit(X_train_cv, y_train_cv)
    acc_xgb.append(accuracy_score(y_test_cv, xgb.predict(X_test_cv)))

print(f"LR CV Accuracy: {np.mean(acc_lr):.2f}")
print(f"XGB CV Accuracy: {np.mean(acc_xgb):.2f}")

# Feature selection with XGB (train on full for importances)
xgb_full = XGBClassifier()
xgb_full.fit(X, y)
importances = pd.Series(xgb_full.feature_importances_, index=X.columns)
print("Top 20 Features:")
print(importances.sort_values(ascending=False).head(20))

# Retrain on top features (>0.01 importance)
top_feats = importances[importances > 0.01].index
X_top = X[top_feats]
acc_xgb_top = []
for train_idx, test_idx in tscv.split(X_top):
    X_train_top, X_test_top = X_top.iloc[train_idx], X_top.iloc[test_idx]
    y_train_top, y_test_top = y.iloc[train_idx], y.iloc[test_idx]
    xgb_top = XGBClassifier()
    xgb_top.fit(X_train_top, y_train_top)
    acc_xgb_top.append(accuracy_score(y_test_top, xgb_top.predict(X_test_top)))

print(f"XGB Accuracy on Top Features: {np.mean(acc_xgb_top):.2f}")

# Dedicated split for Optuna tuning (last 80/20 time-series)
split_idx = int(0.8 * len(X))
X_train_tune, X_test_tune = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_tune, y_test_tune = y.iloc[:split_idx], y.iloc[split_idx:]

# Optuna tuning for XGB
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    xgb_tune = XGBClassifier(**params)
    xgb_tune.fit(X_train_tune, y_train_tune)
    return accuracy_score(y_test_tune, xgb_tune.predict(X_test_tune))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best XGB Params: {study.best_params}")
print(f"Best Tuned Accuracy: {study.best_value:.2f}")

# Save tuned model (retrain with best params on full data)
best_xgb = XGBClassifier(**study.best_params)
best_xgb.fit(X, y)
best_xgb.save_model('models/xgb_tuned_model.json')

# Regression for close (CV similar)
X_reg = df.drop('close_spy', axis=1)
y_reg = df['close_spy']
mse_xgb_list = []
for train_idx, test_idx in tscv.split(X_reg):
    X_train_reg, X_test_reg = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    xgb_reg = XGBRegressor()
    xgb_reg.fit(X_train_reg, y_train_reg)
    mse_xgb_list.append(mean_squared_error(y_test_reg, xgb_reg.predict(X_test_reg)))

print(f"XGB CV MSE: {np.mean(mse_xgb_list):.4f}")