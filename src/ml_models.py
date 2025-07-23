# src/ml_models.py - With MSE import and data guard
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error  # <-- Added MSE here
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
print(f"LR Accuracy: {accuracy_score(y_test, preds_lr):.2f}")

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
preds_xgb = xgb.predict(X_test)
print(f"XGB Accuracy: {accuracy_score(y_test, preds_xgb):.2f}")
xgb.save_model('models/xgb_model.json')

# Regression for close (compare to LSTM)
X_reg = df.drop('close_spy', axis=1)  # Adjust if column is 'close'
y_reg = df['close_spy']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

xgb_reg = XGBRegressor()
xgb_reg.fit(X_train_reg, y_train_reg)
preds_xgb_reg = xgb_reg.predict(X_test_reg)
mse_xgb = mean_squared_error(y_test_reg, preds_xgb_reg)
print(f"XGB MSE: {mse_xgb:.4f}")