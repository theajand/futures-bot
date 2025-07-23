# src/ml_models.py - LSTM scaled, no LSTM in ensemble, TimeSeries CV, Optuna, Mac-optimized MPS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=[0])
if len(df) == 0:
    print("Error: features.csv empty. Rerun feature_eng.py.")
    exit()
df = df.select_dtypes(include=[np.number])
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Scale for LSTM (features + target for regression)
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
scaled_df['target'] = df['target']  # Unscale label
price_scaler = MinMaxScaler()  # For close_spy
scaled_df['close_spy'] = price_scaler.fit_transform(df[['close_spy']])

# Dataset for LSTM
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32), torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=20, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() if output_size == 1 else None

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out) if self.sigmoid else out

# Train LSTM
def train_lstm(X_train, y_train, input_size, epochs=5, batch_size=128, is_class=True):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device} for LSTM training")
    dataset = TimeSeriesDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size, output_size=1 if is_class else 1).to(device)
    criterion = nn.BCELoss() if is_class else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for seq, labels in loader:
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(seq).squeeze()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        print(f"LSTM Training: Epoch {epoch+1}/{epochs} Loss {loss.item():.4f}")
    return model

# Predict with LSTM
def predict_lstm(model, X_test, seq_len=30):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    preds = []
    for i in range(len(X_test) - seq_len):
        seq = torch.tensor(X_test[i:i+seq_len]).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(seq).item()
        preds.append(pred)
    return np.array(preds)

# Classification
X = scaled_df.drop(['target', 'close_spy'], axis=1).values
y = df['target'].values

tscv = TimeSeriesSplit(n_splits=2)
acc_lr, acc_xgb, acc_rf, acc_lstm, acc_ensemble = [], [], [], [], []
for split_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"Starting CV Split {split_num}/2...")
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]

    # LR
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_cv, y_train_cv)
    acc_lr.append(accuracy_score(y_test_cv, lr.predict(X_test_cv)))

    # XGB
    xgb = XGBClassifier()
    xgb.fit(X_train_cv, y_train_cv)
    xgb_prob = xgb.predict_proba(X_test_cv)[:,1]
    acc_xgb.append(accuracy_score(y_test_cv, (xgb_prob > 0.5).astype(int)))

    # RF
    rf = RandomForestClassifier()
    rf.fit(X_train_cv, y_train_cv)
    rf_prob = rf.predict_proba(X_test_cv)[:,1]
    acc_rf.append(accuracy_score(y_test_cv, (rf_prob > 0.5).astype(int)))

    # LSTM
    lstm = train_lstm(X_train_cv, y_train_cv, input_size=X.shape[1])
    lstm_prob = predict_lstm(lstm, X_test_cv)
    acc_lstm.append(accuracy_score(y_test_cv[30:], (lstm_prob > 0.5).astype(int)))

    # Ensemble (XGB + RF only for stability)
    ensemble_prob = (xgb_prob + rf_prob) / 2
    acc_ensemble.append(accuracy_score(y_test_cv, (ensemble_prob > 0.6).astype(int)))
    print(f"CV Split {split_num}/2 Done: Interim Ensemble Acc {acc_ensemble[-1]:.2f}")

print(f"LR CV Accuracy: {np.mean(acc_lr):.2f}")
print(f"XGB CV Accuracy: {np.mean(acc_xgb):.2f}")
print(f"RF CV Accuracy: {np.mean(acc_rf):.2f}")
print(f"LSTM CV Accuracy: {np.mean(acc_lstm):.2f}")
print(f"Ensemble CV Accuracy: {np.mean(acc_ensemble):.2f}")

# Feature importances
print("Computing Feature Importances...")
xgb_full = XGBClassifier()
xgb_full.fit(X, y)
importances = pd.Series(xgb_full.feature_importances_, index=df.drop(['target', 'close_spy'], axis=1).columns)
print("Top 20 Features:")
print(importances.sort_values(ascending=False).head(20))

# Optuna for XGB
split_idx = int(0.8 * len(X))
X_train_tune, X_test_tune = X[:split_idx], X[split_idx:]
y_train_tune, y_test_tune = y[:split_idx], y[split_idx:]
def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 200), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)}
    xgb_tune = XGBClassifier(**params)
    xgb_tune.fit(X_train_tune, y_train_tune)
    return accuracy_score(y_test_tune, xgb_tune.predict(X_test_tune))
study = optuna.create_study(direction='maximize')
print("Starting Optuna Tuning (10 trials)...")
study.optimize(objective, n_trials=10)
print(f"Best XGB Params: {study.best_params}")
print(f"Best Tuned Accuracy: {study.best_value:.2f}")
best_xgb = XGBClassifier(**study.best_params)
best_xgb.fit(X, y)
best_xgb.save_model('models/xgb_tuned_model.json')

# Regression
X_reg = scaled_df.drop('close_spy', axis=1).values
y_reg = scaled_df['close_spy'].values  # Use scaled prices
mse_xgb_list, mse_lstm_list = [], []
tscv_reg = TimeSeriesSplit(n_splits=2)
for split_num, (train_idx, test_idx) in enumerate(tscv_reg.split(X_reg), 1):
    print(f"Starting Regression CV Split {split_num}/2...")
    X_train_reg, X_test_reg = X_reg[train_idx], X_reg[test_idx]
    y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]
    xgb_reg = XGBRegressor()
    xgb_reg.fit(X_train_reg, y_train_reg)
    mse_xgb_list.append(mean_squared_error(y_test_reg, xgb_reg.predict(X_test_reg)))

    lstm_reg = train_lstm(X_train_reg, y_train_reg, input_size=X_reg.shape[1], is_class=False)
    lstm_pred = predict_lstm(lstm_reg, X_test_reg)
    mse_lstm_list.append(mean_squared_error(y_test_reg[30:], lstm_pred))
    print(f"Regression CV Split {split_num}/2 Done")

print(f"XGB CV MSE: {np.mean(mse_xgb_list):.4f}")
print(f"LSTM CV MSE: {np.mean(mse_lstm_list):.4f}")

# Save scaler for backtest
import joblib
joblib.dump(price_scaler, 'models/price_scaler.pkl')