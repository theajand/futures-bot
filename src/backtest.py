# src/backtest.py - Full backtest with ensemble signals (XGB/RF), RL sizing
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
df = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)

# Load models (XGB/RF from ml_models)
xgb = XGBClassifier()
xgb.load_model('models/xgb_tuned_model.json')
rf = RandomForestClassifier()  # Retrain RF for simplicity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['target', 'close_spy'], axis=1))

# Generate ensemble signals
xgb_prob = xgb.predict_proba(X_scaled)[:,1]
rf.fit(X_scaled, df['target'])
rf_prob = rf.predict_proba(X_scaled)[:,1]
df['ensemble_prob'] = (xgb_prob + rf_prob) / 2
df['signal'] = np.where(df['ensemble_prob'] > 0.6, 1, np.where(df['ensemble_prob'] < 0.4, -1, 0))
df['signal'] = df['signal'].shift(1)  # Lag for realism

# RL sizing approx (use rl_env.py DQN in prod)
high_vol_thresh = df['atr_14'].median()
df['size'] = np.where(df['atr_14'].shift(1) < high_vol_thresh, 2, 0.5)

# Returns (5x leverage, no fees yet—add locally)
df['returns'] = df['Close'].pct_change()
df['strat_returns'] = df['signal'] * df['returns'] * 5 * df['size']
df['cum_returns'] = (1 + df['strat_returns']).cumprod().fillna(1)
df['bh_cum_returns'] = (1 + df['returns']).cumprod().fillna(1)

# Metrics
total_return = df['cum_returns'].iloc[-1] - 1
annual_ret = total_return ** (252 / len(df)) - 1 if total_return > -1 else -1
sharpe = df['strat_returns'].mean() / df['strat_returns'].std() * np.sqrt(252) if df['strat_returns'].std() != 0 else 0
max_dd = (df['cum_returns'] / df['cum_returns'].cummax() - 1).min()
win_rate = (df['strat_returns'] > 0).mean()
num_trades = (df['signal'].diff().abs() > 0).sum()

print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annual_ret:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Number of Trades: {num_trades}")
print(f"Buy-Hold Return: {df['bh_cum_returns'].iloc[-1] - 1:.2%}")

# Optional: Plot cumulative returns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(df.index, df['cum_returns'], label='Strategy')
plt.plot(df.index, df['bh_cum_returns'], label='Buy-Hold')
plt.title('Cumulative Returns')
plt.legend()
plt.show()