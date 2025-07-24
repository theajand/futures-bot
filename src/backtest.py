# src/backtest.py - Fixed warnings with loc assignments, 'returns' calc early, OOS walk-forward, fees, stops, full DQN RL sizing
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch

# Load DQN from rl_env
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.out = torch.nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Load data
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
df = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
df['returns'] = df['Close'].pct_change().fillna(0)  # Add early for RL/error fix

# Walk-forward OOS backtest
tscv = TimeSeriesSplit(n_splits=5)
df['signal'] = 0
df['ensemble_prob'] = 0.0
scaler = StandardScaler()

for train_idx, test_idx in tscv.split(df):
    X_train = df.drop(['target', 'close_spy'], axis=1).iloc[train_idx]
    y_train = df['target'].iloc[train_idx]
    X_test = df.drop(['target', 'close_spy'], axis=1).iloc[test_idx]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    xgb = XGBClassifier()
    xgb.fit(X_train_scaled, y_train)
    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    
    xgb_prob = xgb.predict_proba(X_test_scaled)[:,1]
    rf_prob = rf.predict_proba(X_test_scaled)[:,1]
    ensemble_prob = (xgb_prob + rf_prob) / 2
    
    df.loc[df.index[test_idx], 'ensemble_prob'] = ensemble_prob  # loc for warning fix
    df.loc[df.index[test_idx], 'signal'] = np.where(ensemble_prob > 0.6, 1, np.where(ensemble_prob < 0.4, -1, 0))

df['signal'] = df['signal'].shift(1).fillna(0)

# Full RL sizing (load DQN, get action per row)
state_size = 3  # vol, equity, pos
action_size = 5  # 0.5, 0.75, 1, 1.5, 2
agent = DQN(state_size, action_size)
agent.load_state_dict(torch.load('models/dqn_sizing.pth'))
agent.eval()
df['size'] = 1.0  # Default
equity = 100000  # Start capital
for i in range(1, len(df)):
    state = np.array([df['atr_5'].iloc[i-1], equity, df['signal'].iloc[i]])
    state_tensor = torch.tensor(state).float()
    with torch.no_grad():
        action = agent(state_tensor).argmax().item()
    df.loc[df.index[i], 'size'] = [0.5, 0.75, 1.0, 1.5, 2.0][action]  # loc for warning fix
    # Update equity sim (for state)
    ret = df['returns'].iloc[i] * df['signal'].iloc[i] * 5 * df['size'].iloc[i]
    equity *= (1 + ret)

# Returns (5x leverage, 0.1% fees on trades)
trade_fee = 0.001
df['trade_cost'] = np.where(df['signal'].diff().abs() > 0, -trade_fee, 0)
df['strat_returns'] = df['signal'] * df['returns'] * 5 * df['size'] + df['trade_cost']

# Cap returns (1% stop loss, 5% max gain/day for realism)
df['strat_returns'] = np.clip(df['strat_returns'], -0.01, 0.05)

# Cumulative
df['cum_returns'] = (1 + df['strat_returns']).cumprod().fillna(1)
df['bh_cum_returns'] = (1 + df['returns']).cumprod().fillna(1)

# Metrics
total_return = df['cum_returns'].iloc[-1] - 1
annual_ret = (1 + total_return) ** (252 / len(df)) - 1 if total_return > -1 else -1
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

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(df.index, df['cum_returns'], label='Strategy')
plt.plot(df.index, df['bh_cum_returns'], label='Buy-Hold')
plt.title('Cumulative Returns')
plt.legend()
plt.show()