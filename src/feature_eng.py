# src/feature_eng.py - Enhanced with VIX, pure pandas indicators, added ATR/vol_ratio, sentiment temped, no duplicate timestamp
import pandas as pd
import yfinance as yf
import numpy as np
# Removed requests/bs4/vader imports for now—sentiment temped to 0

# Custom RSI (Wilder's method using ewm)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Custom ATR (Average True Range for vol)
def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()  # Wilder ewm
    return atr

# Download SPY data (daily for now; later intraday)
start_date = '2018-01-01'  # 5+ years for robust training
end_date = '2025-07-23'    # Up to current
spy = yf.download('SPY', start=start_date, end=end_date)
spy = spy.droplevel(1, axis=1)  # Flatten MultiIndex columns (drop ticker level)
# No spy['timestamp'] = spy.index — index is already set

# Add basics: RSI, SMAs, MACD (pure pandas)
spy['rsi_14'] = calculate_rsi(spy['Close'], window=14)
spy['sma_20'] = spy['Close'].rolling(window=20).mean()
spy['sma_50'] = spy['Close'].rolling(window=50).mean()
spy['sma_200'] = spy['Close'].rolling(window=200).mean()
spy['ema_12'] = spy['Close'].ewm(span=12, adjust=False).mean()
spy['ema_26'] = spy['Close'].ewm(span=26, adjust=False).mean()
spy['macd'] = spy['ema_12'] - spy['ema_26']
spy['macd_signal'] = spy['macd'].ewm(span=9, adjust=False).mean()

# New feats: ATR, vol_ratio
spy['atr_14'] = calculate_atr(spy['High'], spy['Low'], spy['Close'], window=14)
spy['vol_ratio'] = spy['Volume'] / spy['Volume'].rolling(window=20).mean()

# Targets: Classification (1=up, 0=down) and regression (next close)
spy['target'] = np.where(spy['Close'].shift(-1) > spy['Close'], 1, 0)
spy['close_spy'] = spy['Close'].shift(-1)  # Predict next close

# Add VIX (volatility index)
vix = yf.download('^VIX', start=start_date, end=end_date)
vix = vix.droplevel(1, axis=1)  # Flatten
spy = spy.join(vix['Close'], rsuffix='_vix')  # Align on dates
spy['vix_ma_5'] = spy['Close_vix'].rolling(5).mean()
spy['vix_ma_20'] = spy['Close_vix'].rolling(20).mean()
spy.rename(columns={'Close_vix': 'vix_close'}, inplace=True)

# Sentiment: Temp set to 0 (current scrape not historical—upgrade next)
spy['sentiment_score'] = 0.0  # Placeholder; adds no edge yet

# Clean and save
spy.dropna(inplace=True)
spy.replace([np.inf, -np.inf], 0, inplace=True)
spy.to_csv('data/features.csv', index=True)

print(f"Features engineered and saved: {spy.shape[0]} rows, {spy.shape[1]} columns")
print(spy.head())