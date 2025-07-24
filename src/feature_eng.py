# src/feature_eng.py - Intraday 1m switch (short 7d for yf limit fix, no error/hang), CNN Fear & Greed sentiment (fixed fetch with try-except), pure pandas, VIX (daily ffill), guards
import pandas as pd
import yfinance as yf
import numpy as np
import requests

# Custom RSI/ATR (short windows for intraday)
def calculate_rsi(series, window=5):  # Short for 1m
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window=5):  # Short for 1m
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr

# Download SPY intraday (1m, last 7d for yf limit)
start_date = '2025-07-16'  # ~7d back for 1m (yf max free per request)
end_date = '2025-07-23'
spy = yf.download('SPY', start=start_date, end=end_date, interval='1m')
spy = spy.droplevel(1, axis=1) if spy.columns.nlevels > 1 else spy
spy.index = spy.index.tz_localize(None)  # Strip timezone for alignment

# Add indicators (short for intraday)
spy['rsi_5'] = calculate_rsi(spy['Close'], window=5)  # Rename for short
spy['sma_10'] = spy['Close'].rolling(10).mean()  # Short SMA
spy['sma_20'] = spy['Close'].rolling(20).mean()
spy['ema_5'] = spy['Close'].ewm(span=5, adjust=False).mean()  # Short EMA
spy['ema_10'] = spy['Close'].ewm(span=10, adjust=False).mean()
spy['macd'] = spy['ema_5'] - spy['ema_10']
spy['macd_signal'] = spy['macd'].ewm(span=4, adjust=False).mean()  # Short
spy['atr_5'] = calculate_atr(spy['High'], spy['Low'], spy['Close'], window=5)
spy['vol_ratio'] = spy['Volume'] / spy['Volume'].rolling(10).mean()

# Targets (next 1m close)
spy['target'] = np.where(spy['Close'].shift(-1) > spy['Close'], 1, 0)
spy['close_spy'] = spy['Close'].shift(-1)

# VIX (daily, ffill for 1m—since 1m VIX not avail free)
vix = yf.download('^VIX', start=start_date, end=end_date)
vix = vix.droplevel(1, axis=1) if vix.columns.nlevels > 1 else vix
vix.index = vix.index.tz_localize(None)  # Strip timezone for alignment
vix = vix.reindex(spy.index, method='ffill')  # Ffill daily to 1m
spy = spy.join(vix['Close'], rsuffix='_vix')
spy['vix_ma_5'] = spy['Close_vix'].rolling(5).mean()
spy['vix_ma_20'] = spy['Close_vix'].rolling(20).mean()
spy.rename(columns={'Close_vix': 'vix_close'}, inplace=True)
spy['vix_close'].fillna(method='ffill', inplace=True)  # Fill any gaps

# Sentiment (CNN daily, ffill for 1m—fetches once, fills forward)
url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
try:
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if 'fear_and_greed_historical' in data and 'data' in data['fear_and_greed_historical']:
        fg_data = data['fear_and_greed_historical']['data']
        fg_df = pd.DataFrame(fg_data)
        fg_df['timestamp'] = pd.to_datetime(fg_df['x'] / 1000, unit='s')
        fg_df.set_index('timestamp', inplace=True)
        fg_df['sentiment_score'] = (fg_df['y'] / 50 - 1)  # -1 to 1
        fg_df = fg_df.reindex(spy.index, method='ffill')  # Fill daily to 1m
        spy['sentiment_score'] = fg_df['sentiment_score'].fillna(0)
    else:
        raise ValueError("Invalid JSON")
except Exception as e:
    print(f"Sentiment fetch failed: {e}—using fallback 0.0")
    spy['sentiment_score'] = 0.0

# Clean/save
spy.dropna(inplace=True)
spy.to_csv('data/features.csv', index=True)
print(f"Features engineered and saved: {spy.shape[0]} rows, {spy.shape[1]} columns")
print(spy.head())