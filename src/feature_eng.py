# src/feature_eng.py - With date fix, guards, all-NaN drop, fillna, and raw data clean
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import yfinance as yf
import os

os.makedirs('data', exist_ok=True)

ticker = 'SPY'
df = yf.download(ticker, start='2020-01-01', end='2025-07-22', interval='1d')  # <-- Change end to last available
df.index.name = 'timestamp'
print(f"After download: {len(df)} rows")  # Debug: Expect ~1400

if df.empty:  # <-- Guard: Catch empty fetch
    print("Error: No data fetched from yfinance. Check dates/internet.")
    exit()  # Stop to avoid empty CSV

if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip().lower() for col in df.columns.values]
else:
    df.columns = df.columns.str.lower()

df.to_csv('data/spy_historical.csv')  # Save raw early
df.fillna(0, inplace=True)  # <-- New: Clean NaN in raw data early to prevent propagation
print(f"Saved raw and cleaned: {len(df)} rows")  # Updated print for confirm

df = add_all_ta_features(
    df,
    open='open_spy',
    high='high_spy',
    low='low_spy',
    close='close_spy',
    volume='volume_spy'
)
print(f"After TA-Lib: {len(df)} rows")  # Debug

# Customs/lags (your code)
for window in [5, 10, 20, 50]:
    df[f'rolling_mean_{window}'] = df['close_spy'].rolling(window).mean()
    df[f'rolling_std_{window}'] = df['close_spy'].rolling(window).std()
    df[f'momentum_{window}'] = df['close_spy'] - df['close_spy'].shift(window)
    df[f'rate_of_change_{window}'] = df['close_spy'].pct_change(window)
    df[f'volatility_ratio_{window}'] = df[f'rolling_std_{window}'] / df['close_spy']

for lag in [1, 2, 3, 5]:
    df[f'close_lag_{lag}'] = df['close_spy'].shift(lag)
    df[f'close_diff_{lag}'] = df['close_spy'].diff(lag)

print(f"After customs: {len(df)} rows")

df['target'] = np.where(df['close_spy'].shift(-1) > df['close_spy'], 1, 0)

# Debug: Find all-NaN columns (causing drop to 0)
na_counts = df.isna().sum()
all_nan_cols = na_counts[na_counts == len(df)].index.tolist()
print(f"All-NaN columns: {all_nan_cols} ({len(all_nan_cols)} total)")

# Fix: Drop all-NaN columns
df.drop(columns=all_nan_cols, inplace=True)
print(f"After dropping all-NaN cols: {len(df)} rows, {len(df.columns)} cols")

# Fill remaining NaNs with 0 (e.g., initial rolling/shift NaNs)
df.fillna(0, inplace=True)
print(f"After fillna: {df.isna().sum().sum()} total NaNs (should be 0)")

# Optional dropna (now safe, but fill handles most)
df.dropna(inplace=True)
print(f"After dropna: {len(df)} rows")  # Debug: Expect ~1343

if len(df) < 100:  # <-- Guard: Too few rows = bad data
    print("Error: Too few rows after processing. Check fetch/dates.")
    exit()

df.to_csv('data/features.csv')
print(df.head())