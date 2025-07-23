# src/fetch_data.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config.secrets import API_KEY, SECRET_KEY
import pandas as pd

# Initialize client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request daily SPY bars (expand range for more data)
request_params = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Day,
    start="2020-01-01",  # Back to 2020 for robust backtests
    end="2025-07-23"     # Today: July 23, 2025
)

# Fetch and save
bars = client.get_stock_bars(request_params)
df = bars.df
df.to_csv('data/spy_historical.csv')
print(f"Saved {len(df)} daily bars to data/spy_historical.csv")