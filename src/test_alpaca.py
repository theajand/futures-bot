# src/test_alpaca.py - Updated for modern alpaca-py SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config.secrets import API_KEY, SECRET_KEY  # <-- This line: Import VARIABLES, not keys!

# Initialize StockHistoricalDataClient for paper trading (keys required for stocks)
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Set up request params for SPY 1-minute bars (last 5 days as ES proxy)
request_params = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Minute,
    start="2025-07-17",
    end="2025-07-22",
    limit=1000
)

# Fetch bars
bars = client.get_stock_bars(request_params)

# Convert to DataFrame and print head
df = bars.df
print(df.head())  # OHLCV (open, high, low, close, volume) for SPY
df.to_csv('data/spy_sample.csv')  # Save for backtesting
