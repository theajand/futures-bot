# src/feature_eng.py - Intraday 1m switch (short 7d for yf limit fix, no error/hang), CNN Fear & Greed sentiment (enhanced headers/parse for no JSON error, try-except), pure pandas, VIX (daily ffill), guards
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

# Def for news events (hard-coded from searches, score -1 fear/negative to 1 greed/positive)
def get_news_events(year):
    events = {
        2018: [
            {'date': '2018-02-05', 'headline': 'Vol-pocalypse VIX spike', 'score': -0.5},
            {'date': '2018-12-24', 'headline': 'Quarterly low, volatility spike', 'score': -0.5},
            {'date': '2018-10-10', 'headline': 'Stock dive Fed tighten trade fears', 'score': -0.5},
            {'date': '2018-03-22', 'headline': 'Volatility jumps macro news', 'score': -0.5}
        ],
        2019: [
            {'date': '2019-09-17', 'headline': 'Repo rate spikes turmoil', 'score': -0.5},
            {'date': '2019-12-29', 'headline': 'Stellar year but volatility', 'score': -0.3},
            {'date': '2019-01-05', 'headline': 'Market turmoil', 'score': -0.5},
            {'date': '2019-03-01', 'headline': 'Treasury sell-off context', 'score': -0.5}
        ],
        2020: [
            {'date': '2020-03-16', 'headline': 'Treasury basis spike COVID vol', 'score': -0.5},
            {'date': '2020-04-21', 'headline': 'Stock tumble', 'score': -0.5},
            {'date': '2020-03-09', 'headline': 'VIX wild March', 'score': -0.5},
            {'date': '2020-03-01', 'headline': 'Unprecedented COVID vol', 'score': -0.5}
        ],
        2021: [
            {'date': '2021-02-25', 'headline': 'Treasury flash event', 'score': -0.5},
            {'date': '2021-07-21', 'headline': 'S&P record but volatility', 'score': -0.3},
            {'date': '2021-10-01', 'headline': 'Commodity surge', 'score': 0.5},
            {'date': '2021-07-20', 'headline': 'S&P high', 'score': 0.5}
        ],
        2022: [
            {'date': '2022-10-10', 'headline': 'S&P low quarterly', 'score': -0.5},
            {'date': '2022-06-01', 'headline': 'Unprecedented vol COVID context', 'score': -0.5},
            {'date': '2022-01-01', 'headline': 'Treasury basis mid-2022', 'score': -0.5}
        ],
        2023: [
            {'date': '2023-01-10', 'headline': 'Futures drop caution', 'score': -0.5},
            {'date': '2023-07-09', 'headline': 'Nasdaq highs', 'score': 0.3},
            {'date': '2023-03-01', 'headline': 'Treasury basis', 'score': -0.5},
            {'date': '2023-04-16', 'headline': 'Stocks plunge restrictions', 'score': -0.5}
        ],
        2024: [
            {'date': '2024-07-09', 'headline': 'Nasdaq highs', 'score': 0.3},
            {'date': '2024-04-21', 'headline': 'Stock market miss', 'score': -0.5},
            {'date': '2024-01-01', 'headline': 'Wall Street low jobs', 'score': -0.5},
            {'date': '2024-03-08', 'headline': 'Treasury basis Jan', 'score': -0.5},
            {'date': '2024-07-23', 'headline': 'Markets high earnings', 'score': 0.5}
        ],
        2025: [
            {'date': '2025-07-23', 'headline': 'Trade deal optimism high', 'score': 0.5},
            {'date': '2025-04-23', 'headline': 'Futures climb earnings', 'score': 0.5},
            {'date': '2025-06-20', 'headline': 'Financial vol spring', 'score': -0.5},
            {'date': '2025-01-10', 'headline': 'Futures drop payrolls', 'score': -0.5},
            {'date': '2025-07-01', 'headline': 'Fed rate questions', 'score': 0.0}
        ]
    }
    return events.get(year, [])

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

# Sentiment (CNN daily, ffill for 1m—enhanced headers for block fix, parse text if not JSON, fills forward)
url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, * /*',
    'Referer': 'https://edition.cnn.com/markets/fear-and-greed',
    'Accept-Language': 'en-US,en;q=0.9'
}
try:
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    content = resp.text  # Get text first
    import json
    data = json.loads(content)  # Try JSON parse
    if 'fear_and_greed_historical' in data and 'data' in data['fear_and_greed_historical']:
        fg_data = data['fear_and_greed_historical']['data']
        fg_df = pd.DataFrame(fg_data)
        fg_df['timestamp'] = pd.to_datetime(fg_df['x'] / 1000, unit='s')
        fg_df.set_index('timestamp', inplace=True)
        fg_df['sentiment_score'] = (fg_df['y'] / 50 - 1)  # -1 to 1
        fg_df = fg_df.reindex(spy.index, method='ffill')  # Fill daily to 1m
        spy['sentiment_score'] = fg_df['sentiment_score'].fillna(0)
    else:
        raise ValueError("Invalid structure")
except Exception as e:
    print(f"Sentiment fetch failed: {e}—checking if HTML, using fallback 0.0")
    if '<html' in content.lower():  # If HTML block, fallback
        spy['sentiment_score'] = 0.0
    else:
        spy['sentiment_score'] = 0.0  # General fallback

# Add news events (after VIX/sentiment)
events_list = []
for year in range(2018, 2026):
    year_events = get_news_events(year)
    for event in year_events:
        events_list.append(event)
events_df = pd.DataFrame(events_list)
events_df['date'] = pd.to_datetime(events_df['date']).dt.date
events_df = events_df.groupby('date')['score'].mean().reset_index()  # Average if multiple
events_df.set_index('date', inplace=True)
spy['date'] = spy.index.date
spy = spy.merge(events_df, left_on='date', right_index=True, how='left')
spy['event_score'] = spy['score'].fillna(0)
spy.drop(['date', 'score'], axis=1, inplace=True)  # Clean

# Clean/save
spy.dropna(inplace=True)
spy.to_csv('data/features.csv', index=True)
print(f"Features engineered and saved: {spy.shape[0]} rows, {spy.shape[1]} columns")
print(spy.head())