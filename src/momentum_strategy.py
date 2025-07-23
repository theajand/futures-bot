# src/momentum_strategy.py - Updated with tuned LSTM filter, trade counter, seq ffill/log, and None handle
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MomentumBreakout(bt.Strategy):
    params = (
        ('sma_period', 50),  # 50-day SMA for trend
        ('rsi_period', 14),  # 14-day RSI for momentum
        ('atr_period', 14),  # 14-day ATR for stops
    )

    def __init__(self):
        self.sma = btind.SMA(period=self.p.sma_period)  # Trend filter
        self.rsi = btind.RSI(period=self.p.rsi_period)  # Momentum trigger
        self.atr = btind.ATR(period=self.p.atr_period)  # Volatility stop

        # Load LSTM model and scaler
        self.lstm = load_model('models/lstm_model.h5')
        hist_df = pd.read_csv('data/features.csv')
        self.scaler = MinMaxScaler().fit(hist_df['close_spy'].values.reshape(-1,1))

        # Trade counter
        self.trade_count = 0

    def next(self):
        # Get last 60 closes for LSTM sequence
        closes = self.data.close.get(size=60)
        if len(closes) < 60:  # Skip if not enough data
            return

        # Log closes for debug
        print(f"Closes: {closes}")

        # Clean NaN in closes with ffill (interpolate gaps)
        closes = np.array(closes)
        if np.any(np.isnan(closes)):
            print("Filling NaN in seq")
            closes_pd = pd.Series(closes)
            closes_pd.ffill(inplace=True)
            closes = closes_pd.to_numpy()

        # Guard if still nan or invalid
        if np.any(np.isnan(closes)):
            print("Skipping invalid seq with NaN")
            return

        # Reshape and scale for LSTM input
        seq = closes.reshape(1, 60, 1)
        seq_scaled = self.scaler.transform(seq.reshape(-1,1)).reshape(1, 60, 1)

        # Predict next close (scaled)
        pred_scaled = self.lstm.predict(seq_scaled)[0][0]

        # Inverse scale to real price
        pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        # Log for debug
        print(f"LSTM Pred: {pred:.2f}, Current Close: {self.data.close[0]:.2f}")

        if not self.position:  # No open position
            if self.data.close > self.sma and self.rsi > 60 and pred > self.data.close[0] * 0.99:  # LSTM filter with 1% buffer
                self.buy(size=100)  # 100 shares (adjust for futures later)
                self.sell(exectype=bt.Order.Stop, price=self.data.close - 2 * self.atr[0])  # 2xATR stop
                self.trade_count += 1
                print(f"Trade #{self.trade_count} executed")
        else:  # In position
            if self.rsi < 40:  # Exit if momentum fades
                self.close()

# Setup and run backtest
cerebro = bt.Cerebro()
hist_df = pd.read_csv('data/spy_historical.csv', parse_dates=True, index_col='timestamp')
hist_df = hist_df.rename(columns={'close_spy': 'close', 'open_spy': 'open', 'high_spy': 'high', 'low_spy': 'low', 'volume_spy': 'volume'})  # Rename for Backtrader
hist_df.fillna(0, inplace=True)  # Clean NaN in data load
data = bt.feeds.PandasData(dataname=hist_df)
cerebro.adddata(data)
cerebro.addstrategy(MomentumBreakout)
cerebro.broker.set_cash(100000)  # Starting capital
cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # Fixed 100 shares
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
results = cerebro.run()
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

# Handle None in metrics
sharpe_analysis = results[0].analyzers.sharpe.get_analysis()
sharpe = sharpe_analysis['sharperatio'] if sharpe_analysis['sharperatio'] is not None else 'N/A'
print(f"Sharpe Ratio: {sharpe if sharpe == 'N/A' else f'{sharpe:.2f}'}")

drawdown_analysis = results[0].analyzers.drawdown.get_analysis()
max_drawdown = drawdown_analysis['max']['drawdown'] if 'max' in drawdown_analysis else 'N/A'
print(f"Max Drawdown: {max_drawdown if max_drawdown == 'N/A' else f'{max_drawdown:.2f}%'}")

# Plot results (with voloverlay=False to fix NaN/Inf)
cerebro.plot(voloverlay=False)