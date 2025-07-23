# src/momentum_strategy.py
import backtrader as bt
import backtrader.indicators as btind
import pandas as pd

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

    def next(self):
        if not self.position:  # No open position
            if self.data.close > self.sma and self.rsi > 60:  # Long breakout
                self.buy(size=100)  # 100 shares (adjust for futures later)
                self.sell(exectype=bt.Order.Stop, price=self.data.close - 2 * self.atr[0])  # 2xATR stop
        else:  # In position
            if self.rsi < 40:  # Exit if momentum fades
                self.close()

# Setup and run backtest
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=pd.read_csv('data/spy_historical.csv', parse_dates=True, index_col='timestamp'))
cerebro.adddata(data)
cerebro.addstrategy(MomentumBreakout)
cerebro.broker.set_cash(100000)  # Starting capital
cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # Fixed 100 shares
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
results = cerebro.run()
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
print(f"Sharpe Ratio: {results[0].analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

# Plot results (requires matplotlib)
cerebro.plot()