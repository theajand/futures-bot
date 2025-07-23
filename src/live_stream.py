# src/live_stream.py - Optimized for speed: Added logging, heartbeat, historical sim mode for fast testing
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import asyncio
import logging
from datetime import datetime, timedelta
API_KEY = 'PKMFBMDS1D83ZVC19722'  # Your key
API_SECRET = 'JWImhmQrVuTJxgsXhtemRFXGHTYADgtZ5SQ8Nkre'  # Your secret

# Setup logging (to see activity even when "slow")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Historical sim func (for fast testing—pulls past bars, "streams" them quick)
async def simulate_historical(symbol='SPY', days_back=1):
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.now() - timedelta(days=days_back),
        end=datetime.now()
    )
    bars = client.get_stock_bars(request).df.reset_index()
    for _, bar in bars.iterrows():
        await bar_handler(bar)  # Mimic live
        await asyncio.sleep(0.1)  # "Speed up" sim—adjust to 60 for real-time feel

# Live bar handler
async def bar_handler(bar):
    logging.info(f"New 1-min bar for {bar.symbol}: Open {bar.open}, Close {bar.close}, Vol {bar.volume}")
    # Your bot logic: Compute feats, ensemble signal, RL size, trade

# Heartbeat to "feel" alive during waits
async def heartbeat():
    while True:
        logging.info("Heartbeat: Waiting for next bar... (Market open? Check time.)")
        await asyncio.sleep(10)  # Every 10s

async def main(live_mode=True):
    if live_mode:
        stream = StockDataStream(API_KEY, API_SECRET)
        stream.subscribe_bars(bar_handler, 'SPY')
        asyncio.create_task(heartbeat())  # Run heartbeat parallel
        await stream._run_forever()  # alpaca-py run
    else:
        await simulate_historical()  # Fast sim

# Run: Toggle live_mode=False for quick historical "stream"
asyncio.run(main(live_mode=False))  # Set False for sim speed