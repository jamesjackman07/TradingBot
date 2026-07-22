from bot.data import MarketData
from bot.indicators import Indicators

import matplotlib.pyplot as plt

market = MarketData()

df = market.get_data("SPY")

# Because yfinance returns a MultiIndex
close = df["Close"]["SPY"]

# Calculate indicators
sma20 = Indicators.sma(close, 20)

# Plot
plt.figure(figsize=(12,6))

plt.plot(close, label="SPY")
plt.plot(sma20, label="20 Day SMA")

plt.title("SPY with 20 Day Moving Average")
plt.legend()

plt.grid(True)

plt.show()