from bot.data import MarketData
from bot.strategies.mean_reversion import MeanReversionStrategy
from backtesting.engine import BacktestEngine
from backtesting.metrics import Metrics

import matplotlib.pyplot as plt

# Download data
market = MarketData()
df = market.get_data("SPY")

# Extract closing prices
close = df["Close"]["SPY"]

# Generate strategy signals
strategy = MeanReversionStrategy()
signals = strategy.generate_signals(close)

# Run backtest
engine = BacktestEngine(initial_cash=10000)
equity_curve, trades = engine.run(close, signals)

# Results
initial = 10000
final = equity_curve["Equity"].iloc[-1]

print()

print(f"Initial Capital : ${initial:,.2f}")
print(f"Final Capital   : ${final:,.2f}")
print(f"Total Return    : {Metrics.total_return(initial, final):.2f}%")
print(f"Max Drawdown    : {Metrics.max_drawdown(equity_curve['Equity']):.2f}%")
print(f"Trades          : {len(trades)}")

print("\nTrade History")
print("-" * 30)

for trade in trades:
    print(
        f"{trade.trade_type:<5} "
        f"Day {trade.index:<3} "
        f"Price ${trade.price:.2f}"
    )

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve["Equity"])

plt.title("Equity Curve")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)

plt.show()