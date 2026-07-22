from bot.data import MarketData
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.sma_cross import SMACrossoverStrategy

from backtesting.engine import BacktestEngine
from backtesting.metrics import Metrics


market = MarketData()

df = market.get_data("SPY")

close = df["Close"]["SPY"]

strategies = [
    ("Mean Reversion", MeanReversionStrategy()),
    ("SMA Crossover", SMACrossoverStrategy())
]

results = []

for name, strategy in strategies:

    engine = BacktestEngine()

    signals = strategy.generate_signals(close)

    equity, trades = engine.run(close, signals)

    final = equity["Equity"].iloc[-1]

    results.append(
        {
            "Strategy": name,
            "Return": Metrics.total_return(10000, final),
            "Drawdown": Metrics.max_drawdown(equity["Equity"]),
            "Trades": len(trades)
        }
    )

results.sort(
    key=lambda x: x["Return"],
    reverse=True
)

print()
print("=" * 45)
print("      STRATEGY TOURNAMENT")
print("=" * 45)

for position, result in enumerate(results, start=1):

    print(
        f"{position}. "
        f"{result['Strategy']:<20}"
        f"{result['Return']:>6.2f}%   "
        f"DD {result['Drawdown']:.2f}%   "
        f"Trades {result['Trades']}"
    )