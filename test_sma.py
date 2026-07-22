from bot.research import ResearchSession
from bot.strategies.sma_cross import SMACrossoverStrategy

from backtesting.metrics import Metrics

import matplotlib.pyplot as plt


session = ResearchSession("SPY")

strategy = SMACrossoverStrategy()

equity, trades = session.run(strategy)

final = equity["Equity"].iloc[-1]

print()
print(f"Initial Capital : $10,000.00")
print(f"Final Capital   : ${final:,.2f}")
print(f"Total Return    : {Metrics.total_return(10000, final):.2f}%")
print(f"Max Drawdown    : {Metrics.max_drawdown(equity['Equity']):.2f}%")
print(f"Trades          : {len(trades)}")

print()
print("Trade History")
print("-" * 30)

for trade in trades:

    print(
        f"{trade.trade_type:<5} "
        f"Day {trade.index:<3} "
        f"Price ${trade.price:.2f} "
        f"Shares {trade.shares:.2f}"
    )

plt.figure(figsize=(12,6))

plt.plot(equity["Equity"])

plt.title("Portfolio Equity")

plt.grid(True)

plt.show()