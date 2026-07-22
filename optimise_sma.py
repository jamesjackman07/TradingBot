from bot.research import ResearchSession
from bot.strategies.sma_cross import SMACrossoverStrategy

from backtesting.metrics import Metrics

import pandas as pd


session = ResearchSession("SPY")


fast_periods = [5, 10, 20]
slow_periods = [30, 50, 100]


results = []

total_tests = 0


for fast in fast_periods:
    for slow in slow_periods:

        if fast >= slow:
            continue

        total_tests += 1

        print(f"Testing Fast={fast} Slow={slow}")

        strategy = SMACrossoverStrategy(fast, slow)

        equity, trades = session.run(strategy)

        final = equity["Equity"].iloc[-1]

        results.append(
            {
                "Fast": fast,
                "Slow": slow,
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
print("TOP 5 SMA COMBINATIONS")
print("=" * 45)

for result in results[:5]:

    print(
        f"{result['Fast']:>2}/{result['Slow']:<3}   "
        f"Return {result['Return']:>6.2f}%   "
        f"DD {result['Drawdown']:>5.2f}%   "
        f"Trades {result['Trades']}"
    )


best = results[0]

print()
print("=" * 45)
print("BEST SMA PARAMETERS")
print("=" * 45)

print(f"Fast SMA : {best['Fast']}")
print(f"Slow SMA : {best['Slow']}")
print(f"Return   : {best['Return']:.2f}%")
print(f"Drawdown : {best['Drawdown']:.2f}%")
print(f"Trades   : {best['Trades']}")
print(f"Tests    : {total_tests}")


pd.DataFrame(results).to_csv(
    "optimization_results.csv",
    index=False
)

print()
print("Results exported to optimization_results.csv")