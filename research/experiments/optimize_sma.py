from bot.research import ResearchSession
from optimization.optimizer import Optimizer
from bot.strategies.sma_cross import SMACrossoverStrategy


session = ResearchSession("SPY")

optimizer = Optimizer(session)

results = optimizer.optimize(
    strategy_class=SMACrossoverStrategy,
    parameters={
        "fast": [5, 10, 20],
        "slow": [30, 50, 100]
    },
    sort_by="Return",
    ascending=False
)

print(results.best())

print()

print("Top 5")
print(results.head())

results.export_csv()