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
    sort_by="Return",      # Try "Sharpe", "Win Rate", "Profit Factor", "Drawdown"
    ascending=False        # Use True for Drawdown
)

print(results)

results.to_csv(
    "optimization_results.csv",
    index=False
)