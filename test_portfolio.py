import pandas as pd

from bot.data import MarketData

from bot.strategies.sma_cross import (
    SMACrossoverStrategy
)

from bot.strategies.mean_reversion import (
    MeanReversionStrategy
)

from walkforward.run import WalkForwardRunner

from backtesting.metrics import Metrics


# --------------------------------
# Settings
# --------------------------------

INITIAL_CAPITAL = 10000

TRAIN_SIZE = 504
TEST_SIZE = 126
STEP_SIZE = 126


# --------------------------------
# Download data
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="15y"
)


# --------------------------------
# Strategy configurations
# --------------------------------

strategy_configs = {

    "SMA Crossover": {

        "class": SMACrossoverStrategy,

        "parameters": {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100]
        }
    },

    "Mean Reversion": {

        "class": MeanReversionStrategy,

        "parameters": {
            "period": [10, 20, 30],
            "std": [1.5, 2.0, 2.5]
        }
    }
}


# --------------------------------
# Run strategies independently
# --------------------------------

equity_curves = {}


for name, config in strategy_configs.items():

    print()
    print("=" * 60)
    print(
        f"RUNNING: {name}"
    )
    print("=" * 60)

    runner = WalkForwardRunner(
        data=data,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        initial_cash=INITIAL_CAPITAL
    )

    results, equity = runner.run(
        strategy_class=config["class"],
        parameters=config["parameters"]
    )

    equity_curves[name] = equity.copy()


# --------------------------------
# Align OOS equity curves
# --------------------------------

equity_dataframe = pd.concat(
    equity_curves,
    axis=1
).dropna()


# --------------------------------
# Convert to daily returns
# --------------------------------

strategy_returns = (
    equity_dataframe
    .pct_change(
        fill_method=None
    )
    .fillna(0.0)
)


# --------------------------------
# 50/50 portfolio
# --------------------------------

weights = {
    "SMA Crossover": 0.50,
    "Mean Reversion": 0.50
}


portfolio_returns = (

    strategy_returns[
        "SMA Crossover"
    ] * weights[
        "SMA Crossover"
    ]

    +

    strategy_returns[
        "Mean Reversion"
    ] * weights[
        "Mean Reversion"
    ]
)


# --------------------------------
# Build portfolio equity curve
# --------------------------------

portfolio_equity = (

    INITIAL_CAPITAL

    * (
        1 + portfolio_returns
    ).cumprod()
)

portfolio_equity.name = (
    "50/50 Portfolio"
)


# --------------------------------
# Individual strategy metrics
# --------------------------------

def calculate_metrics(
    name,
    equity
):

    final_capital = float(
        equity.iloc[-1]
    )

    return {

        "Strategy": name,

        "Final Capital":
            final_capital,

        "Return":
            Metrics.total_return(
                INITIAL_CAPITAL,
                final_capital
            ),

        "CAGR":
            Metrics.cagr(
                INITIAL_CAPITAL,
                final_capital,
                len(equity)
            ),

        "Volatility":
            Metrics.volatility(
                equity
            ),

        "Sharpe":
            Metrics.sharpe_ratio(
                equity
            ),

        "Sortino":
            Metrics.sortino_ratio(
                equity
            ),

        "Max Drawdown":
            Metrics.max_drawdown(
                equity
            )
    }


comparison = []


comparison.append(
    calculate_metrics(
        "SMA Crossover",
        equity_dataframe[
            "SMA Crossover"
        ]
    )
)

comparison.append(
    calculate_metrics(
        "Mean Reversion",
        equity_dataframe[
            "Mean Reversion"
        ]
    )
)

comparison.append(
    calculate_metrics(
        "50/50 Portfolio",
        portfolio_equity
    )
)


# --------------------------------
# Buy-and-hold benchmark
# --------------------------------

benchmark_prices = data.loc[
    equity_dataframe.index,
    "Close"
]


# yfinance may return Close
# as a one-column DataFrame.
if hasattr(
    benchmark_prices,
    "columns"
):

    if len(
        benchmark_prices.columns
    ) == 1:

        benchmark_prices = (
            benchmark_prices.iloc[:, 0]
        )


benchmark_prices = (
    benchmark_prices
    .dropna()
    .astype(float)
)


benchmark_equity = (

    benchmark_prices

    / benchmark_prices.iloc[0]

    * INITIAL_CAPITAL
)


comparison.append(
    calculate_metrics(
        "Buy & Hold",
        benchmark_equity
    )
)


# --------------------------------
# Print results
# --------------------------------

print()
print("=" * 110)
print(
    "MULTI-STRATEGY PORTFOLIO COMPARISON"
)
print("=" * 110)

print()

print(
    f"{'Strategy':<22}"
    f"{'Final Capital':>16}"
    f"{'Return':>12}"
    f"{'CAGR':>10}"
    f"{'Vol':>10}"
    f"{'Sharpe':>10}"
    f"{'Sortino':>10}"
    f"{'Max DD':>10}"
)

print("-" * 100)


for result in comparison:

    print(
        f"{result['Strategy']:<22}"
        f"${result['Final Capital']:>15,.2f}"
        f"{result['Return']:>11.2f}%"
        f"{result['CAGR']:>9.2f}%"
        f"{result['Volatility']:>9.2f}%"
        f"{result['Sharpe']:>10.2f}"
        f"{result['Sortino']:>10.2f}"
        f"{result['Max Drawdown']:>9.2f}%"
    )


print()
print("=" * 110)