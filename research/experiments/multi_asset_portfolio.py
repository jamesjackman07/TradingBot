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

PERIOD = "15y"


# --------------------------------
# Systems
# --------------------------------

SYSTEMS = {

    "SPY SMA": {

        "ticker": "SPY",

        "class": SMACrossoverStrategy,

        "parameters": {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100]
        }
    },

    "SPY Mean Rev": {

        "ticker": "SPY",

        "class": MeanReversionStrategy,

        "parameters": {
            "period": [10, 20, 30],
            "std": [1.5, 2.0, 2.5]
        }
    },

    "GLD Mean Rev": {

        "ticker": "GLD",

        "class": MeanReversionStrategy,

        "parameters": {
            "period": [10, 20, 30],
            "std": [1.5, 2.0, 2.5]
        }
    }
}


# --------------------------------
# Download assets once
# --------------------------------

market = MarketData()

asset_data = {}


for ticker in {
    config["ticker"]
    for config in SYSTEMS.values()
}:

    print(
        f"Downloading {ticker}..."
    )

    asset_data[ticker] = market.get_data(
        ticker,
        period=PERIOD
    )


# --------------------------------
# Run each system
# --------------------------------

equity_curves = {}


for system_name, config in (
    SYSTEMS.items()
):

    print()
    print("=" * 70)

    print(
        f"RUNNING: {system_name}"
    )

    print("=" * 70)

    runner = WalkForwardRunner(
        data=asset_data[
            config["ticker"]
        ],
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        initial_cash=INITIAL_CAPITAL
    )

    results, equity = runner.run(
        strategy_class=config["class"],
        parameters=config["parameters"]
    )

    if equity.empty:

        raise ValueError(
            f"No OOS equity generated "
            f"for {system_name}"
        )

    equity_curves[
        system_name
    ] = equity.copy()


# --------------------------------
# Align all systems by date
# --------------------------------

equity_dataframe = pd.concat(
    equity_curves,
    axis=1,
    join="inner"
).dropna()


# --------------------------------
# Normalize each strategy
#
# A value of 1.0 means the original
# starting capital.
# --------------------------------

growth = (
    equity_dataframe
    / equity_dataframe.iloc[0]
)


# ================================================================
# PORTFOLIO 1
#
# Existing SPY-only 50/50 portfolio
# ================================================================

spy_sma_sleeve = (
    INITIAL_CAPITAL
    * 0.50
    * growth["SPY SMA"]
)

spy_mean_reversion_sleeve = (
    INITIAL_CAPITAL
    * 0.50
    * growth["SPY Mean Rev"]
)


spy_portfolio = (
    spy_sma_sleeve
    + spy_mean_reversion_sleeve
)

spy_portfolio.name = (
    "SPY 50/50"
)


# ================================================================
# PORTFOLIO 2
#
# Multi-asset equal-weight portfolio
#
# 1/3 SPY SMA
# 1/3 SPY Mean Reversion
# 1/3 GLD Mean Reversion
# ================================================================

weight = 1 / 3


multi_spy_sma_sleeve = (
    INITIAL_CAPITAL
    * weight
    * growth["SPY SMA"]
)

multi_spy_mean_reversion_sleeve = (
    INITIAL_CAPITAL
    * weight
    * growth["SPY Mean Rev"]
)

multi_gld_mean_reversion_sleeve = (
    INITIAL_CAPITAL
    * weight
    * growth["GLD Mean Rev"]
)


multi_asset_portfolio = (
    multi_spy_sma_sleeve
    + multi_spy_mean_reversion_sleeve
    + multi_gld_mean_reversion_sleeve
)

multi_asset_portfolio.name = (
    "SPY + GLD Equal Weight"
)


# --------------------------------
# Metric helper
# --------------------------------

def calculate_metrics(
    name,
    equity
):

    final_capital = float(
        equity.iloc[-1]
    )

    return {

        "Strategy":
            name,

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


# --------------------------------
# Results
# --------------------------------

comparison = []


# Individual systems
for system_name in SYSTEMS:

    comparison.append(
        calculate_metrics(
            system_name,
            equity_dataframe[
                system_name
            ]
        )
    )


# SPY-only portfolio
comparison.append(
    calculate_metrics(
        "SPY 50/50",
        spy_portfolio
    )
)


# Multi-asset portfolio
comparison.append(
    calculate_metrics(
        "SPY + GLD Equal Weight",
        multi_asset_portfolio
    )
)


# --------------------------------
# SPY buy-and-hold benchmark
# --------------------------------

benchmark_prices = asset_data[
    "SPY"
].loc[
    equity_dataframe.index,
    "Close"
]


# yfinance may return Close as
# a one-column DataFrame.
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
        "SPY Buy & Hold",
        benchmark_equity
    )
)


# --------------------------------
# Print comparison
# --------------------------------

print()
print("=" * 120)

print(
    "MULTI-ASSET PORTFOLIO COMPARISON"
)

print("=" * 120)

print()

print(
    f"{'System / Portfolio':<26}"
    f"{'Final Capital':>16}"
    f"{'Return':>12}"
    f"{'CAGR':>10}"
    f"{'Vol':>10}"
    f"{'Sharpe':>10}"
    f"{'Sortino':>10}"
    f"{'Max DD':>10}"
)

print("-" * 114)


for result in comparison:

    print(
        f"{result['Strategy']:<26}"
        f"${result['Final Capital']:>15,.2f}"
        f"{result['Return']:>11.2f}%"
        f"{result['CAGR']:>9.2f}%"
        f"{result['Volatility']:>9.2f}%"
        f"{result['Sharpe']:>10.2f}"
        f"{result['Sortino']:>10.2f}"
        f"{result['Max Drawdown']:>9.2f}%"
    )


print()
print("=" * 120)


# --------------------------------
# Final portfolio allocations
# --------------------------------

print()
print("=" * 80)

print(
    "FINAL MULTI-ASSET PORTFOLIO ALLOCATION"
)

print("=" * 80)

print()


final_spy_sma = float(
    multi_spy_sma_sleeve.iloc[-1]
)

final_spy_mean_reversion = float(
    multi_spy_mean_reversion_sleeve.iloc[-1]
)

final_gld_mean_reversion = float(
    multi_gld_mean_reversion_sleeve.iloc[-1]
)

final_portfolio = float(
    multi_asset_portfolio.iloc[-1]
)


final_allocations = {

    "SPY SMA":
        final_spy_sma,

    "SPY Mean Rev":
        final_spy_mean_reversion,

    "GLD Mean Rev":
        final_gld_mean_reversion
}


for name, value in (
    final_allocations.items()
):

    final_weight = (
        value
        / final_portfolio
        * 100
    )

    print(
        f"{name:<18}"
        f": ${value:>10,.2f}"
        f" ({final_weight:>6.2f}%)"
    )


print()

print(
    f"{'Total Portfolio':<18}"
    f": ${final_portfolio:>10,.2f}"
)

print()

print("=" * 80)