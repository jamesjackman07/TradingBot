import pandas as pd

from bot.data import MarketData

from bot.strategies.sma_cross import (
    SMACrossoverStrategy
)

from bot.strategies.mean_reversion import (
    MeanReversionStrategy
)

from walkforward.run import WalkForwardRunner


# --------------------------------
# Settings
# --------------------------------

INITIAL_CAPITAL = 10000

TRAIN_SIZE = 504
TEST_SIZE = 126
STEP_SIZE = 126

PERIOD = "15y"


# --------------------------------
# Systems to compare
#
# Each system is one asset +
# one strategy.
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

    "GLD SMA": {
        "ticker": "GLD",
        "class": SMACrossoverStrategy,
        "parameters": {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100]
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
# Download each asset once
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

    ticker = config["ticker"]

    print()
    print("=" * 70)
    print(
        f"RUNNING: {system_name}"
    )
    print("=" * 70)

    runner = WalkForwardRunner(
        data=asset_data[ticker],
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

        print(
            f"No OOS equity generated "
            f"for {system_name}."
        )

        continue

    equity_curves[
        system_name
    ] = equity.copy()


# --------------------------------
# Align all systems by DATE
# --------------------------------

equity_dataframe = pd.concat(
    equity_curves,
    axis=1,
    join="inner"
).dropna()


# --------------------------------
# Convert equity to daily returns
# --------------------------------

daily_returns = (
    equity_dataframe
    .pct_change(
        fill_method=None
    )
    .dropna()
)


# --------------------------------
# Correlation matrix
# --------------------------------

correlation = (
    daily_returns.corr()
)


# --------------------------------
# Print correlation matrix
# --------------------------------

print()
print("=" * 90)
print(
    "MULTI-ASSET DAILY OOS RETURN CORRELATION"
)
print("=" * 90)

print()

print(
    correlation.round(3)
)

print()
print("=" * 90)


# --------------------------------
# Additional diversification stats
# --------------------------------

print()
print("=" * 90)
print(
    "PAIRWISE CORRELATIONS"
)
print("=" * 90)

print()

system_names = list(
    correlation.columns
)


pairs = []


for i in range(
    len(system_names)
):

    for j in range(
        i + 1,
        len(system_names)
    ):

        first = system_names[i]
        second = system_names[j]

        value = correlation.loc[
            first,
            second
        ]

        pairs.append(
            (
                first,
                second,
                value
            )
        )


# Lowest correlation first
pairs.sort(
    key=lambda pair: pair[2]
)


for first, second, value in pairs:

    print(
        f"{first:<18}"
        f" vs "
        f"{second:<18}"
        f" : "
        f"{value:.3f}"
    )


print()
print("=" * 90)