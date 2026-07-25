from bot.data import MarketData

from bot.strategies.sma_cross import (
    SMACrossoverStrategy
)

from bot.strategies.momentum import (
    MomentumStrategy
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
# Download data once
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="15y"
)


# --------------------------------
# Strategies to compare
# --------------------------------

strategies = {

    "SMA Crossover": {

        "class": SMACrossoverStrategy,

        "parameters": {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100]
        }
    },

    "Momentum": {

        "class": MomentumStrategy,

        "parameters": {
            "lookback": [
                20,
                60,
                120,
                252
            ]
        }
    },

    "Mean Reversion": {

        "class": MeanReversionStrategy,

        "parameters": {
            "period": [
                10,
                20,
                30
            ],

            "std": [
                1.5,
                2.0,
                2.5
            ]
        }
    }
}

# --------------------------------
# Run each strategy
# --------------------------------

comparison = []


for name, config in strategies.items():

    print()
    print("=" * 60)
    print(
        f"TESTING: {name}"
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

    if equity.empty:

        print(
            "No equity data generated."
        )

        continue

    # --------------------------------
    # Performance metrics
    # --------------------------------

    final_capital = float(
        equity.iloc[-1]
    )

    total_return = Metrics.total_return(
        INITIAL_CAPITAL,
        final_capital
    )

    cagr = Metrics.cagr(
        INITIAL_CAPITAL,
        final_capital,
        len(equity)
    )

    volatility = Metrics.volatility(
        equity
    )

    sharpe = Metrics.sharpe_ratio(
        equity
    )

    sortino = Metrics.sortino_ratio(
        equity
    )

    drawdown = Metrics.max_drawdown(
        equity
    )

    trades = sum(
        result["testing_trades"]
        for result in results
    )

    profitable_windows = sum(
        result["testing_return"] > 0
        for result in results
    )

    losing_windows = sum(
        result["testing_return"] < 0
        for result in results
    )

    flat_windows = sum(
        result["testing_return"] == 0
        for result in results
    )

    comparison.append({

        "Strategy": name,

        "Final Capital": final_capital,

        "Return": total_return,

        "CAGR": cagr,

        "Volatility": volatility,

        "Sharpe": sharpe,

        "Sortino": sortino,

        "Max Drawdown": drawdown,

        "Trades": trades,

        "Profitable Windows":
            profitable_windows,

        "Losing Windows":
            losing_windows,

        "Flat Windows":
            flat_windows
    })


# --------------------------------
# Buy-and-hold benchmark
# --------------------------------

if comparison:

    first_test_date = data.index[
        TRAIN_SIZE
    ]

    number_of_windows = (
        len(data) - TRAIN_SIZE - TEST_SIZE
    ) // STEP_SIZE + 1

    oos_length = (
        number_of_windows
        * TEST_SIZE
    )

    benchmark_prices = data.loc[
        first_test_date:
    ]["Close"].iloc[
        :oos_length
    ]

    # yfinance may return a
    # one-column DataFrame.
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

    benchmark_final = float(
        benchmark_equity.iloc[-1]
    )

    comparison.append({

        "Strategy": "Buy & Hold",

        "Final Capital":
            benchmark_final,

        "Return":
            Metrics.total_return(
                INITIAL_CAPITAL,
                benchmark_final
            ),

        "CAGR":
            Metrics.cagr(
                INITIAL_CAPITAL,
                benchmark_final,
                len(benchmark_equity)
            ),

        "Volatility":
            Metrics.volatility(
                benchmark_equity
            ),

        "Sharpe":
            Metrics.sharpe_ratio(
                benchmark_equity
            ),

        "Sortino":
            Metrics.sortino_ratio(
                benchmark_equity
            ),

        "Max Drawdown":
            Metrics.max_drawdown(
                benchmark_equity
            ),

        "Trades": 0,

        "Profitable Windows": 0,

        "Losing Windows": 0,

        "Flat Windows": 0
    })


# --------------------------------
# Print comparison table
# --------------------------------

print()
print("=" * 110)
print("STRATEGY COMPARISON")
print("=" * 110)

print()

print(
    f"{'Strategy':<20}"
    f"{'Final Capital':>15}"
    f"{'Return':>12}"
    f"{'CAGR':>10}"
    f"{'Vol':>10}"
    f"{'Sharpe':>10}"
    f"{'Sortino':>10}"
    f"{'Max DD':>10}"
    f"{'Trades':>9}"
)

print("-" * 106)


for result in comparison:

    print(
        f"{result['Strategy']:<20}"
        f"${result['Final Capital']:>14,.2f}"
        f"{result['Return']:>11.2f}%"
        f"{result['CAGR']:>9.2f}%"
        f"{result['Volatility']:>9.2f}%"
        f"{result['Sharpe']:>10.2f}"
        f"{result['Sortino']:>10.2f}"
        f"{result['Max Drawdown']:>9.2f}%"
        f"{result['Trades']:>9}"
    )


print()
print("=" * 110)