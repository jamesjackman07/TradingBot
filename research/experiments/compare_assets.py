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
# Assets
# --------------------------------

ASSETS = {

    "SPY": "US Equities",

    "GLD": "Gold",

    "TLT": "US Treasuries"
}


# --------------------------------
# Strategies
# --------------------------------

STRATEGIES = {

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
# Metric helper
# --------------------------------

def calculate_metrics(
    equity
):

    final_capital = float(
        equity.iloc[-1]
    )

    return {

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
# Run research
# --------------------------------

market = MarketData()

comparison = []


for ticker, asset_name in ASSETS.items():

    print()
    print("=" * 70)

    print(
        f"ASSET: {ticker} "
        f"({asset_name})"
    )

    print("=" * 70)

    # --------------------------------
    # Download this asset
    # --------------------------------

    data = market.get_data(
        ticker,
        period=PERIOD
    )

    # --------------------------------
    # Run each strategy
    # --------------------------------

    for strategy_name, config in (
        STRATEGIES.items()
    ):

        print()

        print(
            f"Running "
            f"{strategy_name}..."
        )

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
                "No OOS equity generated."
            )

            continue

        metrics = calculate_metrics(
            equity
        )

        total_trades = sum(
            result["testing_trades"]
            for result in results
        )

        comparison.append({

            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "Strategy":
                strategy_name,

            "Windows":
                len(results),

            "Trades":
                total_trades,

            **metrics
        })


# --------------------------------
# Print results
# --------------------------------

print()
print("=" * 140)
print(
    "MULTI-ASSET STRATEGY COMPARISON"
)
print("=" * 140)

print()

print(
    f"{'Ticker':<8}"
    f"{'Asset':<18}"
    f"{'Strategy':<18}"
    f"{'Final Capital':>16}"
    f"{'Return':>11}"
    f"{'CAGR':>9}"
    f"{'Vol':>9}"
    f"{'Sharpe':>9}"
    f"{'Sortino':>9}"
    f"{'Max DD':>10}"
    f"{'Trades':>9}"
    f"{'Windows':>9}"
)

print("-" * 135)


for result in comparison:

    print(
        f"{result['Ticker']:<8}"
        f"{result['Asset']:<18}"
        f"{result['Strategy']:<18}"
        f"${result['Final Capital']:>15,.2f}"
        f"{result['Return']:>10.2f}%"
        f"{result['CAGR']:>8.2f}%"
        f"{result['Volatility']:>8.2f}%"
        f"{result['Sharpe']:>9.2f}"
        f"{result['Sortino']:>9.2f}"
        f"{result['Max Drawdown']:>9.2f}%"
        f"{result['Trades']:>9}"
        f"{result['Windows']:>9}"
    )


print()
print("=" * 140)