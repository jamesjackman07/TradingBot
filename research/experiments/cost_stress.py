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

from risk.manager import RiskManager


# --------------------------------
# Settings
# --------------------------------

INITIAL_CAPITAL = 10000

TRAIN_SIZE = 504
TEST_SIZE = 126
STEP_SIZE = 126


# --------------------------------
# Cost scenarios
# --------------------------------

cost_scenarios = {

    "Zero Costs": {
        "commission": 0,
        "slippage": 0
    },

    "Low Costs": {
        "commission": 1,
        "slippage": 0.0002
    },

    "Moderate Costs": {
        "commission": 2,
        "slippage": 0.0005
    },

    "High Costs": {
        "commission": 5,
        "slippage": 0.001
    },

    "Stress Test": {
        "commission": 10,
        "slippage": 0.0025
    }
}


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
# Download data once
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="15y"
)


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
# Run cost scenarios
# --------------------------------

stress_results = []


for scenario_name, costs in (
    cost_scenarios.items()
):

    print()
    print("=" * 70)
    print(
        f"RUNNING COST SCENARIO: "
        f"{scenario_name}"
    )
    print("=" * 70)

    print(
        f"Commission : "
        f"${costs['commission']:.2f}"
    )

    print(
        f"Slippage   : "
        f"{costs['slippage'] * 100:.3f}%"
    )

    # --------------------------------
    # Risk manager for this scenario
    # --------------------------------

    risk_manager = RiskManager(
        commission=costs["commission"],
        slippage=costs["slippage"]
    )

    # --------------------------------
    # Run both strategies
    # --------------------------------

    equity_curves = {}

    for name, config in (
        strategy_configs.items()
    ):

        print(
            f"Running {name}..."
        )

        runner = WalkForwardRunner(
            data=data,
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
            step_size=STEP_SIZE,
            initial_cash=INITIAL_CAPITAL,
            risk_manager=risk_manager
        )

        results, equity = runner.run(
            strategy_class=config["class"],
            parameters=config["parameters"]
        )

        equity_curves[name] = (
            equity.copy()
        )

    # --------------------------------
    # Align curves
    # --------------------------------

    equity_dataframe = pd.concat(
        equity_curves,
        axis=1
    ).dropna()

    # --------------------------------
    # Build non-rebalanced portfolio
    # --------------------------------

    sma_growth = (
        equity_dataframe[
            "SMA Crossover"
        ]
        / equity_dataframe[
            "SMA Crossover"
        ].iloc[0]
    )

    mean_reversion_growth = (
        equity_dataframe[
            "Mean Reversion"
        ]
        / equity_dataframe[
            "Mean Reversion"
        ].iloc[0]
    )

    sma_sleeve = (
        INITIAL_CAPITAL
        * 0.50
        * sma_growth
    )

    mean_reversion_sleeve = (
        INITIAL_CAPITAL
        * 0.50
        * mean_reversion_growth
    )

    portfolio_equity = (
        sma_sleeve
        + mean_reversion_sleeve
    )

    portfolio_equity.name = (
        scenario_name
    )

    # --------------------------------
    # Calculate metrics
    # --------------------------------

    metrics = calculate_metrics(
        portfolio_equity
    )

    stress_results.append({

        "Scenario":
            scenario_name,

        "Commission":
            costs["commission"],

        "Slippage":
            costs["slippage"] * 100,

        **metrics
    })


# --------------------------------
# Print final table
# --------------------------------

print()
print("=" * 120)
print(
    "TRANSACTION COST STRESS TEST"
)
print("=" * 120)

print()

print(
    f"{'Scenario':<18}"
    f"{'Commission':>12}"
    f"{'Slippage':>12}"
    f"{'Final Capital':>16}"
    f"{'Return':>11}"
    f"{'CAGR':>9}"
    f"{'Vol':>9}"
    f"{'Sharpe':>9}"
    f"{'Sortino':>9}"
    f"{'Max DD':>10}"
)

print("-" * 115)


for result in stress_results:

    print(
        f"{result['Scenario']:<18}"
        f"${result['Commission']:>10.2f}"
        f"{result['Slippage']:>11.3f}%"
        f"${result['Final Capital']:>15,.2f}"
        f"{result['Return']:>10.2f}%"
        f"{result['CAGR']:>8.2f}%"
        f"{result['Volatility']:>8.2f}%"
        f"{result['Sharpe']:>9.2f}"
        f"{result['Sortino']:>9.2f}"
        f"{result['Max Drawdown']:>9.2f}%"
    )


print()
print("=" * 120)