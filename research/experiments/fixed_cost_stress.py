import pandas as pd

from bot.data import MarketData
from bot.research import ResearchSession

from bot.strategies.sma_cross import (
    SMACrossoverStrategy
)

from bot.strategies.mean_reversion import (
    MeanReversionStrategy
)

from optimization.optimizer import Optimizer

from walkforward.tester import WalkForwardTester

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

COST_SCENARIOS = {

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
# Download data once
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="15y"
)


# --------------------------------
# Walk-forward structure
#
# IMPORTANT:
# Training optimisation uses
# ZERO transaction costs.
# --------------------------------

zero_cost_manager = RiskManager(
    commission=0,
    slippage=0
)

tester = WalkForwardTester(
    data=data,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    step_size=STEP_SIZE,
    risk_manager=zero_cost_manager
)


# --------------------------------
# Storage
# --------------------------------

# Selected parameters are generated
# once and reused by every scenario.

selected_windows = {
    name: []
    for name in STRATEGIES
}


# --------------------------------
# Helper:
# convert numpy/pandas values
# into normal Python numbers
# --------------------------------

def clean_parameter(value):

    try:

        numeric_value = float(
            value
        )

        if numeric_value.is_integer():

            return int(
                numeric_value
            )

        return numeric_value

    except (
        TypeError,
        ValueError
    ):

        return value


# --------------------------------
# STEP 1
#
# Optimise every training window
# ONCE using zero costs.
# --------------------------------

print()
print("=" * 80)
print(
    "SELECTING PARAMETERS UNDER ZERO COSTS"
)
print("=" * 80)


windows = list(
    tester.windows()
)


for strategy_name, config in (
    STRATEGIES.items()
):

    print()
    print(
        f"Optimising {strategy_name}..."
    )

    strategy_class = config[
        "class"
    ]

    parameters = config[
        "parameters"
    ]

    constraint = getattr(
        strategy_class,
        "parameter_constraint",
        None
    )

    for window_number, window in enumerate(
        windows,
        start=1
    ):

        train_session = (
            tester.create_train_session(
                window
            )
        )

        optimizer = Optimizer(
            train_session
        )

        results = optimizer.optimize(
            strategy_class=strategy_class,
            parameters=parameters,
            constraint=constraint
        )

        best = (
            results.dataframe.iloc[0]
        )

        selected_parameters = {}

        for parameter_name in parameters:

            selected_parameters[
                parameter_name
            ] = clean_parameter(
                best[
                    parameter_name
                ]
            )

        selected_windows[
            strategy_name
        ].append(
            selected_parameters
        )

    print(
        f"Selected parameters for "
        f"{len(windows)} windows."
    )


# --------------------------------
# Helper:
# replay one strategy using the
# already-selected parameters.
# --------------------------------

def run_fixed_strategy(
    strategy_name,
    config,
    risk_manager
):

    strategy_class = config[
        "class"
    ]

    parameter_windows = (
        selected_windows[
            strategy_name
        ]
    )

    current_cash = (
        INITIAL_CAPITAL
    )

    equity_parts = []

    total_trades = 0

    for window_index, window in enumerate(
        windows
    ):

        selected_parameters = (
            parameter_windows[
                window_index
            ]
        )

        # --------------------------------
        # Frozen strategy
        # --------------------------------

        strategy = strategy_class(
            **selected_parameters
        )

        # --------------------------------
        # Determine required warm-up
        # --------------------------------

        if hasattr(
            strategy,
            "warmup_period"
        ):

            warmup_size = int(
                strategy.warmup_period()
            )

        else:

            warmup_size = 0

        # --------------------------------
        # Historical warm-up + test data
        # --------------------------------

        warmup_data = (
            tester.get_test_with_warmup(
                window,
                warmup_size
            )
        )

        warmup_session = (
            ResearchSession(
                data=warmup_data
            )
        )

        all_signals = (
            strategy.generate_signals(
                warmup_session.close
            )
        )

        # --------------------------------
        # Create OOS session using the
        # CURRENT cost scenario.
        # --------------------------------

        test_session = (
            ResearchSession(
                data=window[
                    "test_data"
                ],
                risk_manager=risk_manager
            )
        )

        test_length = len(
            test_session.close
        )

        test_signals = all_signals[
            -test_length:
        ]

        # --------------------------------
        # Carry capital forward
        # --------------------------------

        test_session.engine.initial_cash = (
            current_cash
        )

        # --------------------------------
        # Run exact same frozen strategy
        # under this cost scenario.
        # --------------------------------

        equity, trades = (
            test_session.engine.run(
                test_session.close,
                test_signals
            )
        )

        current_cash = float(
            equity.iloc[-1]
        )

        total_trades += len(
            trades
        )

        # --------------------------------
        # Restore real dates
        # --------------------------------

        dated_equity = equity.copy()

        dated_equity.index = (
            test_session.close.index
        )

        equity_parts.append(
            dated_equity
        )

    # --------------------------------
    # Combined OOS curve
    # --------------------------------

    combined_equity = pd.concat(
        equity_parts
    )

    combined_equity.name = (
        strategy_name
    )

    return (
        combined_equity,
        total_trades
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
# STEP 2
#
# Replay those SAME parameter
# selections under every cost level.
# --------------------------------

stress_results = []


for scenario_name, costs in (
    COST_SCENARIOS.items()
):

    print()
    print("=" * 80)

    print(
        f"REPLAYING: {scenario_name}"
    )

    print("=" * 80)

    print(
        f"Commission : "
        f"${costs['commission']:.2f}"
    )

    print(
        f"Slippage   : "
        f"{costs['slippage'] * 100:.3f}%"
    )

    risk_manager = RiskManager(
        commission=costs[
            "commission"
        ],
        slippage=costs[
            "slippage"
        ]
    )

    scenario_equity = {}

    scenario_trades = {}

    # --------------------------------
    # Replay each strategy
    # --------------------------------

    for strategy_name, config in (
        STRATEGIES.items()
    ):

        print(
            f"Running {strategy_name}..."
        )

        equity, trades = (
            run_fixed_strategy(
                strategy_name,
                config,
                risk_manager
            )
        )

        scenario_equity[
            strategy_name
        ] = equity

        scenario_trades[
            strategy_name
        ] = trades

    # --------------------------------
    # Align both strategy curves
    # --------------------------------

    equity_dataframe = pd.concat(
        scenario_equity,
        axis=1
    ).dropna()

    # --------------------------------
    # Non-rebalanced 50/50 portfolio
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
    # Metrics
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

        "Trades":
            scenario_trades[
                "SMA Crossover"
            ]
            + scenario_trades[
                "Mean Reversion"
            ],

        **metrics
    })


# --------------------------------
# Print selected parameter check
# --------------------------------

print()
print("=" * 80)
print(
    "FIXED PARAMETER CHECK"
)
print("=" * 80)

print()

print(
    "Parameters below were selected "
    "once under zero costs."
)

print(
    "Every cost scenario replayed "
    "these exact selections."
)

print()


for strategy_name, parameter_windows in (
    selected_windows.items()
):

    print(strategy_name)

    for index, parameters in enumerate(
        parameter_windows,
        start=1
    ):

        print(
            f"  Window {index:>2}: "
            f"{parameters}"
        )

    print()


# --------------------------------
# Final stress-test table
# --------------------------------

print()
print("=" * 130)
print(
    "FIXED-PARAMETER TRANSACTION COST STRESS TEST"
)
print("=" * 130)

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
    f"{'Trades':>9}"
)

print("-" * 125)


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
        f"{result['Trades']:>9}"
    )


print()
print("=" * 130)