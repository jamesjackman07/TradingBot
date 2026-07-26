import pandas as pd
import pytest

from walkforward.run import WalkForwardRunner
from risk.manager import RiskManager


def make_data(rows=30):

    index = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="D"
    )

    return pd.DataFrame(
        {
            "Close": [
                100 + i
                for i in range(rows)
            ]
        },
        index=index
    )


class AlwaysBuyStrategy:

    def __init__(self, value=1):
        self.value = value

    def warmup_period(self):
        return 0

    def generate_signals(self, close):

        signals = [
            "HOLD"
            for _ in range(len(close))
        ]

        if signals:
            signals[0] = "BUY"

        return signals


class WarmupStrategy:

    def __init__(self, value=1):
        self.value = value

    def warmup_period(self):
        return 3

    def generate_signals(self, close):

        return [
            "HOLD"
            for _ in range(len(close))
        ]


class ConstrainedStrategy:

    def __init__(self, fast, slow):
        self.fast = fast
        self.slow = slow

    @staticmethod
    def parameter_constraint(parameters):

        return (
            parameters["fast"]
            < parameters["slow"]
        )

    def warmup_period(self):
        return 0

    def generate_signals(self, close):

        return [
            "HOLD"
            for _ in range(len(close))
        ]


# ==================================================
# Construction
# ==================================================

def test_runner_stores_initial_cash():

    data = make_data()

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        initial_cash=25000
    )

    assert runner.initial_cash == 25000


def test_runner_passes_risk_manager_to_tester():

    data = make_data()

    risk = RiskManager(
        commission=5
    )

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        risk_manager=risk
    )

    assert (
        runner.tester.risk_manager
        is risk
    )


# ==================================================
# Basic output
# ==================================================

def test_runner_returns_results_and_equity():

    data = make_data()

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert isinstance(
        results,
        list
    )

    assert isinstance(
        equity,
        pd.Series
    )


def test_combined_equity_is_named_equity():

    data = make_data()

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    _, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert equity.name == "Equity"


# ==================================================
# Windows
# ==================================================

def test_runner_returns_expected_number_of_windows():

    data = make_data(25)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert len(results) == 3


def test_window_numbers_start_at_one():

    data = make_data(25)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert [
        result["window"]
        for result in results
    ] == [
        1,
        2,
        3
    ]


def test_window_dates_are_stored_correctly():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    first = results[0]

    assert first["train_start"] == (
        data.index[0]
    )

    assert first["train_end"] == (
        data.index[9]
    )

    assert first["test_start"] == (
        data.index[10]
    )

    assert first["test_end"] == (
        data.index[14]
    )


# ==================================================
# Parameter selection
# ==================================================

def test_selected_parameters_are_stored():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [7]
        }
    )

    assert results[0][
        "parameters"
    ] == {
        "value": 7
    }

    assert results[0][
        "value"
    ] == 7


def test_integer_parameters_remain_integers():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [7.0]
        }
    )

    value = results[0][
        "parameters"
    ][
        "value"
    ]

    assert value == 7
    assert isinstance(
        value,
        int
    )


def test_float_parameters_remain_floats():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1.5]
        }
    )

    value = results[0][
        "parameters"
    ][
        "value"
    ]

    assert value == pytest.approx(
        1.5
    )

    assert isinstance(
        value,
        float
    )


def test_strategy_parameter_constraint_is_used():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=ConstrainedStrategy,
        parameters={
            "fast": [5, 30],
            "slow": [20]
        }
    )

    for result in results:

        assert (
            result["fast"]
            <
            result["slow"]
        )


# ==================================================
# Warm-up
# ==================================================

def test_warmup_period_is_stored():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=WarmupStrategy,
        parameters={
            "value": [1]
        }
    )

    assert results[0][
        "warmup_bars"
    ] == 3


def test_strategy_without_warmup_method_uses_zero():

    class NoWarmupStrategy:

        def __init__(self, value=1):
            self.value = value

        def generate_signals(
            self,
            close
        ):

            return [
                "HOLD"
                for _ in range(
                    len(close)
                )
            ]

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=NoWarmupStrategy,
        parameters={
            "value": [1]
        }
    )

    assert results[0][
        "warmup_bars"
    ] == 0


# ==================================================
# Capital carry-forward
# ==================================================

def test_first_window_starts_with_initial_cash():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        initial_cash=10000
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert results[0][
        "starting_capital"
    ] == pytest.approx(
        10000
    )


def test_capital_is_carried_to_next_window():

    data = make_data(25)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        initial_cash=10000
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    for i in range(
        1,
        len(results)
    ):

        assert results[i][
            "starting_capital"
        ] == pytest.approx(
            results[i - 1][
                "ending_capital"
            ]
        )


def test_final_equity_matches_final_window_capital():

    data = make_data(25)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        initial_cash=10000
    )

    results, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert equity.iloc[
        -1
    ] == pytest.approx(
        results[-1][
            "ending_capital"
        ]
    )


# ==================================================
# Combined OOS equity
# ==================================================

def test_combined_equity_has_expected_length():

    data = make_data(25)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    expected_length = (
        len(results) * 5
    )

    assert len(equity) == (
        expected_length
    )


def test_combined_equity_uses_test_dates():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    _, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    expected_index = (
        data.index[10:20]
    )

    assert equity.index.equals(
        expected_index
    )


def test_training_dates_are_not_in_combined_equity():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    _, equity = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    initial_training_dates = set(
        data.index[:10]
    )

    equity_dates = set(
        equity.index
    )

    assert initial_training_dates.isdisjoint(
        equity_dates
    )


# ==================================================
# Stored metrics
# ==================================================

def test_training_metrics_are_stored():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    result = results[0]

    assert "training_return" in result
    assert "training_sharpe" in result
    assert "training_drawdown" in result
    assert "training_trades" in result


def test_testing_metrics_are_stored():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    result = results[0]

    assert "testing_return" in result
    assert "testing_sharpe" in result
    assert "testing_drawdown" in result
    assert "testing_trades" in result


def test_trade_counts_are_integers():

    data = make_data(20)

    runner = WalkForwardRunner(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    results, _ = runner.run(
        strategy_class=AlwaysBuyStrategy,
        parameters={
            "value": [1]
        }
    )

    assert isinstance(
        results[0][
            "training_trades"
        ],
        int
    )

    assert isinstance(
        results[0][
            "testing_trades"
        ],
        int
    )


# ==================================================
# Risk manager
# ==================================================

def test_transaction_costs_affect_oos_results():

    data = make_data(20)

    zero_cost_runner = (
        WalkForwardRunner(
            data=data,
            train_size=10,
            test_size=5,
            step_size=5,
            initial_cash=10000,
            risk_manager=RiskManager(
                risk_percent=50,
                commission=0
            )
        )
    )

    cost_runner = (
        WalkForwardRunner(
            data=data,
            train_size=10,
            test_size=5,
            step_size=5,
            initial_cash=10000,
            risk_manager=RiskManager(
                risk_percent=50,
                commission=10
            )
        )
    )

    _, zero_cost_equity = (
        zero_cost_runner.run(
            strategy_class=AlwaysBuyStrategy,
            parameters={
                "value": [1]
            }
        )
    )

    _, cost_equity = (
        cost_runner.run(
            strategy_class=AlwaysBuyStrategy,
            parameters={
                "value": [1]
            }
        )
    )

    assert (
        cost_equity.iloc[-1]
        <
        zero_cost_equity.iloc[-1]
    )