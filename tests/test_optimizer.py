import pandas as pd
import pytest

from optimization.optimizer import Optimizer
from research.results import ResearchResults
from research.result import ResearchResult


# --------------------------------------------------
# Test helpers
# --------------------------------------------------

class DummyStrategy:

    def __init__(
        self,
        fast,
        slow
    ):
        self.fast = fast
        self.slow = slow


class DummySession:

    def __init__(self):
        self.strategies_run = []

    def run(self, strategy):

        self.strategies_run.append(
            strategy
        )

        # Make performance depend on the
        # supplied parameters so that we
        # can test sorting.
        final_equity = (
            10000
            + strategy.fast * 100
            + strategy.slow * 10
        )

        equity = pd.Series([
            10000,
            10100,
            final_equity
        ])

        trades = []

        return equity, trades


# ==================================================
# Optimizer
# ==================================================

def test_optimizer_returns_research_results():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5],
            "slow": [20]
        }
    )

    assert isinstance(
        results,
        ResearchResults
    )


def test_optimizer_tests_all_parameter_combinations():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5, 10],
            "slow": [20, 30]
        }
    )

    assert len(results) == 4

    assert len(
        session.strategies_run
    ) == 4


def test_optimizer_stores_parameters():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5],
            "slow": [20]
        }
    )

    dataframe = (
        results.to_dataframe()
    )

    assert dataframe.iloc[0][
        "fast"
    ] == 5

    assert dataframe.iloc[0][
        "slow"
    ] == 20


def test_optimizer_stores_metrics():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5],
            "slow": [20]
        }
    )

    dataframe = (
        results.to_dataframe()
    )

    expected_columns = {
        "Return",
        "Sharpe",
        "Drawdown",
        "Profit Factor",
        "Win Rate",
        "Trades"
    }

    assert expected_columns.issubset(
        dataframe.columns
    )


def test_optimizer_sorts_return_descending_by_default():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5, 10],
            "slow": [20]
        }
    )

    dataframe = (
        results.to_dataframe()
    )

    assert (
        dataframe.iloc[0]["Return"]
        >=
        dataframe.iloc[1]["Return"]
    )


def test_optimizer_can_sort_ascending():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5, 10],
            "slow": [20]
        },
        ascending=True
    )

    dataframe = (
        results.to_dataframe()
    )

    assert (
        dataframe.iloc[0]["Return"]
        <=
        dataframe.iloc[1]["Return"]
    )


def test_optimizer_can_sort_by_different_metric():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [5, 10],
            "slow": [20, 30]
        },
        sort_by="Sharpe"
    )

    dataframe = (
        results.to_dataframe()
    )

    assert dataframe[
        "Sharpe"
    ].is_monotonic_decreasing


def test_optimizer_rejects_unknown_metric():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    with pytest.raises(
        ValueError,
        match="Unknown metric"
    ):

        optimizer.optimize(
            strategy_class=DummyStrategy,
            parameters={
                "fast": [5],
                "slow": [20]
            },
            sort_by="Not A Metric"
        )


def test_optimizer_respects_constraint():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    results = optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [10, 30],
            "slow": [20]
        },
        constraint=lambda p: (
            p["fast"] < p["slow"]
        )
    )

    dataframe = (
        results.to_dataframe()
    )

    assert len(dataframe) == 1

    assert dataframe.iloc[0][
        "fast"
    ] == 10

    assert dataframe.iloc[0][
        "slow"
    ] == 20


def test_optimizer_only_runs_valid_parameter_combinations():

    session = DummySession()

    optimizer = Optimizer(
        session
    )

    optimizer.optimize(
        strategy_class=DummyStrategy,
        parameters={
            "fast": [10, 30],
            "slow": [20]
        },
        constraint=lambda p: (
            p["fast"] < p["slow"]
        )
    )

    assert len(
        session.strategies_run
    ) == 1

    strategy = (
        session.strategies_run[0]
    )

    assert strategy.fast == 10
    assert strategy.slow == 20


# ==================================================
# ResearchResults
# ==================================================

def make_results():

    dataframe = pd.DataFrame([
        {
            "fast": 5,
            "slow": 20,
            "Return": 10.0,
            "Sharpe": 1.0,
            "Drawdown": 5.0,
            "Profit Factor": 1.5,
            "Win Rate": 50.0,
            "Trades": 10
        },
        {
            "fast": 10,
            "slow": 30,
            "Return": 20.0,
            "Sharpe": 2.0,
            "Drawdown": 7.0,
            "Profit Factor": 2.0,
            "Win Rate": 60.0,
            "Trades": 20
        }
    ])

    return ResearchResults(
        dataframe
    )


def test_research_results_length():

    results = make_results()

    assert len(results) == 2


def test_research_results_index_returns_research_result():

    results = make_results()

    result = results[0]

    assert isinstance(
        result,
        ResearchResult
    )


def test_research_results_best_returns_first_result():

    results = make_results()

    best = results.best()

    assert isinstance(
        best,
        ResearchResult
    )

    assert best["fast"] == 5


def test_research_results_best_empty_returns_none():

    results = ResearchResults(
        pd.DataFrame()
    )

    assert results.best() is None


def test_research_results_sort():

    results = make_results()

    results.sort(
        "Return"
    )

    assert results.best()[
        "Return"
    ] == 20.0


def test_research_results_sort_ascending():

    results = make_results()

    results.sort(
        "Return",
        ascending=True
    )

    assert results.best()[
        "Return"
    ] == 10.0


def test_to_dataframe_returns_copy():

    results = make_results()

    dataframe = (
        results.to_dataframe()
    )

    dataframe.loc[
        0,
        "Return"
    ] = 999

    assert results.dataframe.loc[
        0,
        "Return"
    ] == 10.0


def test_head():

    results = make_results()

    assert len(
        results.head(1)
    ) == 1


def test_tail():

    results = make_results()

    assert len(
        results.tail(1)
    ) == 1


# ==================================================
# ResearchResult
# ==================================================

def test_research_result_parameters():

    result = make_results()[0]

    assert result.parameters == {
        "fast": 5,
        "slow": 20
    }


def test_research_result_metrics():

    result = make_results()[0]

    assert result.metrics[
        "Return"
    ] == 10.0

    assert result.metrics[
        "Sharpe"
    ] == 1.0

    assert result.metrics[
        "Trades"
    ] == 10


def test_research_result_getitem():

    result = make_results()[0]

    assert result["Return"] == 10.0


def test_research_result_repr():

    result = make_results()[0]

    output = repr(result)

    assert "BEST STRATEGY" in output
    assert "Parameters" in output
    assert "Metrics" in output


# ==================================================
# Export
# ==================================================

def test_export_csv(
    tmp_path
):

    results = make_results()

    filename = (
        tmp_path
        / "results.csv"
    )

    results.export_csv(
        filename
    )

    assert filename.exists()

    loaded = pd.read_csv(
        filename
    )

    assert len(loaded) == 2