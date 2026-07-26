import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import Metrics


class DummyTrade:

    def __init__(self, profit):
        self.profit = profit


# --------------------------------------------------
# Total Return
# --------------------------------------------------

def test_total_return_profit():

    result = Metrics.total_return(
        10000,
        12000
    )

    assert result == pytest.approx(20.0)


def test_total_return_loss():

    result = Metrics.total_return(
        10000,
        8000
    )

    assert result == pytest.approx(-20.0)


def test_total_return_flat():

    result = Metrics.total_return(
        10000,
        10000
    )

    assert result == pytest.approx(0.0)


# --------------------------------------------------
# CAGR
# --------------------------------------------------

def test_cagr_one_year():

    result = Metrics.cagr(
        10000,
        11000,
        252
    )

    assert result == pytest.approx(10.0)


def test_cagr_two_years():

    result = Metrics.cagr(
        10000,
        12100,
        504
    )

    assert result == pytest.approx(10.0)


def test_cagr_with_insufficient_periods():

    assert Metrics.cagr(
        10000,
        11000,
        1
    ) == 0.0


# --------------------------------------------------
# Volatility
# --------------------------------------------------

def test_volatility_with_single_value():

    equity = pd.Series([
        10000
    ])

    assert Metrics.volatility(
        equity
    ) == 0.0


def test_volatility_constant_equity():

    equity = pd.Series([
        10000,
        10000,
        10000,
        10000
    ])

    assert Metrics.volatility(
        equity
    ) == pytest.approx(0.0)


def test_volatility_matches_expected_calculation():

    equity = pd.Series([
        100,
        110,
        99,
        108
    ])

    returns = equity.pct_change().dropna()

    expected = (
        returns.std()
        * np.sqrt(252)
        * 100
    )

    assert Metrics.volatility(
        equity
    ) == pytest.approx(expected)


# --------------------------------------------------
# Sharpe Ratio
# --------------------------------------------------

def test_sharpe_single_value():

    equity = pd.Series([
        10000
    ])

    assert Metrics.sharpe_ratio(
        equity
    ) == 0.0


def test_sharpe_constant_equity():

    equity = pd.Series([
        10000,
        10000,
        10000
    ])

    assert Metrics.sharpe_ratio(
        equity
    ) == 0.0


def test_sharpe_matches_expected_calculation():

    equity = pd.Series([
        100,
        102,
        101,
        105,
        104
    ])

    returns = equity.pct_change().dropna()

    expected = (
        returns.mean()
        / returns.std()
    ) * np.sqrt(252)

    assert Metrics.sharpe_ratio(
        equity
    ) == pytest.approx(expected)


def test_sharpe_with_risk_free_rate():

    equity = pd.Series([
        100,
        102,
        101,
        105,
        104
    ])

    risk_free_rate = 0.05

    returns = equity.pct_change().dropna()

    excess = (
        returns
        - risk_free_rate / 252
    )

    expected = (
        excess.mean()
        / excess.std()
    ) * np.sqrt(252)

    assert Metrics.sharpe_ratio(
        equity,
        risk_free_rate
    ) == pytest.approx(expected)


# --------------------------------------------------
# Sortino Ratio
# --------------------------------------------------

def test_sortino_single_value():

    equity = pd.Series([
        10000
    ])

    assert Metrics.sortino_ratio(
        equity
    ) == 0.0


def test_sortino_with_no_negative_returns():

    equity = pd.Series([
        100,
        101,
        102,
        103
    ])

    assert Metrics.sortino_ratio(
        equity
    ) == 0.0


def test_sortino_matches_expected_calculation():

    equity = pd.Series([
        100,
        105,
        100,
        103,
        99,
        104
    ])

    returns = equity.pct_change().dropna()

    downside = returns[
        returns < 0
    ]

    expected = (
        returns.mean()
        / downside.std()
    ) * np.sqrt(252)

    assert Metrics.sortino_ratio(
        equity
    ) == pytest.approx(expected)


# --------------------------------------------------
# Maximum Drawdown
# --------------------------------------------------

def test_max_drawdown_no_drawdown():

    equity = pd.Series([
        100,
        110,
        120,
        130
    ])

    assert Metrics.max_drawdown(
        equity
    ) == pytest.approx(0.0)


def test_max_drawdown():

    equity = pd.Series([
        100,
        120,
        90,
        110
    ])

    # Peak = 120
    # Trough = 90
    #
    # Drawdown:
    # (120 - 90) / 120 = 25%

    assert Metrics.max_drawdown(
        equity
    ) == pytest.approx(25.0)


def test_max_drawdown_uses_highest_previous_peak():

    equity = pd.Series([
        100,
        150,
        120,
        140,
        90
    ])

    expected = (
        (150 - 90)
        / 150
        * 100
    )

    assert Metrics.max_drawdown(
        equity
    ) == pytest.approx(expected)


# --------------------------------------------------
# Win Rate
# --------------------------------------------------

def test_win_rate_no_trades():

    assert Metrics.win_rate(
        []
    ) == 0.0


def test_win_rate():

    trades = [
        DummyTrade(100),
        DummyTrade(-50),
        DummyTrade(200),
        DummyTrade(-25)
    ]

    assert Metrics.win_rate(
        trades
    ) == pytest.approx(50.0)


def test_break_even_trade_is_not_a_win():

    trades = [
        DummyTrade(100),
        DummyTrade(0)
    ]

    assert Metrics.win_rate(
        trades
    ) == pytest.approx(50.0)


# --------------------------------------------------
# Profit Factor
# --------------------------------------------------

def test_profit_factor_no_trades():

    assert Metrics.profit_factor(
        []
    ) == 0.0


def test_profit_factor():

    trades = [
        DummyTrade(100),
        DummyTrade(200),
        DummyTrade(-50),
        DummyTrade(-100)
    ]

    # Gross profit = 300
    # Gross loss = 150

    assert Metrics.profit_factor(
        trades
    ) == pytest.approx(2.0)


def test_profit_factor_all_winners():

    trades = [
        DummyTrade(100),
        DummyTrade(200)
    ]

    assert Metrics.profit_factor(
        trades
    ) == float("inf")


def test_profit_factor_all_losers():

    trades = [
        DummyTrade(-100),
        DummyTrade(-200)
    ]

    assert Metrics.profit_factor(
        trades
    ) == 0.0


# --------------------------------------------------
# Average Win
# --------------------------------------------------

def test_average_win():

    trades = [
        DummyTrade(100),
        DummyTrade(200),
        DummyTrade(-50)
    ]

    assert Metrics.average_win(
        trades
    ) == pytest.approx(150.0)


def test_average_win_without_winners():

    trades = [
        DummyTrade(-100),
        DummyTrade(-50)
    ]

    assert Metrics.average_win(
        trades
    ) == 0.0


# --------------------------------------------------
# Average Loss
# --------------------------------------------------

def test_average_loss():

    trades = [
        DummyTrade(100),
        DummyTrade(-50),
        DummyTrade(-150)
    ]

    assert Metrics.average_loss(
        trades
    ) == pytest.approx(-100.0)


def test_average_loss_without_losers():

    trades = [
        DummyTrade(100),
        DummyTrade(50)
    ]

    assert Metrics.average_loss(
        trades
    ) == 0.0


# --------------------------------------------------
# Best / Worst Trade
# --------------------------------------------------

def test_best_trade():

    trades = [
        DummyTrade(100),
        DummyTrade(-50),
        DummyTrade(250)
    ]

    assert Metrics.best_trade(
        trades
    ) == 250


def test_best_trade_no_trades():

    assert Metrics.best_trade(
        []
    ) == 0.0


def test_worst_trade():

    trades = [
        DummyTrade(100),
        DummyTrade(-75),
        DummyTrade(250)
    ]

    assert Metrics.worst_trade(
        trades
    ) == -75


def test_worst_trade_no_trades():

    assert Metrics.worst_trade(
        []
    ) == 0.0