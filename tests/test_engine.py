import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from risk.manager import RiskManager


def make_prices(values):

    return pd.Series(
        values,
        dtype=float
    )


# --------------------------------------------------
# Basic behaviour
# --------------------------------------------------

def test_no_signals_keeps_cash_constant():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, trades = engine.run(
        prices,
        signals
    )

    assert len(equity) == 3

    assert equity.tolist() == [
        10000,
        10000,
        10000
    ]

    assert trades == []


def test_equity_series_is_named_equity():

    prices = make_prices([
        100,
        100
    ])

    signals = [
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine()

    equity, _ = engine.run(
        prices,
        signals
    )

    assert equity.name == "Equity"


# --------------------------------------------------
# Buying
# --------------------------------------------------

def test_buy_signal_opens_position_next_bar():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        prices,
        signals
    )

    buy = trades[0]

    assert buy.trade_type == "BUY"
    assert buy.price == pytest.approx(110)
    assert buy.shares == pytest.approx(
        10000 / 110
    )
    assert buy.index == 1


def test_duplicate_buy_signal_is_ignored():

    prices = make_prices([
        100,
        110,
        120,
        130
    ])

    signals = [
        "BUY",
        "BUY",
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        prices,
        signals
    )

    buys = [
        trade
        for trade in trades
        if trade.trade_type == "BUY"
    ]

    assert len(buys) == 1


# --------------------------------------------------
# Selling
# --------------------------------------------------

def test_sell_signal_closes_position_next_bar():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    buy = trades[0]
    sell = trades[1]

    assert buy.price == pytest.approx(110)
    assert buy.index == 1

    assert sell.trade_type == "SELL"
    assert sell.price == pytest.approx(120)
    assert sell.index == 2
    assert sell.reason == "SIGNAL"

    assert equity.iloc[-1] == pytest.approx(
        (10000 / 110) * 120
    )


def test_sell_without_position_is_ignored():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "SELL",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, trades = engine.run(
        prices,
        signals
    )

    assert trades == []

    assert equity.iloc[-1] == pytest.approx(
        10000
    )


# --------------------------------------------------
# Profit and loss
# --------------------------------------------------

def test_profitable_trade():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, _ = engine.run(
        prices,
        signals
    )

    expected = (
        10000 / 110
    ) * 120

    assert equity.iloc[-1] == pytest.approx(
        expected
    )


def test_losing_trade():

    prices = make_prices([
        100,
        100,
        80
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, _ = engine.run(
        prices,
        signals
    )

    assert equity.iloc[-1] == pytest.approx(
        8000
    )


def test_open_position_is_marked_to_market():

    prices = make_prices([
        100,
        100,
        110,
        120
    ])

    signals = [
        "BUY",
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, _ = engine.run(
        prices,
        signals
    )

    # Bar 0:
    # BUY signal exists but has not
    # executed yet.
    assert equity.iloc[0] == pytest.approx(
        10000
    )

    # Bar 1:
    # BUY executes at 100.
    assert equity.iloc[1] == pytest.approx(
        10000
    )

    assert equity.iloc[2] == pytest.approx(
        11000
    )

    assert equity.iloc[3] == pytest.approx(
        12000
    )


# --------------------------------------------------
# End-of-data liquidation
# --------------------------------------------------

def test_open_position_is_closed_at_end_of_data():

    prices = make_prices([
        100,
        100,
        120
    ])

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    final_trade = trades[-1]

    assert final_trade.trade_type == "SELL"
    assert final_trade.reason == "END_OF_DATA"
    assert final_trade.index == 2

    assert final_trade.price == pytest.approx(
        120
    )

    assert equity.iloc[-1] == pytest.approx(
        12000
    )


# --------------------------------------------------
# Risk percentage
# --------------------------------------------------

def test_position_size_respects_risk_percent():

    risk = RiskManager(
        risk_percent=50
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    equity, trades = engine.run(
        prices,
        signals
    )

    assert trades[0].shares == pytest.approx(
        50
    )

    assert equity.iloc[-1] == pytest.approx(
        11000
    )


# --------------------------------------------------
# Commission
# --------------------------------------------------

def test_commission_is_charged_on_buy_and_sell():

    risk = RiskManager(
        risk_percent=50,
        commission=10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    equity, _ = engine.run(
        prices,
        signals
    )

    assert equity.iloc[-1] == pytest.approx(
        10980
    )


# --------------------------------------------------
# Slippage
# --------------------------------------------------

def test_slippage_worsens_execution():

    risk = RiskManager(
        risk_percent=50,
        slippage=0.01
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    _, trades = engine.run(
        prices,
        signals
    )

    assert trades[0].price == pytest.approx(
        101
    )

    assert trades[1].price == pytest.approx(
        118.8
    )


# --------------------------------------------------
# Stop loss
# --------------------------------------------------

def test_stop_loss_closes_position():

    risk = RiskManager(
        risk_percent=50,
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        94,
        110
    ])

    signals = [
        "BUY",
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    _, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    exit_trade = trades[-1]

    assert exit_trade.trade_type == "SELL"
    assert exit_trade.reason == "STOP_LOSS"
    assert exit_trade.index == 2

    assert exit_trade.price == pytest.approx(
        94
    )


def test_stop_loss_is_processed_before_new_signal():

    risk = RiskManager(
        risk_percent=50,
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        94,
        110
    ])

    signals = [
        "BUY",
        "HOLD",
        "SELL",
        "HOLD"
    ]

    _, trades = engine.run(
        prices,
        signals
    )

    # Position should already have been
    # stopped before the later SELL
    # becomes actionable.
    assert trades[-1].reason == "STOP_LOSS"


# --------------------------------------------------
# Take profit
# --------------------------------------------------

def test_take_profit_closes_position():

    risk = RiskManager(
        risk_percent=50,
        take_profit=0.10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    prices = make_prices([
        100,
        100,
        111,
        120
    ])

    signals = [
        "BUY",
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    _, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    exit_trade = trades[-1]

    assert exit_trade.trade_type == "SELL"
    assert exit_trade.reason == "TAKE_PROFIT"
    assert exit_trade.index == 2

    assert exit_trade.price == pytest.approx(
        111
    )


# --------------------------------------------------
# Trade metadata
# --------------------------------------------------

def test_trade_value():

    prices = make_prices([
        100,
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        prices,
        signals
    )

    assert trades[0].value == pytest.approx(
        10000
    )

    assert trades[1].value == pytest.approx(
        12000
    )