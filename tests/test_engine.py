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

def test_buy_signal_opens_position():

    prices = make_prices([
        100,
        110
    ])

    signals = [
        "BUY",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        prices,
        signals
    )

    assert trades[0].trade_type == "BUY"
    assert trades[0].price == pytest.approx(100)
    assert trades[0].shares == pytest.approx(100)
    assert trades[0].index == 0


def test_duplicate_buy_signal_is_ignored():

    prices = make_prices([
        100,
        110,
        120
    ])

    signals = [
        "BUY",
        "BUY",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    assert trades[0].trade_type == "BUY"
    assert trades[1].trade_type == "SELL"


# --------------------------------------------------
# Selling
# --------------------------------------------------

def test_sell_signal_closes_position():

    prices = make_prices([
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, trades = engine.run(
        prices,
        signals
    )

    assert len(trades) == 2

    sell = trades[1]

    assert sell.trade_type == "SELL"
    assert sell.price == pytest.approx(120)
    assert sell.index == 1
    assert sell.reason == "SIGNAL"

    assert equity.iloc[-1] == pytest.approx(
        12000
    )


def test_sell_without_position_is_ignored():

    prices = make_prices([
        100,
        110
    ])

    signals = [
        "SELL",
        "SELL"
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
        120
    ])

    signals = [
        "BUY",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, _ = engine.run(
        prices,
        signals
    )

    assert equity.iloc[-1] == pytest.approx(
        12000
    )


def test_losing_trade():

    prices = make_prices([
        100,
        80
    ])

    signals = [
        "BUY",
        "SELL"
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
        110,
        120
    ])

    signals = [
        "BUY",
        "HOLD",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    equity, _ = engine.run(
        prices,
        signals
    )

    assert equity.iloc[0] == pytest.approx(
        10000
    )

    assert equity.iloc[1] == pytest.approx(
        11000
    )

    assert equity.iloc[2] == pytest.approx(
        12000
    )


# --------------------------------------------------
# End-of-data liquidation
# --------------------------------------------------

def test_open_position_is_closed_at_end_of_data():

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
        120
    ])

    signals = [
        "BUY",
        "SELL"
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
        120
    ])

    signals = [
        "BUY",
        "SELL"
    ]

    equity, _ = engine.run(
        prices,
        signals
    )

    # 50 shares:
    #
    # Start       = 10000
    # Buy cost    = 5000
    # Buy fee     = 10
    # Cash        = 4990
    #
    # Sell value  = 6000
    # Sell fee    = 10
    #
    # Final       = 10980

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
        120
    ])

    signals = [
        "BUY",
        "SELL"
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
        94,
        110
    ])

    signals = [
        "BUY",
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
    assert exit_trade.index == 1
    assert exit_trade.price == pytest.approx(
        94
    )


def test_stop_loss_takes_priority_over_signal_processing():

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
        94
    ])

    signals = [
        "BUY",
        "SELL"
    ]

    _, trades = engine.run(
        prices,
        signals
    )

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
        111,
        120
    ])

    signals = [
        "BUY",
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
    assert exit_trade.index == 1
    assert exit_trade.price == pytest.approx(
        111
    )


# --------------------------------------------------
# Trade metadata
# --------------------------------------------------

def test_trade_value():

    prices = make_prices([
        100,
        120
    ])

    signals = [
        "BUY",
        "SELL"
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