import pandas as pd
import pytest

from backtesting.engine import BacktestEngine


def make_ohlc():

    return pd.DataFrame(
        {
            "Open": [
                100.0,
                110.0,
                120.0,
                130.0
            ],
            "High": [
                105.0,
                115.0,
                125.0,
                135.0
            ],
            "Low": [
                95.0,
                105.0,
                115.0,
                125.0
            ],
            "Close": [
                102.0,
                112.0,
                122.0,
                132.0
            ]
        }
    )


def test_buy_signal_executes_at_next_bar_open():

    data = make_ohlc()

    signals = [
        "HOLD",
        "BUY",
        "HOLD",
        "SELL"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        data,
        signals
    )

    buy = trades[0]

    # BUY is generated after bar 1 closes.
    # Therefore execution must happen
    # at bar 2's open.
    assert buy.index == 2

    assert buy.price == pytest.approx(
        120.0
    )


def test_sell_signal_executes_at_next_bar_open():

    data = make_ohlc()

    signals = [
        "BUY",
        "HOLD",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        data,
        signals
    )

    buy = trades[0]
    sell = trades[1]

    assert buy.index == 1

    assert buy.price == pytest.approx(
        110.0
    )

    # SELL generated from bar 2 close,
    # executed at bar 3 open.
    assert sell.index == 3

    assert sell.price == pytest.approx(
        130.0
    )


def test_final_bar_buy_signal_cannot_execute():

    data = make_ohlc()

    signals = [
        "HOLD",
        "HOLD",
        "HOLD",
        "BUY"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        data,
        signals
    )

    # There is no next bar on which
    # the BUY can be executed.
    assert trades == []


def test_execution_does_not_use_signal_bar_close():

    data = make_ohlc()

    signals = [
        "BUY",
        "HOLD",
        "SELL",
        "HOLD"
    ]

    engine = BacktestEngine(
        initial_cash=10000
    )

    _, trades = engine.run(
        data,
        signals
    )

    buy = trades[0]

    # Signal-bar close = 102.
    # Next-bar open = 110.
    assert buy.price != pytest.approx(
        102.0
    )

    assert buy.price == pytest.approx(
        110.0
    )