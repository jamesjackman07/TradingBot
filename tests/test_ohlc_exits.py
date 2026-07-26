import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from risk.manager import RiskManager


def make_data(
    opens,
    highs,
    lows,
    closes
):

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes
        },
        dtype=float
    )


# --------------------------------------------------
# Stop loss
# --------------------------------------------------

def test_intrabar_low_triggers_stop_loss():

    data = make_data(
        opens=[
            100,
            100,
            101,
            102
        ],
        highs=[
            101,
            103,
            104,
            105
        ],
        lows=[
            99,
            99,
            94,
            101
        ],
        closes=[
            100,
            102,
            102,
            104
        ]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_percent=50,
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    assert len(trades) == 2

    buy = trades[0]
    sell = trades[1]

    assert buy.price == pytest.approx(
        100
    )

    assert sell.reason == "STOP_LOSS"
    assert sell.index == 2

    # Entry = 100, so 5% stop = 95.
    assert sell.price == pytest.approx(
        95
    )


def test_close_above_stop_does_not_prevent_intrabar_stop():

    data = make_data(
        opens=[100, 100, 100],
        highs=[101, 102, 105],
        lows=[99, 99, 94],
        closes=[100, 101, 104]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    assert trades[-1].reason == (
        "STOP_LOSS"
    )

    assert trades[-1].price == (
        pytest.approx(95)
    )


# --------------------------------------------------
# Take profit
# --------------------------------------------------

def test_intrabar_high_triggers_take_profit():

    data = make_data(
        opens=[
            100,
            100,
            102,
            103
        ],
        highs=[
            101,
            105,
            112,
            104
        ],
        lows=[
            99,
            99,
            101,
            102
        ],
        closes=[
            100,
            102,
            103,
            103
        ]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_percent=50,
        take_profit=0.10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    sell = trades[-1]

    assert sell.reason == "TAKE_PROFIT"
    assert sell.index == 2

    # Entry = 100, so target = 110.
    assert sell.price == pytest.approx(
        110
    )


def test_close_below_target_does_not_prevent_intrabar_target():

    data = make_data(
        opens=[100, 100, 102],
        highs=[101, 105, 112],
        lows=[99, 99, 100],
        closes=[100, 102, 103]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        take_profit=0.10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    assert trades[-1].reason == (
        "TAKE_PROFIT"
    )

    assert trades[-1].price == (
        pytest.approx(110)
    )


# --------------------------------------------------
# Gap through stop
# --------------------------------------------------

def test_gap_below_stop_fills_at_open():

    data = make_data(
        opens=[
            100,
            100,
            90
        ],
        highs=[
            101,
            102,
            92
        ],
        lows=[
            99,
            99,
            88
        ],
        closes=[
            100,
            101,
            91
        ]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    sell = trades[-1]

    assert sell.reason == "STOP_LOSS"

    # Stop was 95, but market opened
    # below it at 90. We cannot assume
    # a fill at 95.
    assert sell.price == pytest.approx(
        90
    )


# --------------------------------------------------
# Gap through target
# --------------------------------------------------

def test_gap_above_target_fills_at_open():

    data = make_data(
        opens=[
            100,
            100,
            115
        ],
        highs=[
            101,
            103,
            118
        ],
        lows=[
            99,
            99,
            114
        ],
        closes=[
            100,
            102,
            116
        ]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        take_profit=0.10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    sell = trades[-1]

    assert sell.reason == "TAKE_PROFIT"

    # Target = 110, but the market
    # opened above it at 115.
    assert sell.price == pytest.approx(
        115
    )


# --------------------------------------------------
# Ambiguous OHLC bar
# --------------------------------------------------

def test_stop_wins_when_stop_and_target_both_touched():

    data = make_data(
        opens=[
            100,
            100,
            100
        ],
        highs=[
            101,
            103,
            112
        ],
        lows=[
            99,
            99,
            94
        ],
        closes=[
            100,
            101,
            105
        ]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        stop_loss=0.05,
        take_profit=0.10
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    sell = trades[-1]

    # Daily OHLC cannot tell us whether
    # 95 or 110 was reached first.
    #
    # Conservative backtesting policy:
    # assume the adverse level was hit
    # first.
    assert sell.reason == "STOP_LOSS"

    assert sell.price == pytest.approx(
        95
    )