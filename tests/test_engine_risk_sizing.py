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
# Risk-based sizing integration
# --------------------------------------------------

def test_engine_uses_risk_based_position_size():

    data = make_data(
        opens=[100, 100, 100],
        highs=[101, 102, 103],
        lows=[99, 99, 99],
        closes=[100, 101, 102]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_per_trade=0.01,
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

    buy = trades[0]

    # £10,000 account
    # 1% risk = £100
    #
    # Entry = £100
    # Stop = £95
    # Risk/share = £5
    #
    # £100 / £5 = 20 shares
    assert buy.shares == pytest.approx(
        20
    )


def test_stop_loss_produces_expected_price_loss():

    data = make_data(
        opens=[100, 100, 100],
        highs=[101, 102, 101],
        lows=[99, 99, 94],
        closes=[100, 101, 96]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_per_trade=0.01,
        stop_loss=0.05
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    equity, trades = engine.run(
        data,
        signals
    )

    assert trades[0].shares == pytest.approx(
        20
    )

    assert trades[-1].reason == (
        "STOP_LOSS"
    )

    # 20 shares:
    # buy £100
    # stop £95
    #
    # Loss = £100.
    assert equity.iloc[-1] == pytest.approx(
        9900
    )


# --------------------------------------------------
# Actual execution price
# --------------------------------------------------

def test_risk_sizing_uses_slipped_entry_price():

    data = make_data(
        opens=[100, 100, 102],
        highs=[101, 103, 104],
        lows=[99, 99, 100],
        closes=[100, 102, 103]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_per_trade=0.01,
        stop_loss=0.05,
        slippage=0.01
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    buy = trades[0]

    # Market open = £100
    # 1% slippage gives actual
    # entry = £101.
    assert buy.price == pytest.approx(
        101
    )

    # Stop is calculated from the actual
    # £101 entry:
    #
    # £101 * 0.95 = £95.95
    #
    # Risk/share = £5.05
    # Risk budget = £100
    expected_shares = (
        100 / 5.05
    )

    assert buy.shares == pytest.approx(
        expected_shares
    )


# --------------------------------------------------
# Capital cap
# --------------------------------------------------

def test_engine_caps_risk_size_by_available_cash():

    data = make_data(
        opens=[100, 100, 101],
        highs=[101, 102, 103],
        lows=[99, 99.95, 100],
        closes=[100, 101, 102]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_per_trade=0.01,
        stop_loss=0.001
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    buy = trades[0]

    # Risk sizing would request:
    #
    # £100 / £0.10 = 1,000 shares.
    #
    # But a £10,000 cash account can
    # only afford 100 shares at £100.
    assert buy.shares == pytest.approx(
        100
    )


# --------------------------------------------------
# Backwards compatibility
# --------------------------------------------------

def test_engine_uses_allocation_when_risk_mode_disabled():

    data = make_data(
        opens=[100, 100, 110],
        highs=[101, 102, 112],
        lows=[99, 99, 108],
        closes=[100, 101, 111]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_percent=50
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    # Existing behaviour:
    # 50% allocation = £5,000
    # £5,000 / £100 = 50 shares.
    assert trades[0].shares == pytest.approx(
        50
    )


def test_risk_per_trade_without_stop_falls_back_to_allocation():

    data = make_data(
        opens=[100, 100, 110],
        highs=[101, 102, 112],
        lows=[99, 99, 108],
        closes=[100, 101, 111]
    )

    signals = [
        "BUY",
        "HOLD",
        "HOLD"
    ]

    risk = RiskManager(
        risk_percent=50,
        risk_per_trade=0.01
    )

    engine = BacktestEngine(
        initial_cash=10000,
        risk_manager=risk
    )

    _, trades = engine.run(
        data,
        signals
    )

    # No stop means there is no defined
    # price risk per share, so the engine
    # must retain allocation sizing.
    assert trades[0].shares == pytest.approx(
        50
    )