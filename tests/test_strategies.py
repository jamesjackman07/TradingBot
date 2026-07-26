import pandas as pd
import pytest

from bot.indicators import Indicators

from bot.strategies.sma_cross import (
    SMACrossoverStrategy
)

from bot.strategies.momentum import (
    MomentumStrategy
)

from bot.strategies.mean_reversion import (
    MeanReversionStrategy
)


def make_prices(values):

    return pd.Series(
        values,
        dtype=float
    )


# ==================================================
# Indicators
# ==================================================

def test_sma():

    prices = make_prices([
        1,
        2,
        3,
        4,
        5
    ])

    sma = Indicators.sma(
        prices,
        period=3
    )

    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])

    assert sma.iloc[2] == pytest.approx(2)
    assert sma.iloc[3] == pytest.approx(3)
    assert sma.iloc[4] == pytest.approx(4)


def test_ema_has_same_length_as_input():

    prices = make_prices([
        1,
        2,
        3,
        4,
        5
    ])

    ema = Indicators.ema(
        prices,
        period=3
    )

    assert len(ema) == len(prices)


def test_bollinger_bands():

    prices = make_prices([
        1,
        2,
        3,
        4,
        5
    ])

    upper, middle, lower = (
        Indicators.bollinger_bands(
            prices,
            period=3,
            std=2
        )
    )

    assert len(upper) == len(prices)
    assert len(middle) == len(prices)
    assert len(lower) == len(prices)

    assert pd.isna(upper.iloc[0])
    assert pd.isna(middle.iloc[0])
    assert pd.isna(lower.iloc[0])

    assert upper.iloc[-1] > middle.iloc[-1]
    assert lower.iloc[-1] < middle.iloc[-1]


# ==================================================
# SMA Crossover
# ==================================================

def test_sma_default_parameters():

    strategy = SMACrossoverStrategy()

    assert strategy.fast == 20
    assert strategy.slow == 50


def test_sma_parameter_constraint_accepts_valid():

    assert (
        SMACrossoverStrategy.parameter_constraint(
            {
                "fast": 10,
                "slow": 50
            }
        )
        is True
    )


def test_sma_parameter_constraint_rejects_equal():

    assert (
        SMACrossoverStrategy.parameter_constraint(
            {
                "fast": 50,
                "slow": 50
            }
        )
        is False
    )


def test_sma_parameter_constraint_rejects_fast_above_slow():

    assert (
        SMACrossoverStrategy.parameter_constraint(
            {
                "fast": 100,
                "slow": 50
            }
        )
        is False
    )


def test_sma_warmup_period():

    strategy = SMACrossoverStrategy(
        fast=5,
        slow=20
    )

    assert strategy.warmup_period() == 21


def test_sma_signal_length_matches_prices():

    prices = make_prices([
        1,
        2,
        3,
        4,
        5,
        6
    ])

    strategy = SMACrossoverStrategy(
        fast=2,
        slow=3
    )

    signals = strategy.generate_signals(
        prices
    )

    assert len(signals) == len(prices)


def test_sma_holds_during_initial_warmup():

    prices = make_prices([
        1,
        2,
        3
    ])

    strategy = SMACrossoverStrategy(
        fast=2,
        slow=3
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[0] == "HOLD"
    assert signals[1] == "HOLD"


def test_sma_generates_buy_on_upward_crossover():

    prices = make_prices([
        5,
        4,
        3,
        2,
        3,
        4
    ])

    strategy = SMACrossoverStrategy(
        fast=2,
        slow=3
    )

    signals = strategy.generate_signals(
        prices
    )

    assert "BUY" in signals


def test_sma_generates_sell_on_downward_crossover():

    prices = make_prices([
        1,
        2,
        3,
        4,
        3,
        2
    ])

    strategy = SMACrossoverStrategy(
        fast=2,
        slow=3
    )

    signals = strategy.generate_signals(
        prices
    )

    assert "SELL" in signals


# ==================================================
# Momentum
# ==================================================

def test_momentum_default_parameter():

    strategy = MomentumStrategy()

    assert strategy.lookback == 60


def test_momentum_warmup_period():

    strategy = MomentumStrategy(
        lookback=10
    )

    assert strategy.warmup_period() == 11


def test_momentum_signal_length_matches_prices():

    prices = make_prices([
        100,
        101,
        102,
        103,
        104
    ])

    strategy = MomentumStrategy(
        lookback=2
    )

    signals = strategy.generate_signals(
        prices
    )

    assert len(signals) == len(prices)


def test_momentum_holds_during_warmup():

    prices = make_prices([
        100,
        101,
        102,
        103
    ])

    strategy = MomentumStrategy(
        lookback=2
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[0] == "HOLD"
    assert signals[1] == "HOLD"


def test_momentum_buys_when_momentum_becomes_positive():

    prices = make_prices([
        100,
        100,
        101,
        102
    ])

    strategy = MomentumStrategy(
        lookback=2
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[2] == "BUY"


def test_momentum_does_not_repeatedly_buy():

    prices = make_prices([
        100,
        101,
        102,
        103,
        104
    ])

    strategy = MomentumStrategy(
        lookback=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals.count("BUY") == 1


def test_momentum_sells_when_momentum_turns_non_positive():

    prices = make_prices([
        100,
        102,
        104,
        103,
        102
    ])

    strategy = MomentumStrategy(
        lookback=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert "BUY" in signals
    assert "SELL" in signals

    assert signals.index(
        "SELL"
    ) > signals.index(
        "BUY"
    )


# ==================================================
# Mean Reversion
# ==================================================

def test_mean_reversion_default_parameters():

    strategy = MeanReversionStrategy()

    assert strategy.period == 20
    assert strategy.std == 2


def test_mean_reversion_warmup_period():

    strategy = MeanReversionStrategy(
        period=10
    )

    assert strategy.warmup_period() == 11


def test_mean_reversion_signal_length_matches_prices():

    prices = make_prices([
        100,
        101,
        102,
        103,
        104
    ])

    strategy = MeanReversionStrategy(
        period=3,
        std=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert len(signals) == len(prices)


def test_mean_reversion_holds_during_warmup():

    prices = make_prices([
        100,
        101,
        102
    ])

    strategy = MeanReversionStrategy(
        period=3,
        std=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[0] == "HOLD"
    assert signals[1] == "HOLD"


def test_mean_reversion_buys_below_lower_band():

    prices = make_prices([
        100,
        100,
        100,
        100,
        90
    ])

    strategy = MeanReversionStrategy(
        period=3,
        std=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[-1] == "BUY"


def test_mean_reversion_sells_above_upper_band():

    prices = make_prices([
        100,
        100,
        100,
        100,
        110
    ])

    strategy = MeanReversionStrategy(
        period=3,
        std=1
    )

    signals = strategy.generate_signals(
        prices
    )

    assert signals[-1] == "SELL"