import pytest

from risk.manager import RiskManager


def test_default_settings():

    risk = RiskManager()

    assert risk.risk_percent == 100
    assert risk.commission == 0
    assert risk.slippage == 0
    assert risk.stop_loss is None
    assert risk.take_profit is None


def test_uses_all_cash_at_100_percent_risk():

    risk = RiskManager(
        risk_percent=100
    )

    shares = risk.shares_to_buy(
        cash=10000,
        price=100
    )

    assert shares == 100


def test_position_sizing_with_partial_risk():

    risk = RiskManager(
        risk_percent=25
    )

    shares = risk.shares_to_buy(
        cash=10000,
        price=100
    )

    assert shares == 25


def test_position_sizing_allows_fractional_shares():

    risk = RiskManager(
        risk_percent=50
    )

    shares = risk.shares_to_buy(
        cash=10000,
        price=300
    )

    assert shares == pytest.approx(
        16.6666666667
    )


def test_buy_price_without_slippage():

    risk = RiskManager(
        slippage=0
    )

    assert risk.buy_price(
        100
    ) == 100


def test_sell_price_without_slippage():

    risk = RiskManager(
        slippage=0
    )

    assert risk.sell_price(
        100
    ) == 100


def test_buy_price_with_slippage():

    risk = RiskManager(
        slippage=0.001
    )

    assert risk.buy_price(
        100
    ) == pytest.approx(
        100.10
    )


def test_sell_price_with_slippage():

    risk = RiskManager(
        slippage=0.001
    )

    assert risk.sell_price(
        100
    ) == pytest.approx(
        99.90
    )


def test_no_stop_loss_returns_none():

    risk = RiskManager()

    assert risk.stop_price(
        100
    ) is None


def test_stop_loss_price():

    risk = RiskManager(
        stop_loss=0.05
    )

    assert risk.stop_price(
        100
    ) == pytest.approx(
        95
    )


def test_no_take_profit_returns_none():

    risk = RiskManager()

    assert risk.target_price(
        100
    ) is None


def test_take_profit_price():

    risk = RiskManager(
        take_profit=0.10
    )

    assert risk.target_price(
        100
    ) == pytest.approx(
        110
    )


def test_commission_is_stored():

    risk = RiskManager(
        commission=2.50
    )

    assert risk.commission == 2.50


def test_combined_configuration():

    risk = RiskManager(
        risk_percent=20,
        commission=1.50,
        slippage=0.0005,
        stop_loss=0.02,
        take_profit=0.04
    )

    assert risk.risk_percent == 20
    assert risk.commission == 1.50
    assert risk.slippage == 0.0005
    assert risk.stop_loss == 0.02
    assert risk.take_profit == 0.04

    assert risk.shares_to_buy(
        10000,
        100
    ) == pytest.approx(
        19.985
    )

    assert risk.buy_price(
        100
    ) == pytest.approx(
        100.05
    )

    assert risk.sell_price(
        100
    ) == pytest.approx(
        99.95
    )

    assert risk.stop_price(
        100
    ) == pytest.approx(
        98
    )

    assert risk.target_price(
        100
    ) == pytest.approx(
        104
    )