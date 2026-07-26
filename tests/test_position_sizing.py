import pytest

from backtesting.portfolio import Portfolio
from risk.manager import RiskManager


# --------------------------------------------------
# Allocation sizing
# --------------------------------------------------

def test_full_allocation_accounts_for_commission():

    risk = RiskManager(
        risk_percent=100,
        commission=10
    )

    shares = risk.shares_to_buy(
        cash=10000,
        price=100
    )

    total_cost = (
        shares * 100
        + risk.commission
    )

    assert total_cost <= 10000


def test_full_allocation_does_not_create_negative_cash():

    risk = RiskManager(
        risk_percent=100,
        commission=10
    )

    portfolio = Portfolio(
        initial_cash=10000
    )

    shares = risk.shares_to_buy(
        cash=portfolio.cash,
        price=100
    )

    portfolio.buy(
        price=100,
        shares=shares,
        commission=risk.commission
    )

    assert portfolio.cash >= 0


def test_half_allocation_accounts_for_commission():

    risk = RiskManager(
        risk_percent=50,
        commission=10
    )

    shares = risk.shares_to_buy(
        cash=10000,
        price=100
    )

    total_cost = (
        shares * 100
        + risk.commission
    )

    # 50% allocation should consume no
    # more than £5,000 including costs.
    assert total_cost <= 5000


# --------------------------------------------------
# Invalid configuration
# --------------------------------------------------

def test_negative_allocation_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            risk_percent=-1
        )


def test_allocation_above_100_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            risk_percent=101
        )


def test_negative_commission_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            commission=-1
        )


def test_negative_slippage_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            slippage=-0.01
        )


# --------------------------------------------------
# Portfolio protection
# --------------------------------------------------

def test_portfolio_rejects_unaffordable_purchase():

    portfolio = Portfolio(
        initial_cash=10000
    )

    with pytest.raises(
        ValueError
    ):

        portfolio.buy(
            price=100,
            shares=101,
            commission=10
        )


def test_portfolio_rejects_negative_shares():

    portfolio = Portfolio(
        initial_cash=10000
    )

    with pytest.raises(
        ValueError
    ):

        portfolio.buy(
            price=100,
            shares=-10
        )


def test_portfolio_rejects_zero_price():

    portfolio = Portfolio(
        initial_cash=10000
    )

    with pytest.raises(
        ValueError
    ):

        portfolio.buy(
            price=0,
            shares=10
        )