import pytest

from backtesting.portfolio import Portfolio


def test_portfolio_starts_with_initial_cash():

    portfolio = Portfolio(
        initial_cash=10000
    )

    assert portfolio.cash == 10000
    assert portfolio.shares == 0


def test_new_portfolio_has_no_position():

    portfolio = Portfolio(
        initial_cash=10000
    )

    assert portfolio.has_position() is False


def test_buy_reduces_cash_and_adds_shares():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    assert portfolio.cash == 5000
    assert portfolio.shares == 50
    assert portfolio.has_position() is True


def test_buy_includes_commission():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50,
        commission=10
    )

    assert portfolio.cash == 4990
    assert portfolio.shares == 50


def test_cannot_buy_when_already_in_position():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    portfolio.buy(
        price=200,
        shares=10
    )

    assert portfolio.cash == 5000
    assert portfolio.shares == 50


def test_sell_closes_position():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    portfolio.sell(
        price=120
    )

    assert portfolio.cash == 11000
    assert portfolio.shares == 0
    assert portfolio.has_position() is False


def test_sell_includes_commission():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    portfolio.sell(
        price=120,
        commission=10
    )

    assert portfolio.cash == 10990
    assert portfolio.shares == 0


def test_sell_does_nothing_without_position():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.sell(
        price=120,
        commission=10
    )

    assert portfolio.cash == 10000
    assert portfolio.shares == 0


def test_equity_when_holding_cash_only():

    portfolio = Portfolio(
        initial_cash=10000
    )

    assert portfolio.equity(
        current_price=100
    ) == 10000


def test_equity_with_open_position():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    equity = portfolio.equity(
        current_price=120
    )

    assert equity == 11000


def test_profitable_round_trip():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    portfolio.sell(
        price=120
    )

    assert portfolio.cash == 11000


def test_losing_round_trip():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50
    )

    portfolio.sell(
        price=80
    )

    assert portfolio.cash == 9000


def test_round_trip_with_commissions():

    portfolio = Portfolio(
        initial_cash=10000
    )

    portfolio.buy(
        price=100,
        shares=50,
        commission=5
    )

    portfolio.sell(
        price=120,
        commission=5
    )

    assert portfolio.cash == 10990