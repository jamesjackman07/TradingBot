import pytest

from risk.manager import RiskManager


# --------------------------------------------------
# Basic risk-based sizing
# --------------------------------------------------

def test_one_percent_risk_with_five_percent_stop():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    shares = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=95
    )

    # Account risk = £100.
    # Risk/share = £5.
    # £100 / £5 = 20 shares.
    assert shares == pytest.approx(
        20
    )


def test_two_percent_risk():

    risk = RiskManager(
        risk_per_trade=0.02
    )

    shares = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=95
    )

    # £200 risk / £5 per share.
    assert shares == pytest.approx(
        40
    )


def test_tighter_stop_increases_position_size():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    wide_stop = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=95
    )

    tight_stop = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=99
    )

    assert tight_stop > wide_stop


# --------------------------------------------------
# Capital constraints
# --------------------------------------------------

def test_position_is_capped_by_available_cash():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    shares = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=99.90
    )

    # Pure risk sizing would request
    # 1,000 shares = £100,000.
    #
    # Cash account can only afford
    # 100 shares.
    assert shares == pytest.approx(
        100
    )


def test_commission_is_included_in_cash_cap():

    risk = RiskManager(
        risk_per_trade=0.01,
        commission=10
    )

    shares = risk.risk_based_shares(
        cash=10000,
        entry_price=100,
        stop_price=99.90
    )

    # Only £9,990 is available for
    # shares after commission.
    assert shares == pytest.approx(
        99.9
    )


# --------------------------------------------------
# Intended monetary risk
# --------------------------------------------------

def test_position_matches_risk_budget():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    shares = risk.risk_based_shares(
        cash=10000,
        entry_price=50,
        stop_price=48
    )

    loss_at_stop = (
        50 - 48
    ) * shares

    assert loss_at_stop == pytest.approx(
        100
    )


# --------------------------------------------------
# Invalid risk settings
# --------------------------------------------------

def test_negative_risk_per_trade_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            risk_per_trade=-0.01
        )


def test_zero_risk_per_trade_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            risk_per_trade=0
        )


def test_risk_above_100_percent_is_rejected():

    with pytest.raises(
        ValueError
    ):

        RiskManager(
            risk_per_trade=1.01
        )


# --------------------------------------------------
# Invalid stop geometry
# --------------------------------------------------

def test_stop_equal_to_entry_is_rejected():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    with pytest.raises(
        ValueError
    ):

        risk.risk_based_shares(
            cash=10000,
            entry_price=100,
            stop_price=100
        )


def test_stop_above_entry_is_rejected_for_long_trade():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    with pytest.raises(
        ValueError
    ):

        risk.risk_based_shares(
            cash=10000,
            entry_price=100,
            stop_price=105
        )


def test_non_positive_entry_price_is_rejected():

    risk = RiskManager(
        risk_per_trade=0.01
    )

    with pytest.raises(
        ValueError
    ):

        risk.risk_based_shares(
            cash=10000,
            entry_price=0,
            stop_price=-5
        )