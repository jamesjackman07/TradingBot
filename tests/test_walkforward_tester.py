import pandas as pd
import pytest

from walkforward.tester import WalkForwardTester
from bot.research import ResearchSession
from risk.manager import RiskManager


def make_data(rows=20):

    index = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="D"
    )

    return pd.DataFrame(
        {
            "Close": range(
                100,
                100 + rows
            )
        },
        index=index
    )


# ==================================================
# Construction / validation
# ==================================================

def test_valid_settings_are_stored():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    assert tester.train_size == 10
    assert tester.test_size == 5
    assert tester.step_size == 5


def test_rejects_zero_train_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="train_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=0,
            test_size=5,
            step_size=5
        )


def test_rejects_negative_train_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="train_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=-1,
            test_size=5,
            step_size=5
        )


def test_rejects_zero_test_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="test_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=10,
            test_size=0,
            step_size=5
        )


def test_rejects_negative_test_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="test_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=10,
            test_size=-1,
            step_size=5
        )


def test_rejects_zero_step_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="step_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=10,
            test_size=5,
            step_size=0
        )


def test_rejects_negative_step_size():

    data = make_data(20)

    with pytest.raises(
        ValueError,
        match="step_size must be greater than 0"
    ):

        WalkForwardTester(
            data=data,
            train_size=10,
            test_size=5,
            step_size=-1
        )


def test_rejects_insufficient_data():

    data = make_data(14)

    with pytest.raises(
        ValueError,
        match="Not enough data"
    ):

        WalkForwardTester(
            data=data,
            train_size=10,
            test_size=5,
            step_size=5
        )


def test_accepts_exactly_one_window_of_data():

    data = make_data(15)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    windows = list(
        tester.windows()
    )

    assert len(windows) == 1


# ==================================================
# Windows
# ==================================================

def test_first_window_boundaries():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    assert window["train_start"] == 0
    assert window["train_end"] == 10

    assert window["test_start"] == 10
    assert window["test_end"] == 15


def test_first_window_data_lengths():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    assert len(
        window["train_data"]
    ) == 10

    assert len(
        window["test_data"]
    ) == 5


def test_train_and_test_do_not_overlap():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    for window in tester.windows():

        train_index = set(
            window["train_data"].index
        )

        test_index = set(
            window["test_data"].index
        )

        assert train_index.isdisjoint(
            test_index
        )


def test_test_period_immediately_follows_training_period():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    assert (
        window["train_data"].index[-1]
        <
        window["test_data"].index[0]
    )

    assert (
        window["test_start"]
        ==
        window["train_end"]
    )


def test_step_size_moves_next_window():

    data = make_data(25)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    windows = list(
        tester.windows()
    )

    assert windows[0][
        "train_start"
    ] == 0

    assert windows[1][
        "train_start"
    ] == 5

    assert windows[2][
        "train_start"
    ] == 10


def test_expected_number_of_windows():

    data = make_data(25)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    windows = list(
        tester.windows()
    )

    assert len(windows) == 3


def test_incomplete_final_window_is_not_returned():

    data = make_data(22)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    windows = list(
        tester.windows()
    )

    assert len(windows) == 2


def test_window_data_are_copies():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    window["train_data"].iloc[
        0,
        0
    ] = 999999

    assert data.iloc[
        0,
        0
    ] != 999999


# ==================================================
# Risk manager
# ==================================================

def test_default_risk_manager_is_created():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    assert isinstance(
        tester.risk_manager,
        RiskManager
    )


def test_supplied_risk_manager_is_stored():

    data = make_data(20)

    risk = RiskManager(
        commission=5,
        slippage=0.001
    )

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        risk_manager=risk
    )

    assert tester.risk_manager is risk


# ==================================================
# Research sessions
# ==================================================

def test_create_train_session():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    session = (
        tester.create_train_session(
            window
        )
    )

    assert isinstance(
        session,
        ResearchSession
    )


def test_create_test_session():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    session = (
        tester.create_test_session(
            window
        )
    )

    assert isinstance(
        session,
        ResearchSession
    )


def test_train_session_contains_training_data():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    session = (
        tester.create_train_session(
            window
        )
    )

    pd.testing.assert_series_equal(
        session.close,
        window["train_data"]["Close"]
    )


def test_test_session_contains_test_data():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    session = (
        tester.create_test_session(
            window
        )
    )

    pd.testing.assert_series_equal(
        session.close,
        window["test_data"]["Close"]
    )


def test_sessions_use_supplied_risk_manager():

    data = make_data(20)

    risk = RiskManager(
        commission=5
    )

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5,
        risk_manager=risk
    )

    window = next(
        tester.windows()
    )

    train_session = (
        tester.create_train_session(
            window
        )
    )

    test_session = (
        tester.create_test_session(
            window
        )
    )

    assert (
        train_session.engine.risk
        is risk
    )

    assert (
        test_session.engine.risk
        is risk
    )


# ==================================================
# Warm-up
# ==================================================

def test_warmup_adds_previous_data():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    warmup = (
        tester.get_test_with_warmup(
            window,
            warmup_size=3
        )
    )

    # Test period = positions 10:15
    # Warm-up adds positions 7:10.
    assert len(warmup) == 8

    assert warmup.index[0] == (
        data.index[7]
    )

    assert warmup.index[-1] == (
        data.index[14]
    )


def test_zero_warmup_returns_only_test_data():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    warmup = (
        tester.get_test_with_warmup(
            window,
            warmup_size=0
        )
    )

    pd.testing.assert_frame_equal(
        warmup,
        window["test_data"]
    )


def test_warmup_does_not_go_before_start_of_dataset():

    data = make_data(15)

    tester = WalkForwardTester(
        data=data,
        train_size=5,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    warmup = (
        tester.get_test_with_warmup(
            window,
            warmup_size=100
        )
    )

    assert warmup.index[0] == (
        data.index[0]
    )

    assert warmup.index[-1] == (
        data.index[9]
    )


def test_warmup_data_is_a_copy():

    data = make_data(20)

    tester = WalkForwardTester(
        data=data,
        train_size=10,
        test_size=5,
        step_size=5
    )

    window = next(
        tester.windows()
    )

    warmup = (
        tester.get_test_with_warmup(
            window,
            warmup_size=3
        )
    )

    warmup.iloc[
        0,
        0
    ] = 999999

    assert data.iloc[
        7,
        0
    ] != 999999