import pandas as pd

from optimization.optimizer import Optimizer
from bot.research import ResearchSession

from analytics.report import Report

from walkforward.tester import WalkForwardTester


class WalkForwardRunner:

    def __init__(
        self,
        data,
        train_size=504,
        test_size=126,
        step_size=126,
        initial_cash=10000,
        risk_manager=None
    ):

        self.tester = WalkForwardTester(
            data=data,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            risk_manager=risk_manager
        )

        self.initial_cash = initial_cash

    def run(
        self,
        strategy_class,
        parameters
    ):

        window_results = []

        combined_equity_parts = []

        current_cash = self.initial_cash

        # --------------------------------
        # Strategy-specific constraint
        # --------------------------------

        constraint = getattr(
            strategy_class,
            "parameter_constraint",
            None
        )

        for window_number, window in enumerate(
            self.tester.windows(),
            start=1
        ):

            # --------------------------------
            # 1. Create training session
            # --------------------------------

            train_session = (
                self.tester.create_train_session(
                    window
                )
            )

            # --------------------------------
            # 2. Optimise training data
            # --------------------------------

            optimizer = Optimizer(
                train_session
            )

            results = optimizer.optimize(
                strategy_class=strategy_class,
                parameters=parameters,
                constraint=constraint
            )

            best = results.dataframe.iloc[0]

            # --------------------------------
            # 3. Extract selected parameters
            # --------------------------------

            selected_parameters = {}

            for name in parameters:

                value = best[name]

                try:

                    numeric_value = float(value)

                    if numeric_value.is_integer():

                        value = int(
                            numeric_value
                        )

                    else:

                        value = numeric_value

                except (
                    TypeError,
                    ValueError
                ):

                    pass

                selected_parameters[
                    name
                ] = value

            # --------------------------------
            # 4. Freeze selected strategy
            # --------------------------------

            strategy = strategy_class(
                **selected_parameters
            )

            # --------------------------------
            # 5. Determine warm-up
            # --------------------------------

            if hasattr(
                strategy,
                "warmup_period"
            ):

                warmup_size = int(
                    strategy.warmup_period()
                )

            else:

                warmup_size = 0

            # --------------------------------
            # 6. Prepare warm-up data
            # --------------------------------

            warmup_data = (
                self.tester.get_test_with_warmup(
                    window,
                    warmup_size
                )
            )

            warmup_session = ResearchSession(
                data=warmup_data
            )

            all_signals = (
                strategy.generate_signals(
                    warmup_session.close
                )
            )

            # --------------------------------
            # 7. Create true test session
            # --------------------------------

            test_session = (
                self.tester.create_test_session(
                    window
                )
            )

            test_length = len(
                test_session.close
            )

            test_signals = all_signals[
                -test_length:
            ]

            # --------------------------------
            # 8. Carry capital forward
            # --------------------------------

            window_start_cash = (
                current_cash
            )

            test_session.engine.initial_cash = (
                window_start_cash
            )

            # --------------------------------
            # 9. Run OOS backtest
            # --------------------------------

            equity, trades = (
                test_session.engine.run(
                    test_session.close,
                    test_signals
                )
            )

            test_summary = Report.summary(
                equity,
                trades
            )

            current_cash = float(
                equity.iloc[-1]
            )

            # --------------------------------
            # 10. Date the equity curve
            # --------------------------------

            dated_equity = equity.copy()

            dated_equity.index = (
                test_session.close.index
            )

            combined_equity_parts.append(
                dated_equity
            )

            # --------------------------------
            # 11. Store window result
            # --------------------------------

            train_data = window[
                "train_data"
            ]

            test_data = window[
                "test_data"
            ]

            window_result = {

                "window": window_number,

                "train_start":
                    train_data.index[0],

                "train_end":
                    train_data.index[-1],

                "test_start":
                    test_data.index[0],

                "test_end":
                    test_data.index[-1],

                "parameters":
                    selected_parameters,

                "warmup_bars":
                    warmup_size,

                "starting_capital":
                    window_start_cash,

                "ending_capital":
                    current_cash,

                "training_return":
                    best["Return"],

                "training_sharpe":
                    best["Sharpe"],

                "training_drawdown":
                    best["Drawdown"],

                "training_trades":
                    int(best["Trades"]),

                "testing_return":
                    test_summary["return"],

                "testing_sharpe":
                    test_summary["sharpe"],

                "testing_drawdown":
                    test_summary["drawdown"],

                "testing_trades":
                    test_summary["trades"]
            }

            # Keep parameter names available
            # directly in the result as well.
            window_result.update(
                selected_parameters
            )

            window_results.append(
                window_result
            )

        # --------------------------------
        # 12. Combined OOS equity
        # --------------------------------

        if combined_equity_parts:

            combined_equity = pd.concat(
                combined_equity_parts
            )

            combined_equity.name = (
                "Equity"
            )

        else:

            combined_equity = pd.Series(
                dtype=float,
                name="Equity"
            )

        return (
            window_results,
            combined_equity
        )