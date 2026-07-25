import pandas as pd

from optimization.optimizer import Optimizer
from bot.strategies.sma_cross import SMACrossoverStrategy
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
        initial_cash=10000
    ):

        self.tester = WalkForwardTester(
            data=data,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size
        )

        self.initial_cash = initial_cash

    def run(
        self,
        parameters
    ):

        window_results = []

        combined_equity_parts = []

        current_cash = self.initial_cash

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
            # 2. Optimise on training data only
            # --------------------------------

            optimizer = Optimizer(
                train_session
            )

            results = optimizer.optimize(
                strategy_class=SMACrossoverStrategy,
                parameters=parameters,
                constraint=lambda p: (
                    p["fast"] < p["slow"]
                )
            )

            best = results.dataframe.iloc[0]

            fast = int(best["fast"])
            slow = int(best["slow"])

            # --------------------------------
            # 3. Freeze best parameters
            # --------------------------------

            strategy = SMACrossoverStrategy(
                fast=fast,
                slow=slow
            )

            # --------------------------------
            # 4. Prepare indicator warm-up
            # --------------------------------

            warmup_size = slow + 1

            warmup_data = (
                self.tester.get_test_with_warmup(
                    window,
                    warmup_size
                )
            )

            warmup_session = ResearchSession(
                data=warmup_data
            )

            all_signals = strategy.generate_signals(
                warmup_session.close
            )

            # --------------------------------
            # 5. Create true test session
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
            # 6. Carry OOS capital forward
            # --------------------------------

            window_start_cash = current_cash

            test_session.engine.initial_cash = (
                window_start_cash
            )

            # --------------------------------
            # 7. Backtest test period only
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
            # 8. Add dates to equity curve
            # --------------------------------

            dated_equity = equity.copy()

            dated_equity.index = (
                test_session.close.index
            )

            combined_equity_parts.append(
                dated_equity
            )

            # --------------------------------
            # 9. Store window information
            # --------------------------------

            train_data = window["train_data"]
            test_data = window["test_data"]

            window_results.append({

                "window": window_number,

                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],

                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],

                "fast": fast,
                "slow": slow,

                "warmup_bars": warmup_size,

                "starting_capital": (
                    window_start_cash
                ),

                "ending_capital": (
                    current_cash
                ),

                "training_return": best["Return"],
                "training_sharpe": best["Sharpe"],
                "training_drawdown": best["Drawdown"],
                "training_trades": int(
                    best["Trades"]
                ),

                "testing_return": test_summary[
                    "return"
                ],

                "testing_sharpe": test_summary[
                    "sharpe"
                ],

                "testing_drawdown": test_summary[
                    "drawdown"
                ],

                "testing_trades": test_summary[
                    "trades"
                ]
            })

        # --------------------------------
        # 10. Build continuous OOS equity
        # --------------------------------

        if combined_equity_parts:

            combined_equity = pd.concat(
                combined_equity_parts
            )

            combined_equity.name = "Equity"

        else:

            combined_equity = pd.Series(
                dtype=float,
                name="Equity"
            )

        return window_results, combined_equity