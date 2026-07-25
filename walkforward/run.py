from optimization.optimizer import Optimizer
from bot.strategies.sma_cross import SMACrossoverStrategy

from analytics.report import Report

from walkforward.tester import WalkForwardTester


class WalkForwardRunner:

    def __init__(
        self,
        data,
        train_size=504,
        test_size=126,
        step_size=126
    ):

        self.tester = WalkForwardTester(
            data=data,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size
        )

    def run(
        self,
        parameters
    ):

        window_results = []

        for window_number, window in enumerate(
            self.tester.windows(),
            start=1
        ):

            train_session, test_session = (
                self.tester.create_sessions(
                    window
                )
            )

            # --------------------------------
            # 1. Optimise on training data
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
            # 2. Freeze best parameters
            # --------------------------------

            strategy = SMACrossoverStrategy(
                fast=fast,
                slow=slow
            )

            # --------------------------------
            # 3. Run on unseen test data
            # --------------------------------

            equity, trades = test_session.run(
                strategy
            )

            test_summary = Report.summary(
                equity,
                trades
            )

            # --------------------------------
            # 4. Store window information
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

        return window_results