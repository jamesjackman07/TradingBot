from bot.data import MarketData

from backtesting.engine import BacktestEngine
from risk.manager import RiskManager


class ResearchSession:

    def __init__(
        self,
        ticker="SPY",
        risk_manager=None,
        data=None
    ):

        if data is None:

            market = MarketData()

            data = market.get_data(
                ticker
            )

        # --------------------------------
        # Flatten yfinance single-ticker
        # columns when necessary
        # --------------------------------

        data = data.copy()

        if hasattr(
            data.columns,
            "nlevels"
        ):

            if data.columns.nlevels > 1:

                if ticker in data.columns.get_level_values(
                    -1
                ):

                    data = data.xs(
                        ticker,
                        axis=1,
                        level=-1
                    )

                elif len(
                    data.columns.get_level_values(
                        -1
                    ).unique()
                ) == 1:

                    data.columns = (
                        data.columns.get_level_values(
                            0
                        )
                    )

        # --------------------------------
        # Remove rows without Close
        # --------------------------------

        data = data.dropna(
            subset=["Close"]
        )

        self.data = data

        # Strategies still operate on Close.
        self.close = data[
            "Close"
        ]

        self.engine = BacktestEngine(
            risk_manager=(
                risk_manager
                or RiskManager()
            )
        )

    def run(
        self,
        strategy
    ):

        signals = (
            strategy.generate_signals(
                self.close
            )
        )

        # Keep existing engine behaviour
        # for now.
        return self.engine.run(
            self.close,
            signals
        )