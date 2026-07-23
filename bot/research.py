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
        """
        If data is provided, use it.
        Otherwise download data for the ticker.
        """

        if data is None:
            market = MarketData()
            data = market.get_data(ticker)

        self.data = data

        if hasattr(data["Close"], "columns"):
            self.close = data["Close"][ticker]
        else:
            self.close = data["Close"]

        self.engine = BacktestEngine(
            risk_manager=risk_manager or RiskManager()
        )

    def run(self, strategy):

        signals = strategy.generate_signals(
            self.close
        )

        return self.engine.run(
            self.close,
            signals
        )