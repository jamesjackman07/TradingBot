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

        self.data = data

        close = data["Close"]

        # yfinance can return Close as a DataFrame
        # even when only one ticker is requested.
        if hasattr(close, "columns"):

            if ticker in close.columns:
                close = close[ticker]

            elif len(close.columns) == 1:
                close = close.iloc[:, 0]

        # Remove missing prices before generating signals
        # or running the backtest.
        close = close.dropna()

        self.close = close

        self.engine = BacktestEngine(
            risk_manager=risk_manager or RiskManager()
        )

    def run(
        self,
        strategy
    ):

        signals = strategy.generate_signals(
            self.close
        )

        return self.engine.run(
            self.close,
            signals
        )