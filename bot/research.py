from bot.data import MarketData
from backtesting.engine import BacktestEngine


class ResearchSession:

    def __init__(self, ticker="SPY"):

        self.ticker = ticker

        market = MarketData()

        df = market.get_data(ticker)

        self.close = df["Close"][ticker]

        self.engine = BacktestEngine()

    def run(self, strategy):

        signals = strategy.generate_signals(self.close)

        equity, trades = self.engine.run(
            self.close,
            signals
        )

        return equity, trades