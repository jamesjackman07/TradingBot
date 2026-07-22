import yfinance as yf
import pandas as pd


class MarketData:

    def __init__(self):
        pass

    def get_data(self, ticker, period="6mo", interval="1d"):
        """
        Downloads historical market data.

        ticker examples:
            SPY
            QQQ
            BTC-USD
            GLD
        """

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True
        )

        return df