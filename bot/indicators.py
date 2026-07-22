import pandas as pd


class Indicators:

    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def bollinger_bands(data, period=20, std=2):
        sma = data.rolling(window=period).mean()
        standard_deviation = data.rolling(window=period).std()

        upper = sma + (standard_deviation * std)
        lower = sma - (standard_deviation * std)

        return upper, sma, lower