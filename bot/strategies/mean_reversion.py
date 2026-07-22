from bot.strategy import Strategy
from bot.indicators import Indicators


class MeanReversionStrategy(Strategy):

    def generate_signals(self, close):

        upper, middle, lower = Indicators.bollinger_bands(close)

        signals = []

        for i in range(len(close)):

            if close.iloc[i] < lower.iloc[i]:
                signals.append("BUY")

            elif close.iloc[i] > upper.iloc[i]:
                signals.append("SELL")

            else:
                signals.append("HOLD")

        return signals