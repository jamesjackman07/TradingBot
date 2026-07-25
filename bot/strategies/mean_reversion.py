from bot.strategy import Strategy
from bot.indicators import Indicators


class MeanReversionStrategy(Strategy):

    def __init__(
        self,
        period=20,
        std=2
    ):

        self.period = period
        self.std = std

    def warmup_period(self):

        return self.period + 1

    def generate_signals(
        self,
        close
    ):

        upper, middle, lower = (
            Indicators.bollinger_bands(
                close,
                period=self.period,
                std=self.std
            )
        )

        signals = []

        for i in range(len(close)):

            # Bollinger Bands are not
            # available during warm-up.
            if (
                upper.iloc[i] != upper.iloc[i]
                or lower.iloc[i] != lower.iloc[i]
            ):

                signals.append("HOLD")
                continue

            if close.iloc[i] < lower.iloc[i]:

                signals.append("BUY")

            elif close.iloc[i] > upper.iloc[i]:

                signals.append("SELL")

            else:

                signals.append("HOLD")

        return signals