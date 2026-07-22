from bot.strategy import Strategy
from bot.indicators import Indicators


class SMACrossoverStrategy(Strategy):

    def __init__(self, fast=20, slow=50):

        self.fast = fast
        self.slow = slow

    def generate_signals(self, close):

        fast_sma = Indicators.sma(close, self.fast)
        slow_sma = Indicators.sma(close, self.slow)

        signals = []

        for i in range(len(close)):

            if i == 0:
                signals.append("HOLD")
                continue

            previous_fast = fast_sma.iloc[i - 1]
            previous_slow = slow_sma.iloc[i - 1]

            current_fast = fast_sma.iloc[i]
            current_slow = slow_sma.iloc[i]

            if (
                previous_fast <= previous_slow
                and current_fast > current_slow
            ):
                signals.append("BUY")

            elif (
                previous_fast >= previous_slow
                and current_fast < current_slow
            ):
                signals.append("SELL")

            else:
                signals.append("HOLD")

        return signals