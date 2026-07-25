from bot.strategy import Strategy


class MomentumStrategy(Strategy):

    def __init__(
        self,
        lookback=60
    ):

        self.lookback = lookback

    def warmup_period(self):

        return self.lookback + 1

    def generate_signals(
        self,
        close
    ):

        momentum = close.pct_change(
            periods=self.lookback
        )

        signals = []

        in_position = False

        for i in range(len(close)):

            value = momentum.iloc[i]

            # Not enough historical data yet
            if value != value:
                signals.append("HOLD")
                continue

            # Positive momentum
            if (
                value > 0
                and not in_position
            ):

                signals.append("BUY")
                in_position = True

            # Negative momentum
            elif (
                value <= 0
                and in_position
            ):

                signals.append("SELL")
                in_position = False

            else:

                signals.append("HOLD")

        return signals