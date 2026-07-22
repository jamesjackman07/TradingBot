import pandas as pd

from backtesting.trade import Trade


class BacktestEngine:

    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash

    def run(self, prices, signals):

        cash = self.initial_cash
        shares = 0

        equity_curve = []
        trades = []

        for i in range(len(prices)):

            price = prices.iloc[i]
            signal = signals[i]

            # BUY
            if signal == "BUY" and shares == 0:

                shares = cash / price
                cash = 0

                trades.append(
                    Trade(
                        trade_type="BUY",
                        price=price,
                        index=i
                    )
                )

            # SELL
            elif signal == "SELL" and shares > 0:

                cash = shares * price
                shares = 0

                trades.append(
                    Trade(
                        trade_type="SELL",
                        price=price,
                        index=i
                    )
                )

            equity = cash + shares * price
            equity_curve.append(equity)

        return pd.DataFrame(
            {
                "Equity": equity_curve
            }
        ), trades