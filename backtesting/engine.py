import pandas as pd

from backtesting.trade import Trade
from backtesting.portfolio import Portfolio


class BacktestEngine:

    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash

    def run(self, prices, signals):

        portfolio = Portfolio(self.initial_cash)

        equity_curve = []
        trades = []

        for i in range(len(prices)):

            price = prices.iloc[i]
            signal = signals[i]

            # BUY
            if signal == "BUY" and portfolio.shares == 0:

                portfolio.buy(price)

                trades.append(
                    Trade(
                        trade_type="BUY",
                        price=price,
                        index=i
                    )
                )

            # SELL
            elif signal == "SELL" and portfolio.shares > 0:

                portfolio.sell(price)

                trades.append(
                    Trade(
                        trade_type="SELL",
                        price=price,
                        index=i
                    )
                )

            equity = portfolio.equity(price)
            equity_curve.append(equity)

        return pd.DataFrame(
            {
                "Equity": equity_curve
            }
        ), trades