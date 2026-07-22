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
            if signal == "BUY" and not portfolio.has_position():

                portfolio.buy(price)

                trades.append(
                    Trade(
                        trade_type="BUY",
                        price=price,
                        index=i,
                        shares=portfolio.shares
                    )
                )

            # SELL
            elif signal == "SELL" and portfolio.has_position():

                shares = portfolio.shares
                portfolio.sell(price)

                trades.append(
                    Trade(
                        trade_type="SELL",
                        price=price,
                        index=i,
                        shares=shares
                    )
                )

            equity = portfolio.equity(price)
            equity_curve.append(equity)

        if portfolio.has_position():

            final_price = prices.iloc[-1]

            shares = portfolio.shares

            portfolio.sell(final_price)

            trades.append(
                Trade(
                    trade_type="SELL",
                    price=final_price,
                    index=len(prices) - 1,
                    shares=shares
                )
            )

            equity_curve[-1] = portfolio.equity(final_price)

        return pd.DataFrame(
            {
                "Equity": equity_curve
            }
        ), trades