import pandas as pd

from backtesting.trade import Trade
from backtesting.portfolio import Portfolio

from risk.manager import RiskManager


class BacktestEngine:

    def __init__(
        self,
        initial_cash=10000,
        risk_manager=None
    ):

        self.initial_cash = initial_cash

        self.risk = risk_manager or RiskManager()

    def run(self, prices, signals):

        portfolio = Portfolio(self.initial_cash)

        equity_curve = []

        trades = []

        for i in range(len(prices)):

            price = prices.iloc[i]
            signal = signals[i]

            if signal == "BUY" and not portfolio.has_position():

                buy_price = self.risk.buy_price(price)

                shares = self.risk.shares_to_buy(
                    portfolio.cash,
                    buy_price
                )

                portfolio.buy(
                    buy_price,
                    shares,
                    self.risk.commission
                )

                trades.append(
                    Trade(
                        trade_type="BUY",
                        price=buy_price,
                        index=i,
                        shares=shares
                    )
                )

            elif signal == "SELL" and portfolio.has_position():

                sell_price = self.risk.sell_price(price)

                shares = portfolio.shares

                portfolio.sell(
                    sell_price,
                    self.risk.commission
                )

                trades.append(
                    Trade(
                        trade_type="SELL",
                        price=sell_price,
                        index=i,
                        shares=shares
                    )
                )

            equity_curve.append(
                portfolio.equity(price)
            )

        if portfolio.has_position():

            final_price = self.risk.sell_price(
                prices.iloc[-1]
            )

            shares = portfolio.shares

            portfolio.sell(
                final_price,
                self.risk.commission
            )

            trades.append(
                Trade(
                    trade_type="SELL",
                    price=final_price,
                    index=len(prices) - 1,
                    shares=shares
                )
            )

            equity_curve[-1] = portfolio.equity(final_price)

        equity = pd.Series(
            equity_curve,
            name="Equity"
        )

        return equity, trades