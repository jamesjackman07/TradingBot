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

    def run(
        self,
        data,
        signals
    ):

        # --------------------------------
        # Support both legacy Close Series
        # and new OHLC DataFrame input.
        # --------------------------------

        if isinstance(
            data,
            pd.DataFrame
        ):

            close_prices = data["Close"]

            if "Open" in data.columns:
                open_prices = data["Open"]
            else:
                open_prices = close_prices

        else:

            close_prices = data
            open_prices = data

        portfolio = Portfolio(
            self.initial_cash
        )

        equity_curve = []
        trades = []

        entry_price = None

        # Signal generated on bar i - 1
        # is executed on bar i.
        pending_signal = None

        for i in range(
            len(close_prices)
        ):

            open_price = float(
                open_prices.iloc[i]
            )

            close_price = float(
                close_prices.iloc[i]
            )

            # =================================
            # 1. Execute previous bar's signal
            # =================================

            if pending_signal == "BUY":

                if not portfolio.has_position():

                    buy_price = (
                        self.risk.buy_price(
                            open_price
                        )
                    )

                    shares = (
                        self.risk.shares_to_buy(
                            portfolio.cash,
                            buy_price
                        )
                    )

                    portfolio.buy(
                        buy_price,
                        shares,
                        self.risk.commission
                    )

                    entry_price = buy_price

                    trades.append(
                        Trade(
                            trade_type="BUY",
                            price=buy_price,
                            index=i,
                            shares=shares
                        )
                    )

            elif pending_signal == "SELL":

                if portfolio.has_position():

                    sell_price = (
                        self.risk.sell_price(
                            open_price
                        )
                    )

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
                            shares=shares,
                            reason="SIGNAL"
                        )
                    )

                    entry_price = None

            # =================================
            # 2. Stop-loss / take-profit
            # =================================
            #
            # Still close-price based for now.
            # We will replace this with OHLC
            # intrabar handling separately.
            # =================================

            if portfolio.has_position():

                stop_price = (
                    self.risk.stop_price(
                        entry_price
                    )
                )

                target_price = (
                    self.risk.target_price(
                        entry_price
                    )
                )

                exit_trade = False
                exit_reason = None

                if (
                    stop_price is not None
                    and close_price <= stop_price
                ):

                    exit_trade = True
                    exit_reason = "STOP_LOSS"

                elif (
                    target_price is not None
                    and close_price >= target_price
                ):

                    exit_trade = True
                    exit_reason = "TAKE_PROFIT"

                if exit_trade:

                    sell_price = (
                        self.risk.sell_price(
                            close_price
                        )
                    )

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
                            shares=shares,
                            reason=exit_reason
                        )
                    )

                    entry_price = None

            # =================================
            # 3. Mark portfolio to market
            # =================================

            equity_curve.append(
                portfolio.equity(
                    close_price
                )
            )

            # =================================
            # 4. Signal becomes actionable only
            #    on the NEXT bar
            # =================================

            pending_signal = signals[i]

        # =====================================
        # Close any remaining open position
        # at the final available close.
        # =====================================

        if portfolio.has_position():

            final_price = (
                self.risk.sell_price(
                    float(
                        close_prices.iloc[-1]
                    )
                )
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
                    index=len(
                        close_prices
                    ) - 1,
                    shares=shares,
                    reason="END_OF_DATA"
                )
            )

            equity_curve[-1] = (
                portfolio.equity(
                    final_price
                )
            )

        equity = pd.Series(
            equity_curve,
            name="Equity"
        )

        return equity, trades