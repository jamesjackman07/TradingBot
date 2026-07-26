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
        # Market data
        # --------------------------------

        has_ohlc = isinstance(
            data,
            pd.DataFrame
        )

        if has_ohlc:

            close_prices = data["Close"]

            open_prices = (
                data["Open"]
                if "Open" in data.columns
                else close_prices
            )

            high_prices = (
                data["High"]
                if "High" in data.columns
                else close_prices
            )

            low_prices = (
                data["Low"]
                if "Low" in data.columns
                else close_prices
            )

        else:

            # Legacy close-only support.
            open_prices = data
            high_prices = data
            low_prices = data
            close_prices = data

        portfolio = Portfolio(
            self.initial_cash
        )

        equity_curve = []
        trades = []

        entry_price = None
        pending_signal = None

        for i in range(
            len(close_prices)
        ):

            open_price = float(
                open_prices.iloc[i]
            )

            high_price = float(
                high_prices.iloc[i]
            )

            low_price = float(
                low_prices.iloc[i]
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

                exit_price = None
                exit_reason = None

                # -----------------------------
                # Stop-loss
                # -----------------------------
                #
                # Stop is deliberately checked
                # first. If both stop and target
                # occur inside one OHLC bar, we
                # use the conservative outcome.
                # -----------------------------

                if stop_price is not None:

                    # Market gapped below stop.
                    if open_price <= stop_price:

                        exit_price = open_price
                        exit_reason = (
                            "STOP_LOSS"
                        )

                    # Stop touched intrabar.
                    elif low_price <= stop_price:

                        exit_price = stop_price
                        exit_reason = (
                            "STOP_LOSS"
                        )

                # -----------------------------
                # Take-profit
                # -----------------------------

                if (
                    exit_reason is None
                    and target_price is not None
                ):

                    # Market gapped above target.
                    if open_price >= target_price:

                        exit_price = open_price
                        exit_reason = (
                            "TAKE_PROFIT"
                        )

                    # Target touched intrabar.
                    elif high_price >= target_price:

                        exit_price = target_price
                        exit_reason = (
                            "TAKE_PROFIT"
                        )

                # -----------------------------
                # Execute protective exit
                # -----------------------------

                if exit_reason is not None:

                    sell_price = (
                        self.risk.sell_price(
                            exit_price
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
            # 4. Store current signal
            # =================================

            pending_signal = signals[i]

        # =====================================
        # 5. End-of-data liquidation
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
                    index=(
                        len(close_prices) - 1
                    ),
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