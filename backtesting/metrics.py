import numpy as np
import pandas as pd

from backtesting.completed_trade import CompletedTrade


class Metrics:

    TRADING_DAYS = 252

    @staticmethod
    def total_return(initial: float, final: float) -> float:
        return ((final / initial) - 1) * 100

    @staticmethod
    def cagr(initial: float, final: float, periods: int) -> float:

        if periods <= 1:
            return 0.0

        years = periods / Metrics.TRADING_DAYS

        return ((final / initial) ** (1 / years) - 1) * 100

    @staticmethod
    def volatility(equity: pd.Series) -> float:

        returns = equity.pct_change().dropna()

        if returns.empty:
            return 0.0

        return returns.std() * np.sqrt(
            Metrics.TRADING_DAYS
        ) * 100

    @staticmethod
    def sharpe_ratio(
        equity: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:

        returns = equity.pct_change().dropna()

        if returns.empty:
            return 0.0

        excess_returns = (
            returns
            - risk_free_rate / Metrics.TRADING_DAYS
        )

        std = excess_returns.std()

        if std == 0:
            return 0.0

        return (
            excess_returns.mean()
            / std
        ) * np.sqrt(
            Metrics.TRADING_DAYS
        )

    @staticmethod
    def sortino_ratio(
        equity: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:

        returns = equity.pct_change().dropna()

        if returns.empty:
            return 0.0

        downside = returns[returns < 0]

        if downside.empty:
            return 0.0

        downside_std = downside.std()

        if downside_std == 0:
            return 0.0

        excess = (
            returns.mean()
            - risk_free_rate / Metrics.TRADING_DAYS
        )

        return (
            excess
            / downside_std
        ) * np.sqrt(
            Metrics.TRADING_DAYS
        )

    @staticmethod
    def max_drawdown(
        equity: pd.Series
    ) -> float:

        peak = equity.iloc[0]
        max_dd = 0.0

        for value in equity:

            if value > peak:
                peak = value

            drawdown = (
                peak - value
            ) / peak

            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd * 100

    @staticmethod
    def win_rate(
        trades: list[CompletedTrade]
    ) -> float:

        if not trades:
            return 0.0

        winners = sum(
            trade.profit > 0
            for trade in trades
        )

        return winners / len(trades) * 100

    @staticmethod
    def profit_factor(
        trades: list[CompletedTrade]
    ) -> float:

        gross_profit = sum(
            trade.profit
            for trade in trades
            if trade.profit > 0
        )

        gross_loss = abs(sum(
            trade.profit
            for trade in trades
            if trade.profit < 0
        ))

        # No trades or no profit
        if gross_profit == 0:
            return 0.0

        # Profitable trades with no losses
        if gross_loss == 0:
            return float("inf")

        return gross_profit / gross_loss

    @staticmethod
    def average_win(
        trades: list[CompletedTrade]
    ) -> float:

        winners = [
            trade.profit
            for trade in trades
            if trade.profit > 0
        ]

        if not winners:
            return 0.0

        return sum(winners) / len(winners)

    @staticmethod
    def average_loss(
        trades: list[CompletedTrade]
    ) -> float:

        losers = [
            trade.profit
            for trade in trades
            if trade.profit < 0
        ]

        if not losers:
            return 0.0

        return sum(losers) / len(losers)

    @staticmethod
    def best_trade(
        trades: list[CompletedTrade]
    ) -> float:

        if not trades:
            return 0.0

        return max(
            trade.profit
            for trade in trades
        )

    @staticmethod
    def worst_trade(
        trades: list[CompletedTrade]
    ) -> float:

        if not trades:
            return 0.0

        return min(
            trade.profit
            for trade in trades
        )