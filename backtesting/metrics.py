import numpy as np
import pandas as pd


class Metrics:

    TRADING_DAYS = 252

    @staticmethod
    def total_return(initial: float, final: float) -> float:
        """
        Calculates total percentage return.
        """
        return ((final / initial) - 1) * 100

    @staticmethod
    def cagr(
        initial: float,
        final: float,
        periods: int
    ) -> float:
        """
        Compound Annual Growth Rate.
        """

        if periods <= 1:
            return 0.0

        years = periods / Metrics.TRADING_DAYS

        return ((final / initial) ** (1 / years) - 1) * 100

    @staticmethod
    def volatility(equity: pd.Series) -> float:
        """
        Annualised volatility based on daily returns.
        """

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
        """
        Annualised Sharpe Ratio.
        """

        returns = equity.pct_change().dropna()

        if returns.empty:
            return 0.0

        excess_returns = (
            returns - risk_free_rate / Metrics.TRADING_DAYS
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
        """
        Annualised Sortino Ratio.
        """

        returns = equity.pct_change().dropna()

        if returns.empty:
            return 0.0

        downside = returns[returns < 0]

        if downside.empty:
            return 0.0

        downside_std = downside.std()

        if downside_std == 0:
            return 0.0

        excess_returns = (
            returns.mean()
            - risk_free_rate / Metrics.TRADING_DAYS
        )

        return (
            excess_returns
            / downside_std
        ) * np.sqrt(
            Metrics.TRADING_DAYS
        )

    @staticmethod
    def max_drawdown(
        equity: pd.Series
    ) -> float:
        """
        Maximum percentage drawdown.
        """

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