import matplotlib.pyplot as plt
import pandas as pd


class Plots:

    @staticmethod
    def equity_curve(equity: pd.Series) -> None:
        """
        Plots the portfolio equity curve.
        """

        plt.figure(figsize=(10, 5))

        plt.plot(
            equity,
            linewidth=2,
            label="Equity"
        )

        plt.title("Portfolio Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value ($)")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()