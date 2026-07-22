import matplotlib.pyplot as plt


class Plots:

    @staticmethod
    def equity_curve(equity):

        plt.figure(figsize=(12,6))

        plt.plot(equity["Equity"])

        plt.title("Portfolio Equity")

        plt.grid(True)

        plt.show()