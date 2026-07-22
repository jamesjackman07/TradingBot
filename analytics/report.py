from backtesting.metrics import Metrics


class Report:

    @staticmethod
    def summary(equity, trades):

        final = equity["Equity"].iloc[-1]

        return {
            "Initial Capital": 10000,
            "Final Capital": final,
            "Return": Metrics.total_return(10000, final),
            "Drawdown": Metrics.max_drawdown(
                equity["Equity"]
            ),
            "Trades": len(trades)
        }

    @staticmethod
    def print(summary):

        print()

        print("=" * 40)
        print("BACKTEST REPORT")
        print("=" * 40)

        print(
            f"Initial Capital : "
            f"${summary['Initial Capital']:,.2f}"
        )

        print(
            f"Final Capital   : "
            f"${summary['Final Capital']:,.2f}"
        )

        print(
            f"Return           : "
            f"{summary['Return']:.2f}%"
        )

        print(
            f"Drawdown         : "
            f"{summary['Drawdown']:.2f}%"
        )

        print(
            f"Trades           : "
            f"{summary['Trades']}"
        )