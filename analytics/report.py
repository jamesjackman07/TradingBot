from backtesting.metrics import Metrics


class Report:

    @staticmethod
    def summary(equity, trades):

        initial = equity.iloc[0]
        final = equity.iloc[-1]

        return {
            "initial": initial,
            "final": final,
            "return": Metrics.total_return(initial, final),
            "drawdown": Metrics.max_drawdown(equity),
            "trades": len(trades)
        }

    @staticmethod
    def print(summary):

        print("\n========================================")
        print("BACKTEST REPORT")
        print("========================================")
        print(f"Initial Capital : ${summary['initial']:,.2f}")
        print(f"Final Capital   : ${summary['final']:,.2f}")
        print(f"Return           : {summary['return']:.2f}%")
        print(f"Drawdown         : {summary['drawdown']:.2f}%")
        print(f"Trades           : {summary['trades']}")