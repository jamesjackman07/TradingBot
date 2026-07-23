from backtesting.metrics import Metrics


class Report:

    @staticmethod
    def summary(equity, trades):

        initial = equity.iloc[0]
        final = equity.iloc[-1]

        return {
            "initial": initial,
            "final": final,

            "return": Metrics.total_return(
                initial,
                final
            ),

            "cagr": Metrics.cagr(
                initial,
                final,
                len(equity)
            ),

            "volatility": Metrics.volatility(
                equity
            ),

            "sharpe": Metrics.sharpe_ratio(
                equity
            ),

            "sortino": Metrics.sortino_ratio(
                equity
            ),

            "drawdown": Metrics.max_drawdown(
                equity
            ),

            "trades": len(trades)
        }

    @staticmethod
    def print(summary):

        print()
        print("=" * 40)
        print("BACKTEST REPORT")
        print("=" * 40)

        print(
            f"Initial Capital : ${summary['initial']:,.2f}"
        )

        print(
            f"Final Capital   : ${summary['final']:,.2f}"
        )

        print()

        print(
            f"Total Return    : {summary['return']:.2f}%"
        )

        print(
            f"CAGR            : {summary['cagr']:.2f}%"
        )

        print(
            f"Volatility      : {summary['volatility']:.2f}%"
        )

        print(
            f"Sharpe Ratio    : {summary['sharpe']:.2f}"
        )

        print(
            f"Sortino Ratio   : {summary['sortino']:.2f}"
        )

        print()

        print(
            f"Max Drawdown    : {summary['drawdown']:.2f}%"
        )

        print(
            f"Trades          : {summary['trades']}"
        )