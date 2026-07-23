from backtesting.metrics import Metrics
from backtesting.completed_trade import CompletedTrade


class Report:

    @staticmethod
    def build_completed_trades(trades):

        completed = []

        buy_trade = None

        for trade in trades:

            if trade.trade_type == "BUY":
                buy_trade = trade

            elif (
                trade.trade_type == "SELL"
                and buy_trade is not None
            ):

                completed.append(

                    CompletedTrade(

                        entry_price=buy_trade.price,
                        exit_price=trade.price,

                        shares=buy_trade.shares,

                        entry_index=buy_trade.index,
                        exit_index=trade.index

                    )

                )

                buy_trade = None

        return completed

    @staticmethod
    def summary(equity, trades):

        completed = Report.build_completed_trades(
            trades
        )

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

            "completed_trades": completed,

            "trades": len(completed),

            "win_rate": Metrics.win_rate(
                completed
            ),

            "profit_factor": Metrics.profit_factor(
                completed
            ),

            "average_win": Metrics.average_win(
                completed
            ),

            "average_loss": Metrics.average_loss(
                completed
            ),

            "best_trade": Metrics.best_trade(
                completed
            ),

            "worst_trade": Metrics.worst_trade(
                completed
            )
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
            f"Completed Trades: {summary['trades']}"
        )

        print(
            f"Win Rate        : {summary['win_rate']:.2f}%"
        )

        print(
            f"Profit Factor   : {summary['profit_factor']:.2f}"
        )

        print()

        print(
            f"Average Win     : ${summary['average_win']:,.2f}"
        )

        print(
            f"Average Loss    : ${summary['average_loss']:,.2f}"
        )

        print(
            f"Best Trade      : ${summary['best_trade']:,.2f}"
        )

        print(
            f"Worst Trade     : ${summary['worst_trade']:,.2f}"
        )