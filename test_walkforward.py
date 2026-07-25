from bot.data import MarketData

from walkforward.run import WalkForwardRunner

from backtesting.metrics import Metrics
from analytics.plots import Plots


# --------------------------------
# Download market data
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="15y"
)


# --------------------------------
# Create rolling walk-forward runner
# --------------------------------

runner = WalkForwardRunner(
    data=data,
    train_size=504,
    test_size=126,
    step_size=126,
    initial_cash=10000
)


# --------------------------------
# Parameter grid
# --------------------------------

parameters = {
    "fast": [5, 10, 20],
    "slow": [30, 50, 100]
}


# --------------------------------
# Run rolling walk-forward analysis
# --------------------------------

results, combined_equity = runner.run(
    parameters=parameters
)


# --------------------------------
# Print individual windows
# --------------------------------

print()
print("=" * 60)
print("ROLLING WALK-FORWARD REPORT")
print("=" * 60)


for result in results:

    print()
    print("-" * 60)
    print(
        f"WINDOW {result['window']}"
    )
    print("-" * 60)

    print()
    print("Periods")

    print(
        f"  Training : "
        f"{result['train_start']} "
        f"to {result['train_end']}"
    )

    print(
        f"  Testing  : "
        f"{result['test_start']} "
        f"to {result['test_end']}"
    )

    print()
    print("Selected Parameters")

    print(
        f"  Fast SMA : {result['fast']}"
    )

    print(
        f"  Slow SMA : {result['slow']}"
    )

    print()
    print("Training / In-Sample")

    print(
        f"  Return       : "
        f"{result['training_return']:.2f}%"
    )

    print(
        f"  Sharpe       : "
        f"{result['training_sharpe']:.2f}"
    )

    print(
        f"  Max Drawdown : "
        f"{result['training_drawdown']:.2f}%"
    )

    print(
        f"  Trades       : "
        f"{result['training_trades']}"
    )

    print()
    print("Testing / Out-of-Sample")

    print(
        f"  Start Capital: "
        f"${result['starting_capital']:,.2f}"
    )

    print(
        f"  End Capital  : "
        f"${result['ending_capital']:,.2f}"
    )

    print(
        f"  Return       : "
        f"{result['testing_return']:.2f}%"
    )

    print(
        f"  Sharpe       : "
        f"{result['testing_sharpe']:.2f}"
    )

    print(
        f"  Max Drawdown : "
        f"{result['testing_drawdown']:.2f}%"
    )

    print(
        f"  Trades       : "
        f"{result['testing_trades']}"
    )


# --------------------------------
# Combined OOS statistics
# --------------------------------

if not combined_equity.empty:

    initial_capital = runner.initial_cash

    final_capital = float(
        combined_equity.iloc[-1]
    )

    combined_return = Metrics.total_return(
        initial_capital,
        final_capital
    )

    combined_cagr = Metrics.cagr(
        initial_capital,
        final_capital,
        len(combined_equity)
    )

    combined_volatility = Metrics.volatility(
        combined_equity
    )

    combined_sharpe = Metrics.sharpe_ratio(
        combined_equity
    )

    combined_sortino = Metrics.sortino_ratio(
        combined_equity
    )

    combined_drawdown = Metrics.max_drawdown(
        combined_equity
    )

    total_trades = sum(
        result["testing_trades"]
        for result in results
    )

    profitable_windows = sum(
        result["testing_return"] > 0
        for result in results
    )

    losing_windows = sum(
        result["testing_return"] < 0
        for result in results
    )

    flat_windows = sum(
        result["testing_return"] == 0
        for result in results
    )

    print()
    print("=" * 60)
    print("COMBINED OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 60)

    print()

    print(
        f"Initial Capital     : "
        f"${initial_capital:,.2f}"
    )

    print(
        f"Final Capital       : "
        f"${final_capital:,.2f}"
    )

    print()

    print(
        f"Combined OOS Return : "
        f"{combined_return:.2f}%"
    )

    print(
        f"OOS CAGR            : "
        f"{combined_cagr:.2f}%"
    )

    print(
        f"OOS Volatility      : "
        f"{combined_volatility:.2f}%"
    )

    print(
        f"OOS Sharpe Ratio    : "
        f"{combined_sharpe:.2f}"
    )

    print(
        f"OOS Sortino Ratio   : "
        f"{combined_sortino:.2f}"
    )

    print(
        f"OOS Max Drawdown    : "
        f"{combined_drawdown:.2f}%"
    )

    print()

    print(
        f"Windows             : "
        f"{len(results)}"
    )

    print(
        f"Profitable Windows  : "
        f"{profitable_windows}"
    )

    print(
        f"Losing Windows      : "
        f"{losing_windows}"
    )

    print(
        f"Flat Windows        : "
        f"{flat_windows}"
    )

    print(
        f"Total OOS Trades    : "
        f"{total_trades}"
    )

    print()
    print("=" * 60)


# --------------------------------
# Buy-and-hold benchmark
# --------------------------------

if not combined_equity.empty:

    # Use the exact same dates as the
    # walk-forward out-of-sample period.
    benchmark_prices = data.loc[
        combined_equity.index,
        "Close"
    ]

    # yfinance may return Close as a
    # one-column DataFrame.
    if hasattr(
        benchmark_prices,
        "columns"
    ):

        if len(
            benchmark_prices.columns
        ) == 1:

            benchmark_prices = (
                benchmark_prices.iloc[:, 0]
            )

    benchmark_prices = (
        benchmark_prices
        .dropna()
        .astype(float)
    )

    # --------------------------------
    # Build benchmark equity curve
    # --------------------------------

    benchmark_equity = (
        benchmark_prices
        / benchmark_prices.iloc[0]
        * runner.initial_cash
    )

    benchmark_equity.name = (
        "Buy & Hold"
    )

    # --------------------------------
    # Benchmark metrics
    # --------------------------------

    benchmark_final = float(
        benchmark_equity.iloc[-1]
    )

    benchmark_return = (
        Metrics.total_return(
            runner.initial_cash,
            benchmark_final
        )
    )

    benchmark_cagr = Metrics.cagr(
        runner.initial_cash,
        benchmark_final,
        len(benchmark_equity)
    )

    benchmark_volatility = (
        Metrics.volatility(
            benchmark_equity
        )
    )

    benchmark_sharpe = (
        Metrics.sharpe_ratio(
            benchmark_equity
        )
    )

    benchmark_sortino = (
        Metrics.sortino_ratio(
            benchmark_equity
        )
    )

    benchmark_drawdown = (
        Metrics.max_drawdown(
            benchmark_equity
        )
    )

    # --------------------------------
    # Print comparison
    # --------------------------------

    print()
    print("=" * 70)
    print(
        "STRATEGY VS BUY-AND-HOLD"
    )
    print("=" * 70)

    print()

    print(
        f"{'Metric':<22}"
        f"{'Strategy':>18}"
        f"{'Buy & Hold':>18}"
    )

    print("-" * 58)

    print(
        f"{'Final Capital':<22}"
        f"${final_capital:>17,.2f}"
        f"${benchmark_final:>17,.2f}"
    )

    print(
        f"{'Total Return':<22}"
        f"{combined_return:>17.2f}%"
        f"{benchmark_return:>17.2f}%"
    )

    print(
        f"{'CAGR':<22}"
        f"{combined_cagr:>17.2f}%"
        f"{benchmark_cagr:>17.2f}%"
    )

    print(
        f"{'Volatility':<22}"
        f"{combined_volatility:>17.2f}%"
        f"{benchmark_volatility:>17.2f}%"
    )

    print(
        f"{'Sharpe Ratio':<22}"
        f"{combined_sharpe:>18.2f}"
        f"{benchmark_sharpe:>18.2f}"
    )

    print(
        f"{'Sortino Ratio':<22}"
        f"{combined_sortino:>18.2f}"
        f"{benchmark_sortino:>18.2f}"
    )

    print(
        f"{'Max Drawdown':<22}"
        f"{combined_drawdown:>17.2f}%"
        f"{benchmark_drawdown:>17.2f}%"
    )

    print()
    print("=" * 70)


# --------------------------------
# Plot combined OOS equity curve
# --------------------------------

if not combined_equity.empty:

    Plots.equity_curve(
        combined_equity
    )