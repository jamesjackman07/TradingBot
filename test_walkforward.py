from bot.data import MarketData
from walkforward.run import WalkForwardRunner


# --------------------------------
# Download market data
# --------------------------------

market = MarketData()

data = market.get_data(
    "SPY",
    period="5y"
)


# --------------------------------
# Create rolling walk-forward runner
# --------------------------------

runner = WalkForwardRunner(
    data=data,
    train_size=504,
    test_size=126,
    step_size=126
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

results = runner.run(
    parameters=parameters
)


# --------------------------------
# Print report
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
# Aggregate statistics
# --------------------------------

if results:

    testing_returns = [
        result["testing_return"]
        for result in results
    ]

    testing_sharpes = [
        result["testing_sharpe"]
        for result in results
    ]

    testing_drawdowns = [
        result["testing_drawdown"]
        for result in results
    ]

    testing_trades = [
        result["testing_trades"]
        for result in results
    ]

    profitable_windows = sum(
        value > 0
        for value in testing_returns
    )

    losing_windows = sum(
        value < 0
        for value in testing_returns
    )

    flat_windows = sum(
        value == 0
        for value in testing_returns
    )

    average_return = (
        sum(testing_returns)
        / len(testing_returns)
    )

    average_sharpe = (
        sum(testing_sharpes)
        / len(testing_sharpes)
    )

    worst_drawdown = max(
        testing_drawdowns
    )

    total_trades = sum(
        testing_trades
    )

    print()
    print("=" * 60)
    print("OUT-OF-SAMPLE SUMMARY")
    print("=" * 60)

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

    print()

    print(
        f"Average OOS Return  : "
        f"{average_return:.2f}%"
    )

    print(
        f"Average OOS Sharpe  : "
        f"{average_sharpe:.2f}"
    )

    print(
        f"Worst OOS Drawdown  : "
        f"{worst_drawdown:.2f}%"
    )

    print(
        f"Total OOS Trades    : "
        f"{total_trades}"
    )

print()
print("=" * 60)