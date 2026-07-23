from itertools import product
import pandas as pd

from analytics.report import Report
from research.results import ResearchResults


class Optimizer:

    def __init__(self, session):
        self.session = session

    def optimize(
        self,
        strategy_class,
        parameters,
        sort_by="Return",
        ascending=False
    ):

        parameter_names = list(parameters.keys())
        parameter_values = list(parameters.values())

        results = []

        for values in product(*parameter_values):

            strategy_parameters = dict(
                zip(parameter_names, values)
            )

            strategy = strategy_class(
                **strategy_parameters
            )

            equity, trades = self.session.run(
                strategy
            )

            summary = Report.summary(
                equity,
                trades
            )

            result = strategy_parameters.copy()

            result["Return"] = summary["return"]
            result["Sharpe"] = summary["sharpe"]
            result["Drawdown"] = summary["drawdown"]
            result["Profit Factor"] = summary["profit_factor"]
            result["Win Rate"] = summary["win_rate"]
            result["Trades"] = summary["trades"]

            results.append(result)

        dataframe = pd.DataFrame(results)

        if sort_by not in dataframe.columns:
            raise ValueError(
                f"Unknown metric '{sort_by}'. "
                f"Available metrics: {list(dataframe.columns)}"
            )

        dataframe.sort_values(
            by=sort_by,
            ascending=ascending,
            inplace=True
        )

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        return ResearchResults(
            dataframe
        )