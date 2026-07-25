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
        ascending=False,
        constraint=None
    ):

        parameter_names = list(parameters.keys())
        parameter_values = list(parameters.values())

        results = []

        for values in product(*parameter_values):

            current_parameters = dict(
                zip(parameter_names, values)
            )

            if (
                constraint is not None
                and not constraint(current_parameters)
            ):
                continue

            strategy = strategy_class(
                **current_parameters
            )

            equity, trades = self.session.run(
                strategy
            )

            summary = Report.summary(
                equity,
                trades
            )

            results.append({

                **current_parameters,

                "Return": summary["return"],
                "Sharpe": summary["sharpe"],
                "Drawdown": summary["drawdown"],
                "Profit Factor": summary["profit_factor"],
                "Win Rate": summary["win_rate"],
                "Trades": summary["trades"]

            })

        dataframe = pd.DataFrame(results)

        if sort_by not in dataframe.columns:
            raise ValueError(
                f"Unknown metric '{sort_by}'"
            )

        dataframe = dataframe.sort_values(
            by=sort_by,
            ascending=ascending
        ).reset_index(drop=True)

        return ResearchResults(dataframe)