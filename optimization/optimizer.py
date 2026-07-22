from itertools import product

import pandas as pd

from analytics.report import Report


class Optimizer:

    def __init__(self, session):

        self.session = session

    def optimize(self, strategy_class, parameters):

        parameter_names = list(parameters.keys())
        parameter_values = list(parameters.values())

        results = []

        for values in product(*parameter_values):

            strategy_parameters = dict(zip(parameter_names, values))

            strategy = strategy_class(**strategy_parameters)

            equity, trades = self.session.run(strategy)

            summary = Report.summary(equity, trades)

            result = strategy_parameters.copy()

            result["Return"] = summary["return"]
            result["Drawdown"] = summary["drawdown"]
            result["Trades"] = summary["trades"]

            results.append(result)

        dataframe = pd.DataFrame(results)

        dataframe.sort_values(
            by="Return",
            ascending=False,
            inplace=True
        )

        dataframe.reset_index(drop=True, inplace=True)

        return dataframe