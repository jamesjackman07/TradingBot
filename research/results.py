import pandas as pd

from research.result import ResearchResult


class ResearchResults:

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __repr__(self):
        return repr(self.dataframe)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return ResearchResult(
            self.dataframe.iloc[item]
        )

    def sort(
        self,
        metric,
        ascending=False
    ):

        self.dataframe = self.dataframe.sort_values(
            by=metric,
            ascending=ascending
        ).reset_index(drop=True)

        return self

    def best(self):

        if self.dataframe.empty:
            return None

        return ResearchResult(
            self.dataframe.iloc[0]
        )

    def head(self, n=5):
        return self.dataframe.head(n)

    def tail(self, n=5):
        return self.dataframe.tail(n)

    def to_dataframe(self):
        return self.dataframe.copy()

    def export_csv(
        self,
        filename="optimization_results.csv"
    ):

        self.dataframe.to_csv(
            filename,
            index=False
        )

        print(
            f"Results exported to '{filename}'"
        )