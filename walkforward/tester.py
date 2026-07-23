import pandas as pd

from optimization.optimizer import Optimizer
from analytics.report import Report


class WalkForwardTester:

    def __init__(self, session):
        self.session = session

    def split_data(self, data, train_ratio=0.7):
        """
        Split a DataFrame into train and test sets.
        """
        split = int(len(data) * train_ratio)

        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()

        return train, test