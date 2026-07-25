from bot.research import ResearchSession


class WalkForwardTester:

    def __init__(
        self,
        data,
        train_size=504,
        test_size=126,
        step_size=126
    ):
        self.data = data

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

        self._validate_settings()

    def _validate_settings(self):

        if self.train_size <= 0:
            raise ValueError(
                "train_size must be greater than 0"
            )

        if self.test_size <= 0:
            raise ValueError(
                "test_size must be greater than 0"
            )

        if self.step_size <= 0:
            raise ValueError(
                "step_size must be greater than 0"
            )

        if len(self.data) < (
            self.train_size + self.test_size
        ):
            raise ValueError(
                "Not enough data for one "
                "walk-forward window"
            )

    def windows(self):

        start = 0

        while True:

            train_start = start
            train_end = (
                train_start + self.train_size
            )

            test_start = train_end
            test_end = (
                test_start + self.test_size
            )

            if test_end > len(self.data):
                break

            train_data = self.data.iloc[
                train_start:train_end
            ].copy()

            test_data = self.data.iloc[
                test_start:test_end
            ].copy()

            yield {
                "train_data": train_data,
                "test_data": test_data,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end
            }

            start += self.step_size

    def create_sessions(
        self,
        window
    ):

        train_session = ResearchSession(
            data=window["train_data"]
        )

        test_session = ResearchSession(
            data=window["test_data"]
        )

        return train_session, test_session