from dataclasses import dataclass


@dataclass
class CompletedTrade:
    """
    Represents a completed trade from entry to exit.
    """

    entry_price: float
    exit_price: float

    shares: float

    entry_index: int
    exit_index: int

    @property
    def profit(self) -> float:
        """
        Profit in currency.
        """
        return (
            self.exit_price
            - self.entry_price
        ) * self.shares

    @property
    def return_percent(self) -> float:
        """
        Percentage return.
        """
        return (
            (
                self.exit_price
                - self.entry_price
            )
            / self.entry_price
        ) * 100

    @property
    def duration(self) -> int:
        """
        Number of bars held.
        """
        return (
            self.exit_index
            - self.entry_index
        )