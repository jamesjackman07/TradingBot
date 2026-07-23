from dataclasses import dataclass


@dataclass
class Trade:
    trade_type: str
    price: float
    index: int
    shares: float
    reason: str = "SIGNAL"

    @property
    def value(self):
        return self.price * self.shares