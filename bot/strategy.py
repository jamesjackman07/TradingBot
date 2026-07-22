from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def generate_signals(self, data):
        """
        Returns BUY, SELL or HOLD signals.
        """
        pass