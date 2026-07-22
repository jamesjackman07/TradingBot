class Portfolio:

    def __init__(self, initial_cash=10000):

        self.cash = initial_cash
        self.shares = 0

    def buy(self, price):

        if self.shares == 0:

            self.shares = self.cash / price
            self.cash = 0

    def sell(self, price):

        if self.shares > 0:

            self.cash = self.shares * price
            self.shares = 0

    def equity(self, price):

        return self.cash + self.shares * price

    def has_position(self):

        return self.shares > 0