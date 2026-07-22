class Portfolio:

    def __init__(self, initial_cash):

        self.cash = initial_cash
        self.shares = 0

    def buy(self, price, shares, commission=0):

        if self.shares == 0:

            self.cash -= (shares * price) + commission
            self.shares = shares

    def sell(self, price, commission=0):

        if self.shares > 0:

            self.cash += (self.shares * price) - commission
            self.shares = 0

    def equity(self, current_price):

        return self.cash + (self.shares * current_price)

    def has_position(self):

        return self.shares > 0