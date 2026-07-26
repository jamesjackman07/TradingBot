class Portfolio:

    def __init__(
        self,
        initial_cash
    ):

        if initial_cash < 0:
            raise ValueError(
                "initial_cash cannot be negative"
            )

        self.cash = initial_cash
        self.shares = 0

    def buy(
        self,
        price,
        shares,
        commission=0
    ):

        if price <= 0:
            raise ValueError(
                "price must be greater than 0"
            )

        if shares <= 0:
            raise ValueError(
                "shares must be greater than 0"
            )

        if commission < 0:
            raise ValueError(
                "commission cannot be negative"
            )

        if self.shares != 0:
            return

        total_cost = (
            shares * price
            + commission
        )

        if total_cost > self.cash + 1e-9:
            raise ValueError(
                "Insufficient cash for purchase"
            )

        self.cash -= total_cost
        self.shares = shares

    def sell(
        self,
        price,
        commission=0
    ):

        if price <= 0:
            raise ValueError(
                "price must be greater than 0"
            )

        if commission < 0:
            raise ValueError(
                "commission cannot be negative"
            )

        if self.shares > 0:

            self.cash += (
                self.shares * price
            ) - commission

            self.shares = 0

    def equity(
        self,
        current_price
    ):

        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than 0"
            )

        return (
            self.cash
            + self.shares * current_price
        )

    def has_position(self):

        return self.shares > 0