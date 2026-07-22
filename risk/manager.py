class RiskManager:

    def __init__(
        self,
        risk_percent=100,
        commission=1,
        slippage=0.002
    ):
        self.risk_percent = risk_percent
        self.commission = commission
        self.slippage = slippage

    def shares_to_buy(self, cash, price):

        investment = cash * (self.risk_percent / 100)

        shares = investment / price

        return shares

    def buy_price(self, price):

        return price * (1 + self.slippage)

    def sell_price(self, price):

        return price * (1 - self.slippage)