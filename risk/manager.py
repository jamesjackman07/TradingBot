class RiskManager:

    def __init__(
        self,
        risk_percent=100,
        commission=0,
        slippage=0,
        stop_loss=None,
        take_profit=None
    ):
        self.risk_percent = risk_percent
        self.commission = commission
        self.slippage = slippage

        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def shares_to_buy(
        self,
        cash,
        price
    ):

        investment = cash * (
            self.risk_percent / 100
        )

        return investment / price

    def buy_price(
        self,
        price
    ):
        return price * (
            1 + self.slippage
        )

    def sell_price(
        self,
        price
    ):
        return price * (
            1 - self.slippage
        )

    def stop_price(
        self,
        entry_price
    ):

        if self.stop_loss is None:
            return None

        return entry_price * (
            1 - self.stop_loss
        )

    def target_price(
        self,
        entry_price
    ):

        if self.take_profit is None:
            return None

        return entry_price * (
            1 + self.take_profit
        )