class RiskManager:

    def __init__(
        self,
        risk_percent=100,
        commission=0,
        slippage=0,
        stop_loss=None,
        take_profit=None
    ):

        if not 0 <= risk_percent <= 100:
            raise ValueError(
                "risk_percent must be between 0 and 100"
            )

        if commission < 0:
            raise ValueError(
                "commission cannot be negative"
            )

        if slippage < 0:
            raise ValueError(
                "slippage cannot be negative"
            )

        if (
            stop_loss is not None
            and stop_loss <= 0
        ):
            raise ValueError(
                "stop_loss must be greater than 0"
            )

        if (
            take_profit is not None
            and take_profit <= 0
        ):
            raise ValueError(
                "take_profit must be greater than 0"
            )

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

        if cash < 0:
            raise ValueError(
                "cash cannot be negative"
            )

        if price <= 0:
            raise ValueError(
                "price must be greater than 0"
            )

        allocation = cash * (
            self.risk_percent / 100
        )

        # Commission must fit inside
        # the allocated capital.
        available_for_shares = max(
            0,
            allocation - self.commission
        )

        return (
            available_for_shares
            / price
        )

    def buy_price(
        self,
        price
    ):

        if price <= 0:
            raise ValueError(
                "price must be greater than 0"
            )

        return price * (
            1 + self.slippage
        )

    def sell_price(
        self,
        price
    ):

        if price <= 0:
            raise ValueError(
                "price must be greater than 0"
            )

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