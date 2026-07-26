class RiskManager:

    def __init__(
        self,
        risk_percent=100,
        commission=0,
        slippage=0,
        stop_loss=None,
        take_profit=None,
        risk_per_trade=None
    ):

        # --------------------------------
        # Allocation validation
        # --------------------------------

        if not 0 <= risk_percent <= 100:
            raise ValueError(
                "risk_percent must be between 0 and 100"
            )

        # --------------------------------
        # Cost validation
        # --------------------------------

        if commission < 0:
            raise ValueError(
                "commission cannot be negative"
            )

        if slippage < 0:
            raise ValueError(
                "slippage cannot be negative"
            )

        # --------------------------------
        # Protective exit validation
        # --------------------------------

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

        # --------------------------------
        # Risk-per-trade validation
        # --------------------------------

        if risk_per_trade is not None:

            if not 0 < risk_per_trade <= 1:
                raise ValueError(
                    "risk_per_trade must be "
                    "greater than 0 and no more than 1"
                )

        self.risk_percent = risk_percent

        self.commission = commission
        self.slippage = slippage

        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.risk_per_trade = risk_per_trade

    # ====================================
    # Existing allocation-based sizing
    # ====================================

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

        available_for_shares = max(
            0,
            allocation - self.commission
        )

        return (
            available_for_shares
            / price
        )

    # ====================================
    # Risk-based position sizing
    # ====================================

    def risk_based_shares(
        self,
        cash,
        entry_price,
        stop_price
    ):

        if self.risk_per_trade is None:
            raise ValueError(
                "risk_per_trade must be configured "
                "for risk-based sizing"
            )

        if cash < 0:
            raise ValueError(
                "cash cannot be negative"
            )

        if entry_price <= 0:
            raise ValueError(
                "entry_price must be greater than 0"
            )

        if stop_price <= 0:
            raise ValueError(
                "stop_price must be greater than 0"
            )

        if stop_price >= entry_price:
            raise ValueError(
                "stop_price must be below entry_price "
                "for a long position"
            )

        # --------------------------------
        # Maximum intended monetary loss
        # --------------------------------

        risk_budget = (
            cash * self.risk_per_trade
        )

        # --------------------------------
        # Monetary risk per share
        # --------------------------------

        risk_per_share = (
            entry_price - stop_price
        )

        risk_based_shares = (
            risk_budget / risk_per_share
        )

        # --------------------------------
        # Cash affordability
        # --------------------------------

        available_for_shares = max(
            0,
            cash - self.commission
        )

        affordable_shares = (
            available_for_shares
            / entry_price
        )

        # Never allow risk sizing to
        # create leverage in a cash account.
        return min(
            risk_based_shares,
            affordable_shares
        )

    # ====================================
    # Execution prices
    # ====================================

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

    # ====================================
    # Protective prices
    # ====================================

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