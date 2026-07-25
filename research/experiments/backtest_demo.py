from bot.research import ResearchSession
from bot.strategies.mean_reversion import MeanReversionStrategy

from analytics.report import Report
from analytics.plots import Plots
from risk.manager import RiskManager

risk = RiskManager(
    stop_loss=0.03,
    take_profit=0.06
)

session = ResearchSession(
    "SPY",
    risk_manager=risk
)

strategy = MeanReversionStrategy()

equity, trades = session.run(strategy)


summary = Report.summary(
    equity,
    trades
)

Report.print(summary)

Plots.equity_curve(equity)