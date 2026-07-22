from bot.research import ResearchSession
from bot.strategies.mean_reversion import MeanReversionStrategy

from analytics.report import Report
from analytics.plots import Plots


session = ResearchSession("SPY")

strategy = MeanReversionStrategy()

equity, trades = session.run(strategy)


summary = Report.summary(
    equity,
    trades
)

Report.print(summary)

Plots.equity_curve(equity)