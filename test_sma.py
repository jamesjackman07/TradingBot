from bot.research import ResearchSession
from bot.strategies.sma_cross import SMACrossoverStrategy

from analytics.report import Report
from analytics.plots import Plots


session = ResearchSession("SPY")

strategy = SMACrossoverStrategy()

equity, trades = session.run(strategy)


summary = Report.summary(
    equity,
    trades
)

Report.print(summary)

Plots.equity_curve(equity)