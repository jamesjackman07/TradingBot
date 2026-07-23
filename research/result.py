class ResearchResult:

    def __init__(self, row):
        self.row = row

    @property
    def parameters(self):
        return {
            "fast": int(self.row["fast"]),
            "slow": int(self.row["slow"])
        }

    @property
    def metrics(self):
        return {
            "Return": self.row["Return"],
            "Sharpe": self.row["Sharpe"],
            "Drawdown": self.row["Drawdown"],
            "Profit Factor": self.row["Profit Factor"],
            "Win Rate": self.row["Win Rate"],
            "Trades": self.row["Trades"]
        }

    def __getitem__(self, key):
        return self.row[key]

    def __repr__(self):

        output = []

        output.append("=" * 30)
        output.append("BEST STRATEGY")
        output.append("=" * 30)

        output.append("")
        output.append("Parameters")

        for key, value in self.parameters.items():
            output.append(f"  {key}: {value}")

        output.append("")
        output.append("Metrics")

        for key, value in self.metrics.items():

            if key == "Trades":
                output.append(f"  {key}: {int(value)}")
            elif isinstance(value, float):
                output.append(f"  {key}: {value:.2f}")
            else:
                output.append(f"  {key}: {value}")

        return "\n".join(output)