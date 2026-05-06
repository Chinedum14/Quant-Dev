from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class IVFAResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    selected_symbols: list[str]


class InverseVolatilityFuturesStrategy:
    def __init__(
        self,
        top_n: int = 5,
        vol_window: int = 30,
        rebalance_frequency: str = "W-MON",
        leverage: float = 0.5,
        initial_cash: float = 100000.0,
    ) -> None:
        self.top_n = int(top_n)
        self.vol_window = int(vol_window)
        self.rebalance_frequency = str(rebalance_frequency)
        self.leverage = float(leverage)
        self.initial_cash = float(initial_cash)

    def run(self, csv_path: str | Path) -> IVFAResult:
        prices = self._load_prices(csv_path=Path(csv_path))
        returns = prices.pct_change()
        volatility = returns.rolling(self.vol_window).std(ddof=0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        current_weights = pd.Series(0.0, index=prices.columns)

        portfolio_value.iloc[0] = self.initial_cash
        rebalance_dates = set(prices.resample(self.rebalance_frequency).last().index)

        for i in range(1, len(prices.index)):
            now = prices.index[i]
            prev = prices.index[i - 1]

            day_ret = (prices.loc[now] / prices.loc[prev] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (
                1.0 + float((current_weights * day_ret).sum())
            )

            if now in rebalance_dates:
                vol_row = volatility.loc[now].replace([np.inf, -np.inf], np.nan).dropna()
                vol_row = vol_row[vol_row > 0.0]
                selected = vol_row.nsmallest(self.top_n)

                if not selected.empty:
                    inverse = 1.0 / selected
                    alloc = inverse / inverse.sum()
                    current_weights = pd.Series(0.0, index=prices.columns)
                    current_weights.loc[alloc.index] = alloc.values * self.leverage
            weights.loc[now] = current_weights

        selected_symbols = (
            weights.replace(0.0, np.nan).dropna(axis=1, how="all").columns.tolist()
        )
        return IVFAResult(
            equity_curve=portfolio_value.dropna(),
            weights=weights,
            selected_symbols=selected_symbols,
        )

    def _load_prices(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in frame.columns}

        # Supported formats:
        # 1) Wide: Date, COL1, COL2, ...
        # 2) Long: Date, Symbol, Close
        if {"date", "symbol", "close"}.issubset(cols):
            frame[cols["date"]] = pd.to_datetime(frame[cols["date"]], utc=False)
            frame[cols["close"]] = pd.to_numeric(frame[cols["close"]], errors="coerce")
            wide = (
                frame[[cols["date"], cols["symbol"], cols["close"]]]
                .pivot_table(index=cols["date"], columns=cols["symbol"], values=cols["close"], aggfunc="last")
                .sort_index()
            )
            prices = wide
        elif "date" in cols:
            date_col = cols["date"]
            frame[date_col] = pd.to_datetime(frame[date_col], utc=False)
            frame = frame.sort_values(date_col).set_index(date_col)
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric price columns found in wide CSV format.")
            prices = frame[numeric_cols]
        else:
            raise ValueError("CSV must include either [Date, Symbol, Close] or [Date, wide price columns].")

        prices = prices.sort_index().ffill().dropna(how="all")
        prices = prices.dropna(axis=1, how="all")
        if prices.empty:
            raise ValueError("No usable price data found in CSV.")
        return prices


def performance_summary(equity_curve: pd.Series) -> dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    periods_per_year = 252.0
    years = max(len(returns) / periods_per_year, 1.0 / periods_per_year)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float((returns.mean() * periods_per_year) / annualized_vol) if annualized_vol > 0 else 0.0
    rolling_max = equity_curve.cummax()
    max_drawdown = float((equity_curve / rolling_max - 1.0).min())

    return {
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse Volatility Futures Allocation (CSV input)")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--top-n", type=int, default=5, help="Number of lowest-vol symbols to hold")
    parser.add_argument("--vol-window", type=int, default=30, help="Rolling volatility lookback")
    parser.add_argument("--rebalance", default="W-MON", help="Pandas resample frequency, e.g., W-MON, M")
    parser.add_argument("--leverage", type=float, default=0.5, help="Total gross allocation")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Starting portfolio value")
    args = parser.parse_args()

    strategy = InverseVolatilityFuturesStrategy(
        top_n=args.top_n,
        vol_window=args.vol_window,
        rebalance_frequency=args.rebalance,
        leverage=args.leverage,
        initial_cash=args.initial_cash,
    )
    result = strategy.run(csv_path=args.csv)
    summary = performance_summary(result.equity_curve)

    print("Symbols traded:", ", ".join(result.selected_symbols))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
