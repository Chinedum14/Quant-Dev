from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CommodityTermStructureResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    roll_signal: pd.DataFrame
    long_symbols: dict[str, list[str]]
    short_symbols: dict[str, list[str]]


class CommodityTermStructureStrategy:
    """
    Commodity term-structure (roll yield) strategy:
    - Compute roll signal from near/far contracts
    - Monthly rebalance
    - Long top quintile (most backwardated)
    - Short bottom quintile (most contangoed)
    """

    def __init__(
        self,
        rebalance_frequency: str = "M",
        gross_leverage: float = 1.0,
        initial_cash: float = 100000.0,
    ) -> None:
        self.rebalance_frequency = str(rebalance_frequency)
        self.gross_leverage = float(gross_leverage)
        self.initial_cash = float(initial_cash)

    def run(self, csv_path: str | Path) -> CommodityTermStructureResult:
        near_prices, far_prices = self._load_near_far_prices(Path(csv_path))
        near_prices = near_prices.sort_index().ffill()
        far_prices = far_prices.sort_index().ffill()

        common = near_prices.columns.intersection(far_prices.columns)
        near_prices = near_prices[common].dropna(axis=1, how="all")
        far_prices = far_prices[common].dropna(axis=1, how="all")
        if near_prices.empty or far_prices.empty:
            raise ValueError("No usable overlapping symbols in near/far data.")

        # Positive signal => backwardation. Negative => contango.
        roll_signal = (near_prices / far_prices - 1.0).replace([np.inf, -np.inf], np.nan)

        daily_returns = near_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weights = pd.DataFrame(0.0, index=near_prices.index, columns=near_prices.columns)
        equity_curve = pd.Series(index=near_prices.index, dtype=float)
        equity_curve.iloc[0] = self.initial_cash

        long_picks: dict[str, list[str]] = {}
        short_picks: dict[str, list[str]] = {}
        current_weights = pd.Series(0.0, index=near_prices.columns)

        rebalance_dates = set(near_prices.resample(self.rebalance_frequency).last().index)
        for i in range(1, len(near_prices.index)):
            now = near_prices.index[i]
            day_ret = daily_returns.loc[now]
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1.0 + float((current_weights * day_ret).sum()))

            if now in rebalance_dates:
                signal_row = roll_signal.loc[now].dropna()
                if len(signal_row) >= 5:
                    q = max(1, len(signal_row) // 5)
                    ranked = signal_row.sort_values(ascending=False)
                    longs = ranked.head(q).index.tolist()
                    shorts = ranked.tail(q).index.tolist()

                    current_weights = pd.Series(0.0, index=near_prices.columns)
                    if longs:
                        current_weights.loc[longs] = (self.gross_leverage * 0.5) / len(longs)
                    if shorts:
                        current_weights.loc[shorts] = -(self.gross_leverage * 0.5) / len(shorts)

                    date_key = now.strftime("%Y-%m-%d")
                    long_picks[date_key] = longs
                    short_picks[date_key] = shorts
            weights.loc[now] = current_weights

        return CommodityTermStructureResult(
            equity_curve=equity_curve.dropna(),
            weights=weights,
            roll_signal=roll_signal,
            long_symbols=long_picks,
            short_symbols=short_picks,
        )

    def _load_near_far_prices(self, csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in frame.columns}

        # Supported format A (long):
        # Date, Symbol, Near, Far
        if {"date", "symbol", "near", "far"}.issubset(cols):
            d = cols["date"]
            s = cols["symbol"]
            n = cols["near"]
            f = cols["far"]
            frame[d] = pd.to_datetime(frame[d], utc=False)
            frame[n] = pd.to_numeric(frame[n], errors="coerce")
            frame[f] = pd.to_numeric(frame[f], errors="coerce")

            near_prices = (
                frame[[d, s, n]]
                .pivot_table(index=d, columns=s, values=n, aggfunc="last")
                .sort_index()
            )
            far_prices = (
                frame[[d, s, f]]
                .pivot_table(index=d, columns=s, values=f, aggfunc="last")
                .sort_index()
            )
            return near_prices, far_prices

        # Supported format B (wide):
        # Date, CL_NEAR, CL_FAR, GC_NEAR, GC_FAR, ...
        if "date" not in cols:
            raise ValueError("CSV must include a Date column.")

        date_col = cols["date"]
        frame[date_col] = pd.to_datetime(frame[date_col], utc=False)
        frame = frame.sort_values(date_col).set_index(date_col)

        near_map: dict[str, pd.Series] = {}
        far_map: dict[str, pd.Series] = {}
        for c in frame.columns:
            c_up = str(c).upper()
            if c_up.endswith("_NEAR"):
                symbol = c_up[: -len("_NEAR")]
                near_map[symbol] = pd.to_numeric(frame[c], errors="coerce")
            elif c_up.endswith("_FAR"):
                symbol = c_up[: -len("_FAR")]
                far_map[symbol] = pd.to_numeric(frame[c], errors="coerce")

        if not near_map or not far_map:
            raise ValueError(
                "Wide format requires near/far columns like CL_NEAR and CL_FAR "
                "or use long format: Date, Symbol, Near, Far."
            )

        near_prices = pd.concat(near_map, axis=1).sort_index()
        far_prices = pd.concat(far_map, axis=1).sort_index()
        return near_prices, far_prices


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
    parser = argparse.ArgumentParser(description="Commodity Term Structure / Roll Yield Strategy")
    parser.add_argument("--csv", required=True, help="Path to CSV input")
    parser.add_argument("--rebalance", default="M", help="Rebalance frequency (default: M)")
    parser.add_argument("--gross-leverage", type=float, default=1.0, help="Gross exposure (long+short)")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Starting portfolio value")
    args = parser.parse_args()

    strategy = CommodityTermStructureStrategy(
        rebalance_frequency=args.rebalance,
        gross_leverage=args.gross_leverage,
        initial_cash=args.initial_cash,
    )
    result = strategy.run(csv_path=args.csv)
    summary = performance_summary(result.equity_curve)

    print("Rebalance snapshots:", len(result.long_symbols))
    print("Latest long basket:", result.long_symbols[list(result.long_symbols.keys())[-1]] if result.long_symbols else [])
    print("Latest short basket:", result.short_symbols[list(result.short_symbols.keys())[-1]] if result.short_symbols else [])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
