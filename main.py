from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from HMM import fit_hmm_model_from_returns, predict_hmm_state
from Stationarity import is_stationary_series


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    selected_universe: list[str]


class HMMPairsLikeEtfStrategy:
    def __init__(
        self,
        initial_cash: float = 100000.0,
        max_universe_size: int = 8,
        zscore_window: int = 100,
        min_signal_points: int = 150,
        hmm_lookback_days: int = 900,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.max_universe_size = int(max_universe_size)
        self.zscore_window = int(zscore_window)
        self.min_signal_points = int(min_signal_points)
        self.hmm_lookback_days = int(hmm_lookback_days)
        self.etf_candidates = [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "TLT",
            "IEF",
            "GLD",
            "SLV",
            "USO",
            "XLF",
            "XLE",
            "XLK",
            "XLV",
            "XLI",
            "XLP",
            "XLY",
            "VNQ",
            "EEM",
            "FXI",
            "ARKK",
        ]

    def run(
        self,
        start: str = "2018-01-01",
        end: str | None = None,
        data_dir: str | Path | None = None,
    ) -> BacktestResult:
        prices, volumes = self._load_market_data(start=start, end=end, data_dir=data_dir)
        if prices.empty:
            raise ValueError("No price data loaded.")

        selected = self._select_universe(prices, volumes)
        prices = prices[selected].dropna(how="all")
        if prices.empty:
            raise ValueError("No usable price rows after universe selection.")
        prices = prices.ffill().dropna(how="any")

        models: dict[str, object | None] = {s: None for s in selected}
        weights = pd.DataFrame(0.0, index=prices.index, columns=selected)
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        current_weights = pd.Series(0.0, index=selected)
        portfolio_value.iloc[0] = self.initial_cash

        for i in range(1, len(prices.index)):
            now = prices.index[i]
            prev = prices.index[i - 1]
            day_ret = (prices.loc[now] / prices.loc[prev] - 1.0).replace([np.inf, -np.inf], np.nan)
            day_ret = day_ret.fillna(0.0)
            portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (
                1.0 + float((current_weights * day_ret).sum())
            )

            if i < self.hmm_lookback_days + self.min_signal_points:
                weights.loc[now] = current_weights
                continue

            if now.month != prev.month:
                self._refit_models(models=models, prices=prices, at_index=i)

            current_weights = self._rebalance(
                models=models,
                prices=prices,
                at_index=i,
                current_weights=current_weights,
            )
            weights.loc[now] = current_weights

        return BacktestResult(
            equity_curve=portfolio_value.dropna(),
            weights=weights,
            selected_universe=selected,
        )

    def _select_universe(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> list[str]:
        common_cols = [c for c in self.etf_candidates if c in prices.columns and c in volumes.columns]
        if not common_cols:
            raise ValueError("None of the ETF candidates were found in loaded data.")

        latest_price = prices[common_cols].ffill().iloc[-1]
        dv = (prices[common_cols] * volumes[common_cols]).tail(20).mean()
        eligible = dv[(latest_price > 5) & (dv > 0)].sort_values(ascending=False)
        selected = eligible.head(self.max_universe_size).index.tolist()
        if not selected:
            raise ValueError("Universe selection returned no symbols.")
        return selected

    def _refit_models(self, models: dict[str, object | None], prices: pd.DataFrame, at_index: int) -> None:
        for symbol in models:
            close = prices[symbol].iloc[: at_index + 1].dropna()
            if len(close) < self.hmm_lookback_days + 1:
                continue
            returns = close.pct_change().dropna()
            if len(returns) < self.hmm_lookback_days:
                continue
            try:
                models[symbol] = fit_hmm_model_from_returns(
                    returns=returns,
                    n_states=2,
                    lookback_days=self.hmm_lookback_days,
                    random_state=42,
                )
            except Exception:
                continue

    def _rebalance(
        self,
        models: dict[str, object | None],
        prices: pd.DataFrame,
        at_index: int,
        current_weights: pd.Series,
    ) -> pd.Series:
        buy_symbols: list[str] = []
        sell_symbols: list[str] = []

        for symbol, model in models.items():
            if model is None:
                continue

            close = prices[symbol].iloc[max(0, at_index - 500) : at_index + 1].dropna()
            if len(close) < self.min_signal_points:
                continue

            returns = close.pct_change().dropna()
            if len(returns) < self.min_signal_points - 1:
                continue

            if not is_stationary_series(returns):
                continue

            zscore = self._latest_zscore(returns)
            if zscore is None or np.isnan(zscore):
                continue

            try:
                current_state = predict_hmm_state(model, returns.tail(20))
            except Exception:
                continue

            invested = current_weights.get(symbol, 0.0) > 0.0
            if zscore < -1.0 and current_state == 0:
                buy_symbols.append(symbol)
            elif zscore > 1.0 or (invested and current_state == 1):
                sell_symbols.append(symbol)

        invested_symbols = current_weights[current_weights > 0].index.tolist()
        target_symbols = list(set(invested_symbols + buy_symbols) - set(sell_symbols))
        new_weights = pd.Series(0.0, index=current_weights.index)

        if target_symbols:
            weight = 1.0 / len(target_symbols)
            for symbol in target_symbols:
                new_weights.loc[symbol] = weight
        return new_weights

    def _latest_zscore(self, returns: pd.Series) -> float | None:
        zscore_path = Path(__file__).with_name("Z-score.py")
        if not zscore_path.exists():
            return self._fallback_latest_zscore(returns)

        try:
            spec = importlib.util.spec_from_file_location("zscore_module", str(zscore_path))
            if spec is None or spec.loader is None:
                return self._fallback_latest_zscore(returns)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "latest_zscore"):
                return float(module.latest_zscore(returns, window=self.zscore_window))
        except Exception:
            pass

        return self._fallback_latest_zscore(returns)

    def _fallback_latest_zscore(self, returns: pd.Series) -> float | None:
        if len(returns) < self.zscore_window:
            return None
        sample = returns.tail(self.zscore_window)
        std = sample.std(ddof=0)
        if std == 0 or np.isnan(std):
            return None
        return float((sample.iloc[-1] - sample.mean()) / std)

    def _load_market_data(
        self,
        start: str,
        end: str | None,
        data_dir: str | Path | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data_dir is not None:
            return self._load_from_csv_directory(Path(data_dir))
        return self._load_from_yfinance(start=start, end=end)

    def _load_from_csv_directory(self, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        close_map: dict[str, pd.Series] = {}
        volume_map: dict[str, pd.Series] = {}
        for symbol in self.etf_candidates:
            path = data_dir / f"{symbol}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            cols = {c.lower(): c for c in frame.columns}
            if "date" not in cols or "close" not in cols:
                continue
            frame[cols["date"]] = pd.to_datetime(frame[cols["date"]], utc=False)
            frame = frame.sort_values(cols["date"]).set_index(cols["date"])
            close_map[symbol] = pd.to_numeric(frame[cols["close"]], errors="coerce")
            if "volume" in cols:
                volume_map[symbol] = pd.to_numeric(frame[cols["volume"]], errors="coerce")
            else:
                volume_map[symbol] = pd.Series(0.0, index=frame.index)

        if not close_map:
            raise ValueError(f"No usable CSV files found in: {data_dir}")

        prices = pd.concat(close_map, axis=1).sort_index()
        volumes = pd.concat(volume_map, axis=1).sort_index().fillna(0.0)
        return prices, volumes

    def _load_from_yfinance(self, start: str, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for standalone market data download. "
                "Install with: pip install yfinance"
            ) from exc

        data = yf.download(
            tickers=self.etf_candidates,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data.empty:
            raise ValueError("yfinance returned no data.")

        close_map: dict[str, pd.Series] = {}
        volume_map: dict[str, pd.Series] = {}
        for symbol in self.etf_candidates:
            if symbol not in data.columns.get_level_values(0):
                continue
            symbol_data = data[symbol]
            if "Close" not in symbol_data.columns:
                continue
            close_map[symbol] = pd.to_numeric(symbol_data["Close"], errors="coerce")
            if "Volume" in symbol_data.columns:
                volume_map[symbol] = pd.to_numeric(symbol_data["Volume"], errors="coerce")
            else:
                volume_map[symbol] = pd.Series(0.0, index=symbol_data.index)

        prices = pd.concat(close_map, axis=1).sort_index()
        volumes = pd.concat(volume_map, axis=1).sort_index().fillna(0.0)
        return prices, volumes


def _performance_summary(equity_curve: pd.Series) -> dict[str, float]:
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
    drawdowns = equity_curve / rolling_max - 1.0
    max_drawdown = float(drawdowns.min())

    return {
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


if __name__ == "__main__":
    strategy = HMMPairsLikeEtfStrategy()
    result = strategy.run(start="2018-01-01", end=None, data_dir=None)
    summary = _performance_summary(result.equity_curve)

    print("Selected universe:", ", ".join(result.selected_universe))
    print(json.dumps(summary, indent=2))
