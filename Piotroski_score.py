from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PiotroskiFactors:
    year: int
    roa: float
    operating_cash_flow: float
    total_assets: float
    long_term_debt: float
    current_ratio: float
    ordinary_shares_number: float
    gross_margin: float
    assets_turnover: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator not in (0.0, -0.0) else default


class PiotroskiScore:
    # Source: https://www.anderson.ucla.edu/documents/areas/prg/asam/2019/F-Score.pdf

    def get_score(self, factors: list[PiotroskiFactors]) -> int:
        """
        Compute F-score using the 3 most recent yearly records.

        factors[0] -> year t (most recent)
        factors[1] -> year t-1
        factors[2] -> year t-2
        """
        if len(factors) < 3:
            raise ValueError("Need at least 3 yearly factor records.")

        return (
            self.roa_score(factors)
            + self.operating_cash_flow_score(factors)
            + self.roa_change_score(factors)
            + self.accruals_score(factors)
            + self.leverage_score(factors)
            + self.liquidity_score(factors)
            + self.share_issued_score(factors)
            + self.gross_margin_score(factors)
            + self.asset_turnover_score(factors)
        )

    def roa_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].roa > 0)

    def operating_cash_flow_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].operating_cash_flow > 0)

    def roa_change_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].roa > factors[1].roa)

    def accruals_score(self, factors: list[PiotroskiFactors]) -> int:
        operating_cashflow = factors[0].operating_cash_flow
        roa = factors[0].roa
        total_assets = factors[1].total_assets  # Beginning of year t.
        return int(_safe_div(operating_cashflow, total_assets) > roa)

    def leverage_score(self, factors: list[PiotroskiFactors]) -> int:
        lt_t = factors[0].long_term_debt
        lt_t1 = factors[1].long_term_debt
        a_t = factors[0].total_assets
        a_t1 = factors[1].total_assets
        a_t2 = factors[2].total_assets

        avg_assets_t = (a_t + a_t1) / 2.0
        avg_assets_t1 = (a_t1 + a_t2) / 2.0
        leverage_t = _safe_div(lt_t, avg_assets_t)
        leverage_t1 = _safe_div(lt_t1, avg_assets_t1)
        return int(leverage_t < leverage_t1)

    def liquidity_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].current_ratio > factors[1].current_ratio)

    def share_issued_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].ordinary_shares_number <= factors[1].ordinary_shares_number)

    def gross_margin_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].gross_margin > factors[1].gross_margin)

    def asset_turnover_score(self, factors: list[PiotroskiFactors]) -> int:
        return int(factors[0].assets_turnover > factors[1].assets_turnover)


def factors_from_dataframe(df: pd.DataFrame) -> list[PiotroskiFactors]:
    """
    Convert dataframe rows to PiotroskiFactors.

    Required columns:
    - year
    - roa
    - operating_cash_flow
    - total_assets
    - long_term_debt
    - current_ratio
    - ordinary_shares_number
    - gross_margin
    - assets_turnover
    """
    required_cols = {
        "year",
        "roa",
        "operating_cash_flow",
        "total_assets",
        "long_term_debt",
        "current_ratio",
        "ordinary_shares_number",
        "gross_margin",
        "assets_turnover",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = (
        df.copy()
        .assign(year=lambda x: pd.to_numeric(x["year"], errors="coerce"))
        .dropna(subset=["year"])
        .sort_values("year", ascending=False)
    )

    factors: list[PiotroskiFactors] = []
    for _, row in data.iterrows():
        factors.append(
            PiotroskiFactors(
                year=int(row["year"]),
                roa=_safe_float(row["roa"]),
                operating_cash_flow=_safe_float(row["operating_cash_flow"]),
                total_assets=_safe_float(row["total_assets"]),
                long_term_debt=_safe_float(row["long_term_debt"]),
                current_ratio=_safe_float(row["current_ratio"]),
                ordinary_shares_number=_safe_float(row["ordinary_shares_number"]),
                gross_margin=_safe_float(row["gross_margin"]),
                assets_turnover=_safe_float(row["assets_turnover"]),
            )
        )
    return factors


def factors_from_csv(csv_path: str | Path) -> list[PiotroskiFactors]:
    df = pd.read_csv(csv_path)
    return factors_from_dataframe(df)


def factors_from_online(symbol: str) -> list[PiotroskiFactors]:
    """
    Fetch annual financial statements from Yahoo Finance (via yfinance)
    and derive the factors required for Piotroski F-score.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to use online data: pip install yfinance") from exc

    ticker = yf.Ticker(symbol)
    fin = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow

    if fin is None or bs is None or cf is None or fin.empty or bs.empty or cf.empty:
        raise ValueError(f"No annual statement data found online for '{symbol}'.")

    years = sorted(
        set(fin.columns.tolist()) & set(bs.columns.tolist()) & set(cf.columns.tolist()),
        reverse=True,
    )
    if len(years) < 3:
        raise ValueError(f"Need at least 3 annual periods for '{symbol}', found {len(years)}.")

    factors: list[PiotroskiFactors] = []
    for period in years:
        yr = int(period.year)

        net_income = _safe_float(fin.at["Net Income", period] if "Net Income" in fin.index else 0.0)
        total_revenue = _safe_float(
            fin.at["Total Revenue", period] if "Total Revenue" in fin.index else 0.0
        )
        gross_profit = _safe_float(
            fin.at["Gross Profit", period] if "Gross Profit" in fin.index else 0.0
        )

        total_assets = _safe_float(
            bs.at["Total Assets", period] if "Total Assets" in bs.index else 0.0
        )
        current_assets = _safe_float(
            bs.at["Current Assets", period] if "Current Assets" in bs.index else 0.0
        )
        current_liabilities = _safe_float(
            bs.at["Current Liabilities", period] if "Current Liabilities" in bs.index else 0.0
        )
        long_term_debt = _safe_float(
            bs.at["Long Term Debt", period] if "Long Term Debt" in bs.index else 0.0
        )
        ordinary_shares = _safe_float(
            bs.at["Ordinary Shares Number", period]
            if "Ordinary Shares Number" in bs.index
            else 0.0
        )

        operating_cash_flow = _safe_float(
            cf.at["Operating Cash Flow", period] if "Operating Cash Flow" in cf.index else 0.0
        )

        roa = _safe_div(net_income, total_assets)
        current_ratio = _safe_div(current_assets, current_liabilities)
        gross_margin = _safe_div(gross_profit, total_revenue)
        assets_turnover = _safe_div(total_revenue, total_assets)

        factors.append(
            PiotroskiFactors(
                year=yr,
                roa=roa,
                operating_cash_flow=operating_cash_flow,
                total_assets=total_assets,
                long_term_debt=long_term_debt,
                current_ratio=current_ratio,
                ordinary_shares_number=ordinary_shares,
                gross_margin=gross_margin,
                assets_turnover=assets_turnover,
            )
        )

    return factors


def compute_piotroski_from_csv(csv_path: str | Path) -> int:
    scorer = PiotroskiScore()
    factors = factors_from_csv(csv_path)
    return scorer.get_score(factors[:3])


def compute_piotroski_from_online(symbol: str) -> int:
    scorer = PiotroskiScore()
    factors = factors_from_online(symbol)
    return scorer.get_score(factors[:3])


if __name__ == "__main__":
    # Example:
    # score = compute_piotroski_from_online("AAPL")
    # print("Piotroski F-score:", score)
    pass
