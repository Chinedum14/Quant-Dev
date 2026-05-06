from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_report(
    series: pd.Series,
    alpha: float = 0.05,
    min_samples: int = 40,
) -> dict[str, Any]:
    """
    Run ADF + KPSS tests and return a stationarity report for a Series.

    A series is considered stationary when:
    - ADF p-value < alpha
    - KPSS p-value > alpha
    """
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_samples:
        return {
            "overall_stationary": False,
            "adf_p_value": np.nan,
            "kpss_p_value": np.nan,
            "reason": f"insufficient_samples_{len(clean)}",
        }

    adf_result = adfuller(clean, autolag="AIC")
    kpss_result = kpss(clean, regression="c", nlags="auto")

    adf_p = float(adf_result[1])
    kpss_p = float(kpss_result[1])
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha

    return {
        "overall_stationary": adf_stationary and kpss_stationary,
        "adf_statistic": float(adf_result[0]),
        "adf_p_value": adf_p,
        "adf_stationary": adf_stationary,
        "kpss_statistic": float(kpss_result[0]),
        "kpss_p_value": kpss_p,
        "kpss_stationary": kpss_stationary,
        "samples": int(len(clean)),
    }


def is_stationary_series(
    series: pd.Series,
    alpha: float = 0.05,
    min_samples: int = 40,
) -> bool:
    """Return True when ADF and KPSS jointly classify the series as stationary."""
    report = stationarity_report(series=series, alpha=alpha, min_samples=min_samples)
    return bool(report["overall_stationary"])
