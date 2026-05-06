import numpy as np
import pandas as pd


def add_zscore_feature(
    df: pd.DataFrame,
    column: str,
    new_column: str | None = None,
    window: int | None = None,
    groupby: str | list[str] | None = None,
    ddof: int = 0,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add a z-score feature column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numeric column to standardize.
    new_column : str | None, default None
        Name of z-score column. Defaults to f"{column}_zscore".
    window : int | None, default None
        If provided, compute rolling z-score over this window.
        If None, compute z-score over the entire series (or per group).
    groupby : str | list[str] | None, default None
        Optional grouping key(s) for group-wise z-score.
    ddof : int, default 0
        Delta degrees of freedom used in std calculation.
    inplace : bool, default False
        If True, mutate and return original dataframe.
        If False, return a copy.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    if window is not None and window <= 1:
        raise ValueError("window must be greater than 1 when provided.")

    if new_column is None:
        new_column = f"{column}_zscore"

    result = df if inplace else df.copy()

    def _zscore_series(s: pd.Series) -> pd.Series:
        if window is None:
            mean = s.mean()
            std = s.std(ddof=ddof)
            if std == 0 or np.isnan(std):
                return pd.Series(np.nan, index=s.index)
            return (s - mean) / std

        rolling_mean = s.rolling(window=window, min_periods=window).mean()
        rolling_std = s.rolling(window=window, min_periods=window).std(ddof=ddof)
        z = (s - rolling_mean) / rolling_std
        z = z.mask(rolling_std == 0)
        return z

    if groupby is None:
        result[new_column] = _zscore_series(result[column])
    else:
        result[new_column] = (
            result.groupby(groupby, group_keys=False)[column]
            .apply(_zscore_series)
            .reindex(result.index)
        )

    return result


def latest_zscore(
    series: pd.Series,
    window: int = 100,
    ddof: int = 0,
) -> float:
    """Return the z-score of the latest value in a rolling window."""
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    if len(clean) < window:
        raise ValueError(f"Need at least {window} samples; got {len(clean)}.")

    sample = clean.tail(window)
    std = sample.std(ddof=ddof)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((sample.iloc[-1] - sample.mean()) / std)
