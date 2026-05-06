from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "hmmlearn is required for HMM.py. Install with: pip install hmmlearn"
    ) from exc


def train_hmm_on_900_days(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_states: int = 3,
    lookback_days: int = 900,
    date_column: str | None = None,
    covariance_type: str = "diag",
    n_iter: int = 250,
    random_state: int = 42,
) -> tuple[GaussianHMM, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if n_states < 2:
        raise ValueError("n_states must be at least 2.")

    working = df.copy()

    if date_column is not None:
        if date_column not in working.columns:
            raise KeyError(f"date_column '{date_column}' not found in DataFrame.")
        working = working.sort_values(by=date_column)

    if feature_columns is None:
        feature_columns = working.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [c for c in feature_columns if c not in working.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")

    if not feature_columns:
        raise ValueError("No numeric features available for HMM training.")

    model_data = working[feature_columns].dropna()
    if len(model_data) < lookback_days:
        raise ValueError(
            f"Need at least {lookback_days} clean rows for training; got {len(model_data)}."
        )

    model_data = model_data.tail(lookback_days)
    X = model_data.to_numpy()

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)

    result = working.loc[model_data.index].copy()
    result["hidden_state"] = hidden_states
    for k in range(n_states):
        result[f"state_prob_{k}"] = state_probs[:, k]

    return model, result


def fit_hmm_model_from_returns(
    returns: pd.Series,
    n_states: int = 2,
    lookback_days: int = 900,
    covariance_type: str = "diag",
    n_iter: int = 250,
    random_state: int = 42,
) -> GaussianHMM:
    """Fit an HMM using the most recent `lookback_days` return observations."""
    clean = pd.Series(returns).dropna()
    if len(clean) < lookback_days:
        raise ValueError(
            f"Need at least {lookback_days} return observations; got {len(clean)}."
        )

    X = clean.tail(lookback_days).to_numpy().reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_hmm_state(model: GaussianHMM, recent_returns: pd.Series) -> int:
    """Predict the latest hidden state from a sequence of recent returns."""
    clean = pd.Series(recent_returns).dropna()
    if clean.empty:
        raise ValueError("recent_returns is empty after dropping NaN values.")

    X = clean.to_numpy().reshape(-1, 1)
    states = model.predict(X)
    return int(states[-1])
