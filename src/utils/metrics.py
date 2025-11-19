"""
Sharpe/Sortino/MaxDD/CVaR implementations
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence


# -------------------------------
# METRIC COMPUTATION
# -------------------------------

def sharpe_ratio(returns: Sequence[float], freq: float = 24 * 365) -> float:
    """
    Annualized Sharpe ratio: mean(ret)/std(ret) * sqrt(freq)
    freq = 24*365 for hourly returns in crypto.
    """
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(freq) * arr.mean() / (std + 1e-12))


def sortino_ratio(returns: Sequence[float], freq: float = 24 * 365) -> float:
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    downside = arr[arr < 0]
    if downside.size == 0:
        return float(np.sqrt(freq) * arr.mean())
    denom = downside.std()
    if denom < 1e-12:
        return 0.0
    return float(np.sqrt(freq) * arr.mean() / (denom + 1e-12))


def max_drawdown(pv: Sequence[float]) -> float:
    """
    Returns the max drawdown percentage (negative number).
    """
    pv = np.array(pv, dtype=float)
    if pv.size == 0:
        return 0.0
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / (peak + 1e-12)
    return float(dd.min())


def cvar(returns: Sequence[float], alpha: float = 0.95) -> float:
    """
    Conditional Value-at-Risk (95% default).
    """
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    losses = -arr
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    if tail.size == 0:
        return float(q)
    return float(tail.mean())


def annualized_return(pv: Sequence[float], freq: float = 24 * 365) -> float:
    """
    Annualized return from portfolio value series.
    """
    pv = np.array(pv, dtype=float)
    if pv.size < 2:
        return 0.0

    total_ret = pv[-1] / (pv[0] + 1e-12) - 1.0
    periods = len(pv)
    years = periods / freq
    if years <= 0:
        return 0.0

    return float((1.0 + total_ret) ** (1.0 / years) - 1.0)


# -------------------------------
# EVALUATION ENGINE
# -------------------------------

def _step_env_and_collect(model, env, max_steps=100000):
    """
    Runs model deterministically in evaluation mode.
    Collects NAV time series and returns.
    Compatible with Gymnasium + SB3 VecEnv.
    """

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out

    pv_list = []
    ret_list = []

    steps = 0
    done = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        step_out = env.step(action)
        # SB3 VecEnv style: (obs, rewards, dones, infos)
        if len(step_out) == 4:
            obs, rewards, dones, info = step_out
            # Vectorized env: dones is an array
            if isinstance(dones, (list, np.ndarray)):
                done = bool(np.any(dones))
            else:
                done = bool(dones)
            info0 = info[0] if isinstance(info, (list, tuple)) else info
        else:
            # Gymnasium single-env style (not typical with DummyVecEnv, but safe)
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            info0 = info[0] if isinstance(info, (list, tuple)) else info

        pv = float(info0.get("portfolio_value", np.nan))
        if np.isnan(pv):
            pv = 0.0

        pv_list.append(pv)

        # return series
        if len(pv_list) > 1:
            prev = pv_list[-2]
            if prev <= 0:
                prev = 1e-12
            ret = (pv_list[-1] - prev) / prev
            ret_list.append(float(ret))

        steps += 1

    return pv_list, ret_list


def evaluate_on_env(model, env, max_steps=100000) -> Dict[str, Any]:
    """
    Returns:
        {
            "pv": [...],
            "returns": [...],
            "cumulative_return": ...,
            "annualized_return": ...,
            "sharpe": ...,
            "sortino": ...,
            "max_drawdown": ...,
            "cvar95": ...
        }
    """
    pv, rets = _step_env_and_collect(model, env, max_steps=max_steps)

    if len(pv) == 0:
        return {
            "pv": [],
            "returns": [],
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "cvar95": 0.0,
        }

    cum = pv[-1] / (pv[0] + 1e-12) - 1.0
    ann = annualized_return(pv)
    sharpe = sharpe_ratio(rets)
    sortino = sortino_ratio(rets)
    mdd = max_drawdown(pv)
    c95 = cvar(rets, alpha=0.95)

    return {
        "pv": pv,
        "returns": rets,
        "cumulative_return": cum,
        "annualized_return": ann,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "cvar95": float(c95),
    }


# -------------------------------
# FILE UTILITIES
# -------------------------------

def save_metrics_json(metrics: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)


def metrics_to_csv(metrics_history: Sequence[Dict[str, Any]], path: str):
    """
    Strips PV and return series (arrays not suited for CSV).
    """
    rows = []
    for m in metrics_history:
        base = {k: v for k, v in m.items() if k not in ["pv", "returns"]}
        rows.append(base)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df
