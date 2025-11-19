"""
Feature engineering pipeline for multi-asset portfolio RL.
Computed Technical Indicators
"""

import os
import json
import numpy as np
import pandas as pd
import ta
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
SCALER_DIR = os.path.join(DATA_DIR, "scalers")
os.makedirs(SCALER_DIR, exist_ok=True)

# -----------------------
# TECHNICAL INDICATORS
# -----------------------

def _ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["returns"] = df["close"].pct_change()
    # Causal log return: uses only past prices (2-step lag)
    df["log_return"] = np.log(df["close"].shift(1) / (df["close"].shift(2) + 1e-12))

    # trend (causal: use shift(1))
    df["ema_12"] = ta.trend.EMAIndicator(df["close"].shift(1), window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(df["close"].shift(1), window=26).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"].shift(1), window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"].shift(1), window=200).ema_indicator()

    # momentum
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"].shift(1), window=14).rsi()
    macd = ta.trend.MACD(df["close"].shift(1))
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # volatility & bands
    bb = ta.volatility.BollingerBands(df["close"].shift(1), window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    atr = ta.volatility.AverageTrueRange(
        df["high"].shift(1), df["low"].shift(1), df["close"].shift(1), window=14
    )
    df["atr_14"] = atr.average_true_range()

    df["vol_ema_20"] = ta.trend.EMAIndicator(df["volume"].shift(1), window=20).ema_indicator()

    return df


def _microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hl_range"] = (df["high"].shift(1) - df["low"].shift(1)) / (df["close"].shift(2) + 1e-12)
    df["oc_gap"] = (df["open"].shift(1) - df["close"].shift(2)) / (df["close"].shift(2) + 1e-12)
    df["realized_vol_10"] = df["log_return"].shift(1).rolling(10, min_periods=3).std()
    df["realized_vol_30"] = df["log_return"].shift(1).rolling(30, min_periods=5).std()
    df["vol_shock"] = (
        df["volume"].shift(1) / (df["volume"].shift(1).rolling(20, min_periods=5).mean() + 1e-12) - 1.0
    )

    return df


def _rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    shifted_close = df["close"].shift(1)
    # Allow some warmup with min_periods instead of dropping too many rows
    mean_20 = shifted_close.rolling(20, min_periods=3).mean()
    std_20 = shifted_close.rolling(20, min_periods=3).std()
    df["zscore_20"] = (shifted_close - mean_20) / (std_20 + 1e-12)

    df["return_ma_5"] = df["returns"].shift(1).rolling(5, min_periods=2).mean()
    df["return_ma_20"] = df["returns"].shift(1).rolling(20, min_periods=3).mean()
    df["return_std_20"] = df["returns"].shift(1).rolling(20, min_periods=3).std()

    return df


# -----------------------
# MAIN FEATURE PIPELINE
# -----------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical, microstructure, and rolling-statistics features.
    Requires columns: timestamp, open, high, low, close, volume.
    Returns a feature-enriched DataFrame, strictly causal & dropna'ed.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        raise ValueError("Missing timestamp column")

    df = _ta_indicators(df)
    df = _microstructure_features(df)
    df = _rolling_stats(df)

    df = df.dropna().reset_index(drop=True)
    return df


def _feature_columns(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]


def fit_scaler_on_train(train_df: pd.DataFrame) -> StandardScaler:
    cols = _feature_columns(train_df)
    scaler = StandardScaler()
    if len(cols) > 0 and len(train_df) > 0:
        scaler.fit(train_df[cols].values)
    return scaler


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    cols = _feature_columns(df)
    if len(cols) > 0:
        df.loc[:, cols] = scaler.transform(df[cols].values)
    return df


def save_scaler(scaler: StandardScaler, symbol: str) -> str:
    fname = os.path.join(SCALER_DIR, f"{symbol.replace('/', '_')}_scaler.pkl")
    joblib.dump(scaler, fname)
    return fname


# -----------------------
# FILE PROCESSING
# -----------------------

def process_symbol_file(path: str, save_unscaled: bool = True) -> Tuple[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path} missing required OHLCV columns")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # deduce symbol
    token = os.path.basename(path).split("_")[0]
    symbol = token.replace("_", "/")

    feat = add_features(raw)
    feat["symbol"] = symbol

    out_path = path.replace(".csv", "_features.csv")
    if save_unscaled:
        feat.to_csv(out_path, index=False)

        meta = {
            "symbol": symbol,
            "rows": len(feat),
            "start": feat["timestamp"].iloc[0].isoformat(),
            "end": feat["timestamp"].iloc[-1].isoformat(),
            "cols": list(feat.columns),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(out_path.replace(".csv", ".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[features] Processed {symbol}: {len(feat)} rows -> {out_path}")

    return out_path, len(feat)


def process_all(data_dir: str = DATA_DIR):
    files = [f for f in os.listdir(data_dir) if f.endswith("_1h.csv")]
    if not files:
        print("[features] No raw OHLCV files found.")
        return
    for f in files:
        process_symbol_file(os.path.join(data_dir, f))


# -----------------------
# DICT PROCESSING (for walk-forward / PPO)
# -----------------------

def prepare_feature_dict(raw_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    For a dict of raw *_features CSVs (already saved), reloads and cleans them.
    """
    final = {}
    for sym in raw_dict.keys():
        fname = os.path.join(DATA_DIR, f"{sym.replace('/', '_')}_1h_features.csv")
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"[features] Missing feature file for {sym}: {fname}\n"
                f"Run: python -m src.data.features"
            )
        df = pd.read_csv(fname)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        final[sym] = df

    print(f"[features] Loaded features for {len(final)} assets.")
    return final


def fit_scalers_on_train_dict(train_dict: Dict[str, pd.DataFrame], save: bool = False):
    scalers = {}
    for sym, df in train_dict.items():
        sc = fit_scaler_on_train(df)
        scalers[sym] = sc
        if save:
            save_scaler(sc, sym)
    return scalers


def transform_dict_with_scalers(df_dict: Dict[str, pd.DataFrame], scalers):
    out = {}
    for sym, df in df_dict.items():
        out[sym] = transform_with_scaler(df, scalers[sym])
    return out


def align_all_assets(data_dict: Dict[str, pd.DataFrame]):
    """
    Align assets to common timestamps.
    Ensures all assets share the same index and equal length.
    """
    starts = [df["timestamp"].min() for df in data_dict.values()]
    ends = [df["timestamp"].max() for df in data_dict.values()]

    g_start = max(starts)
    g_end = min(ends)

    aligned = {}
    for sym, df in data_dict.items():
        cut = df[(df["timestamp"] >= g_start) & (df["timestamp"] <= g_end)].copy()
        if cut.empty:
            raise RuntimeError(f"[features] {sym} empty after alignment.")
        aligned[sym] = cut.reset_index(drop=True)

    lengths = {sym: len(df) for sym, df in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"[features] Length mismatch after alignment: {lengths}")

    print(f"[features] Aligned {len(aligned)} assets to equal length {list(lengths.values())[0]}")
    return aligned


if __name__ == "__main__":
    print("[features] Running process_all()")
    process_all(DATA_DIR)
