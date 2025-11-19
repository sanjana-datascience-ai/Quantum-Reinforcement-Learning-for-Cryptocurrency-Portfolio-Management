"""
OHLCV downloader for Binance (ccxt).
"""

import os
import time
import json
from typing import List
from datetime import datetime

import ccxt
import pandas as pd

# -----------------------
# DIRECTORY SETUP
# -----------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# EXCHANGE SETUP
# -----------------------

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

exchange_kwargs = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
if API_KEY and API_SECRET:
    exchange_kwargs.update({"apiKey": API_KEY, "secret": API_SECRET})

exchange = ccxt.binance(exchange_kwargs)

# Cache markets once
try:
    MARKETS = exchange.load_markets()
except Exception as e:
    raise RuntimeError(f"[loader] Failed to load Binance markets: {e}")


# -----------------------
# HELPERS
# -----------------------

def _ensure_valid_symbol(symbol: str) -> str:
    """
    Normalize symbol to the exchange's canonical format.

    Handles:
      - Different casing (btc/usdt -> BTC/USDT)
      - Canonical variations (BTC/USDT:USDT)
    """
    sym = symbol.upper()

    # Direct match
    if sym in MARKETS:
        return MARKETS[sym].get("symbol", sym)

    # Try to find canonical symbol starting with requested base/quote
    for m, meta in MARKETS.items():
        if m.upper().startswith(sym):
            return meta.get("symbol", m)

    raise ValueError(f"[loader] Symbol '{symbol}' not available on Binance.")


def _to_ms_timestamp(dt_str: str) -> int:
    dt = datetime.fromisoformat(dt_str)
    return int(dt.timestamp() * 1000)


# -----------------------
# OHLCV FETCHING
# -----------------------

def fetch_ohlcv_full(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2017-01-01",
    end_date: str = "2025-01-01",
    limit: int = 1000,
    retry_sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Fetches full OHLCV history between start_date and end_date for a symbol.
    Returns a DataFrame with timestamp (UTC), OHLCV, symbol.
    """

    symbol = _ensure_valid_symbol(symbol)

    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    all_candles = []
    attempts = 0
    max_attempts = 5

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
            attempts = 0
        except ccxt.NetworkError as e:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"[loader] Too many retries fetching {symbol}: {e}")
            sleep_for = retry_sleep * (2 ** attempts)
            print(f"[loader] Network error, retry {attempts}/{max_attempts}, sleeping {sleep_for:.2f}s")
            time.sleep(sleep_for)
            continue
        except Exception as e:
            raise RuntimeError(f"[loader] Fatal error fetching {symbol}: {e}")

        if not candles:
            # No more data
            break

        all_candles.extend(candles)

        last_ts = candles[-1][0]

        # Defensive: some exchanges repeat last candle or return single candle
        if len(candles) <= 1 or last_ts == since:
            # Skip ahead one timeframe to avoid infinite loops
            since += timeframe_ms
            time.sleep(0.25)
            continue

        since = last_ts + timeframe_ms
        time.sleep(0.25)

    if not all_candles:
        raise RuntimeError(f"[loader] No OHLCV data returned for {symbol}.")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# -----------------------
# SAVING DATA
# -----------------------

def save_symbol_df(df: pd.DataFrame, timeframe: str = "1h"):
    symbol = df["symbol"].iloc[0].replace("/", "_")
    fname = f"{symbol}_{timeframe}.csv"
    path = os.path.join(DATA_DIR, fname)

    df.to_csv(path, index=False)

    meta = {
        "symbol": df["symbol"].iloc[0],
        "rows": len(df),
        "start": df["timestamp"].iloc[0].isoformat(),
        "end": df["timestamp"].iloc[-1].isoformat(),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    meta_path = path.replace(".csv", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[loader] Saved: {path}   ({meta['rows']} rows). Meta: {meta_path}")


def save_symbol_list(
    symbols: List[str],
    timeframe: str = "1h",
    start_date: str = "2017-01-01",
    end_date: str = "2025-01-01",
    overwrite: bool = False,
):
    """
    Downloads and saves OHLCV for a list of symbols.
    """
    for sym in symbols:
        norm_sym = _ensure_valid_symbol(sym)
        fname = f"{norm_sym.replace('/', '_')}_{timeframe}.csv"
        fpath = os.path.join(DATA_DIR, fname)

        if os.path.exists(fpath) and not overwrite:
            print(f"[loader] Skipping existing {fpath}")
            continue

        print(f"[loader] Downloading {norm_sym} ({start_date} → {end_date})...")
        df = fetch_ohlcv_full(norm_sym, timeframe=timeframe, start_date=start_date, end_date=end_date)
        if df["close"].isnull().any():
            df = df.dropna(subset=["close"]).reset_index(drop=True)

        save_symbol_df(df, timeframe=timeframe)


# -----------------------
# LOAD FEATURE FILES
# -----------------------

def load_all_symbols(timeframe: str = "1h") -> dict:
    """
    Loads already processed *_features.csv files into a dict[symbol] -> DataFrame.
    """
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(f"_{timeframe}_features.csv")
    ]

    if not files:
        raise RuntimeError(
            f"[loader] No feature files found in {DATA_DIR}. "
            f"Run: python -m src.data.features"
        )

    data_dict = {}
    for f in files:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        symbol = f.replace(f"_{timeframe}_features.csv", "").replace("_", "/")
        data_dict[symbol] = df

    print(f"[loader] Loaded {len(data_dict)} feature datasets.")
    return data_dict


if __name__ == "__main__":
    TOP_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
        "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"
    ]

    TF = os.environ.get("FINRL_OHLCV_TF", "1h")
    START = os.environ.get("FINRL_OHLCV_START", "2017-09-01")
    END = os.environ.get("FINRL_OHLCV_END", "2025-09-01")
    OVERWRITE = os.environ.get("FINRL_OHLCV_OVERWRITE", "0") == "1"

    save_symbol_list(TOP_SYMBOLS, timeframe=TF, start_date=START, end_date=END, overwrite=OVERWRITE)
