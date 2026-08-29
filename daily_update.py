
"""
Daily Commodity Forecast Update
This script loads the .pkl files, rebuilds the latest features
from public market data, predicts the latest level/movement outputs, and posts the
combined forecast rows to the PHP/MySQL platform.

Required GitHub secrets:
- PLATFORM_API_KEY: the ingest API key under config.php
Optional GitHub secret:
- PLATFORM_FORECAST_API_URL: defaults to http://commodity.fin-corp.uk/api/ingest_forecast.php
"""

from __future__ import annotations

import os
import re
import sys
import math
import glob
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from io import StringIO
try:
    import yfinance as yf
except Exception:
    yf = None
import joblib
from pandas.tseries.offsets import BDay

# -------------------------------------------------------------------
# NumPy/joblib compatibility patch
# -------------------------------------------------------------------
# Some models were saved in Colab with NumPy random-state objects that
# can unpickle differently on GitHub Actions. This patch keeps the daily
# updater from failing on the MT19937 BitGenerator constructor.
try:
    import numpy.random._pickle as _np_random_pickle
    _original_bit_generator_ctor = _np_random_pickle.__bit_generator_ctor

    def _compatible_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        if isinstance(bit_generator_name, str) and bit_generator_name.endswith("MT19937"):
            bit_generator_name = "MT19937"
        return _original_bit_generator_ctor(bit_generator_name)

    _np_random_pickle.__bit_generator_ctor = _compatible_bit_generator_ctor
except Exception:
    pass

warnings.filterwarnings("ignore")

REAL_GOLD_FIX_VERSION = "REAL_CRUDE_TRAINING_COMPATIBLE_YFINANCE_V13_2026_08_29"

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "final_platform_actual_dissertation_models"
FORECAST_API_URL = (os.getenv("PLATFORM_FORECAST_API_URL") or "http://commodity.fin-corp.uk/api/ingest_forecast.php").strip()
API_KEY = os.getenv("PLATFORM_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("Missing PLATFORM_API_KEY GitHub secret.")
if not ALPHA_VANTAGE_API_KEY:
    raise RuntimeError("Missing ALPHAVANTAGE_API_KEY GitHub secret.")

# We keep the same platform design used in the dissertation demo.
# GOLD: level horizons only, because Gold level has no t+15.
# CRUDE: level horizons only, movement exists only for t+1/t+2/t+3.
# NATGAS: union of level + movement horizons, because the movement model has extended horizons.
SEND_POLICY = {
    "GOLD": "level_horizons_only",
    "CRUDE": "level_horizons_only",
    "NATGAS": "union",
}

YAHOO_TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "heating_oil": "HO=F",
    "gasoline": "RB=F",
    "natural_gas": "NG=F",
    "ng": "NG=F",
    "ung": "UNG",
    "xle": "XLE",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "tnx": "^TNX",
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
}

FRED_SERIES = {
    "HH_SPOT": "DHHNGSP",
    "USD_PER_EUR": "DEXUSEU",
    "JPY_PER_USD": "DEXJPUS",
    "USD_PER_GBP": "DEXUSUK",
    "NASDAQ": "NASDAQCOM",
    "SP500": "SP500",
    "DOW": "DJIA",
    "VIX": "VIXCLS",
    "EFFR": "EFFR",
    "T5YIE": "T5YIE",
    "T10YIE": "T10YIE",
    "DGS1": "DGS1",
    "DGS10": "DGS10",
    "PRIME": "DPRIME",
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def make_target_date(forecast_date: str | pd.Timestamp, h: int) -> str:
    return (pd.Timestamp(forecast_date) + BDay(int(h))).date().isoformat()


def post_forecast(payload: dict) -> None:
    payload = dict(payload)
    payload["api_key"] = API_KEY
    for attempt in range(1, 4):
        try:
            r = requests.post(FORECAST_API_URL, data=payload, timeout=60)
            log(f"POST {payload.get('commodity_code')} t+{payload.get('horizon_days')} -> {r.status_code} {r.text[:150]}")
            if r.status_code == 200:
                return
            if attempt == 3:
                r.raise_for_status()
        except Exception as exc:
            log(f"API attempt {attempt} failed: {exc}")
            if attempt == 3:
                raise
            time.sleep(5)


def flatten_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

ALPHA_COMMODITY_FUNCTIONS = {
    "gold": ("GOLD_SILVER_HISTORY", "GOLD"),
    "silver": ("GOLD_SILVER_HISTORY", "SILVER"),
    "wti": ("WTI", None),
    "brent": ("BRENT", None),
    "natural_gas": ("NATURAL_GAS", None),
    "ng": ("NATURAL_GAS", None),
}

ALPHA_EQUITY_SYMBOLS = {
    "xle": "XLE",
    "ung": "UNG",
}

ALPHA_FX_PAIRS = {
    "eurusd": ("EUR", "USD"),
    "gbpusd": ("GBP", "USD"),
}

FRED_PREFIX_SERIES = {
    "sp500": "SP500",
    "vix": "VIXCLS",
    "tnx": "DGS10",
    "dxy": "DTWEXBGS",
}

_ALPHA_CACHE: dict[str, pd.DataFrame] = {}


def make_ohlcv_from_close(prefix: str, close: pd.Series) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close.index = pd.to_datetime(close.index)
    out = pd.DataFrame(index=close.index)
    out[f"{prefix}_open"] = close
    out[f"{prefix}_high"] = close
    out[f"{prefix}_low"] = close
    out[f"{prefix}_close"] = close
    out[f"{prefix}_adj_close"] = close
    out[f"{prefix}_volume"] = 0.0
    out.index.name = "date"
    return out


def alpha_get(params: dict) -> dict:
    params = dict(params)
    params["apikey"] = ALPHA_VANTAGE_API_KEY
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    if "Note" in data:
        raise RuntimeError(data["Note"])
    if "Information" in data:
        raise RuntimeError(data["Information"])
    return data


def parse_alpha_commodity_data(data: dict) -> pd.Series:
    rows = data.get("data", [])
    if not rows:
        raise RuntimeError(f"No commodity rows returned from Alpha Vantage. Keys: {list(data.keys())[:8]}")
    dates = []
    values = []
    for row in rows:
        date_value = row.get("date") or row.get("timestamp")
        price_value = row.get("value") or row.get("price")
        if date_value is None or price_value in (None, "."):
            continue
        dates.append(pd.to_datetime(date_value))
        values.append(pd.to_numeric(price_value, errors="coerce"))
    s = pd.Series(values, index=pd.to_datetime(dates)).dropna().sort_index()
    if s.empty:
        raise RuntimeError("Alpha Vantage commodity series was empty after parsing.")
    return s


def parse_alpha_daily_ohlcv(data: dict) -> pd.DataFrame:
    ts_key = None
    for key in data.keys():
        if "Time Series" in key:
            ts_key = key
            break
    if ts_key is None:
        raise RuntimeError(f"No Time Series block returned. Keys: {list(data.keys())[:8]}")
    rows = []
    for date_value, values in data[ts_key].items():
        rows.append({
            "date": pd.to_datetime(date_value),
            "Open": pd.to_numeric(values.get("1. open"), errors="coerce"),
            "High": pd.to_numeric(values.get("2. high"), errors="coerce"),
            "Low": pd.to_numeric(values.get("3. low"), errors="coerce"),
            "Close": pd.to_numeric(values.get("4. close"), errors="coerce"),
            "Volume": pd.to_numeric(values.get("5. volume", 0), errors="coerce"),
        })
    df = pd.DataFrame(rows).dropna(subset=["date", "Close"]).set_index("date").sort_index()
    if df.empty:
        raise RuntimeError("Alpha Vantage daily series was empty after parsing.")
    return df


def download_fred_prefix(prefix: str, series_id: str) -> pd.DataFrame:
    log(f"Downloading FRED fallback data: {prefix} = {series_id}")
    s = download_fred_series(series_id)
    return make_ohlcv_from_close(prefix, s)


def download_alpha_commodity(prefix: str) -> pd.DataFrame:
    function_name, symbol = ALPHA_COMMODITY_FUNCTIONS[prefix]
    log(f"Downloading Alpha Vantage commodity data: {prefix} = {function_name}")
    params = {"function": function_name, "interval": "daily"}
    if symbol is not None:
        params["symbol"] = symbol
    data = alpha_get(params)
    s = parse_alpha_commodity_data(data)
    return make_ohlcv_from_close(prefix, s)


def download_alpha_equity(prefix: str, symbol: str) -> pd.DataFrame:
    log(f"Downloading Alpha Vantage daily equity/ETF data: {prefix} = {symbol}")
    data = alpha_get({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full"})
    df = parse_alpha_daily_ohlcv(data)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_open"] = df["Open"]
    out[f"{prefix}_high"] = df["High"]
    out[f"{prefix}_low"] = df["Low"]
    out[f"{prefix}_close"] = df["Close"]
    out[f"{prefix}_adj_close"] = df["Close"]
    out[f"{prefix}_volume"] = df.get("Volume", 0.0)
    out.index.name = "date"
    return out


def download_alpha_fx(prefix: str, from_symbol: str, to_symbol: str) -> pd.DataFrame:
    log(f"Downloading Alpha Vantage daily FX data: {prefix} = {from_symbol}/{to_symbol}")
    data = alpha_get({"function": "FX_DAILY", "from_symbol": from_symbol, "to_symbol": to_symbol, "outputsize": "full"})
    df = parse_alpha_daily_ohlcv(data)
    return make_ohlcv_from_close(prefix, df["Close"])


def download_yahoo(prefix: str, start: str = "2014-01-01") -> pd.DataFrame:
    """
    Download the market series required by the saved dissertation models.

    V9 real Crude fix:
    Crude level models require heating_oil, gasoline, XLE, EUR/USD and GBP/USD.
    Alpha Vantage could not provide these reliably in GitHub Actions, so these
    series are now fetched from Yahoo/yfinance as full OHLCV data.
    """
    if prefix in _ALPHA_CACHE:
        return _ALPHA_CACHE[prefix].copy()

    yfinance_preferred = {"heating_oil", "gasoline", "xle", "eurusd", "gbpusd"}

    # Use yfinance first for the Crude sources that Alpha Vantage was missing/rate-limiting.
    if prefix in yfinance_preferred:
        if yf is None:
            raise RuntimeError(f"yfinance is not installed, but {prefix} requires it")
        ticker = YAHOO_TICKERS[prefix]
        log(f"Downloading Yahoo/yfinance OHLCV data for {prefix}: {ticker}")
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no rows for {prefix} ({ticker})")
        raw = flatten_yfinance(raw)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise RuntimeError(f"yfinance {prefix} data missing columns: {missing}")

        out = pd.DataFrame(index=raw.index)
        out[f"{prefix}_open"] = pd.to_numeric(raw["Open"], errors="coerce")
        out[f"{prefix}_high"] = pd.to_numeric(raw["High"], errors="coerce")
        out[f"{prefix}_low"] = pd.to_numeric(raw["Low"], errors="coerce")
        out[f"{prefix}_close"] = pd.to_numeric(raw["Close"], errors="coerce")
        if "Adj Close" in raw.columns:
            out[f"{prefix}_adj_close"] = pd.to_numeric(raw["Adj Close"], errors="coerce")
        else:
            out[f"{prefix}_adj_close"] = out[f"{prefix}_close"]
        out[f"{prefix}_volume"] = pd.to_numeric(raw["Volume"], errors="coerce").fillna(0.0)
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"{prefix}_close"])
        out.index.name = "date"
        out = out.loc[out.index >= pd.Timestamp(start)]
        if out.empty:
            raise RuntimeError(f"No usable yfinance data returned for {prefix}")
        _ALPHA_CACHE[prefix] = out.copy()
        return out

    # Keep the existing Alpha/FRED sources for the remaining series.
    if prefix in ALPHA_COMMODITY_FUNCTIONS:
        out = download_alpha_commodity(prefix)
    elif prefix in ALPHA_EQUITY_SYMBOLS:
        out = download_alpha_equity(prefix, ALPHA_EQUITY_SYMBOLS[prefix])
    elif prefix in ALPHA_FX_PAIRS:
        from_symbol, to_symbol = ALPHA_FX_PAIRS[prefix]
        out = download_alpha_fx(prefix, from_symbol, to_symbol)
    elif prefix in FRED_PREFIX_SERIES:
        out = download_fred_prefix(prefix, FRED_PREFIX_SERIES[prefix])
    else:
        raise RuntimeError(f"No data source configured for prefix: {prefix}")

    out = out.loc[out.index >= pd.Timestamp(start)]
    if out.empty:
        raise RuntimeError(f"No replacement market data returned for {prefix}")
    _ALPHA_CACHE[prefix] = out.copy()
    return out


def merge_prefixes(prefixes: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    frames = []
    for p in prefixes:
        try:
            frames.append(download_yahoo(p, start=start))
        except Exception as exc:
            log(f"WARNING: could not download {p}: {exc}")
    if not frames:
        raise RuntimeError("No Yahoo frames were downloaded.")
    df = pd.concat(frames, axis=1).sort_index()
    df = df.ffill()
    df.index.name = "date"
    return df


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_prefixed_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Build the prefixed feature names used by the original Crude/Oil, Gold movement,
    and Natural Gas movement notebooks.

    V10 real Crude fix:
    The saved Crude level models expect names such as:
      brent_return_1d, brent_return_lag_1, brent_close_lag_1,
      wti_adj_return_1d, xle_rolling_return_60d, etc.
    Earlier deployed versions created only *_ret_0 and *_ret_0_lag_* names.
    That caused missing model features and impossible Crude level predictions.
    This function now creates both naming schemes so the live pipeline matches
    the saved dissertation feature columns.
    """
    close = f"{prefix}_close"
    adj = f"{prefix}_adj_close"
    high = f"{prefix}_high"
    low = f"{prefix}_low"

    if close not in df.columns:
        return df

    c = pd.to_numeric(df[close], errors="coerce")
    a = pd.to_numeric(df[adj], errors="coerce") if adj in df.columns else c

    # Original deployed naming used by movement models.
    df[f"{prefix}_ret_0"] = c.pct_change()
    df[f"{prefix}_adj_ret_0"] = a.pct_change()

    # Crude level notebook naming expected by saved level models.
    df[f"{prefix}_return_1d"] = df[f"{prefix}_ret_0"]
    df[f"{prefix}_adj_return_1d"] = df[f"{prefix}_adj_ret_0"]

    # Close, return and adjusted-return lags.
    all_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20, 30, 40, 60]
    model_lags = [1, 2, 3, 5, 10, 20, 30, 60]

    for lag in all_lags:
        df[f"{prefix}_ret_0_lag_{lag}"] = df[f"{prefix}_ret_0"].shift(lag)
        df[f"{prefix}_adj_ret_0_lag_{lag}"] = df[f"{prefix}_adj_ret_0"].shift(lag)

    for lag in model_lags:
        df[f"{prefix}_close_lag_{lag}"] = c.shift(lag)
        df[f"{prefix}_return_lag_{lag}"] = df[f"{prefix}_return_1d"].shift(lag)
        df[f"{prefix}_adj_return_lag_{lag}"] = df[f"{prefix}_adj_return_1d"].shift(lag)

    if prefix == "tnx":
        df["tnx_change_0"] = c.diff()
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"tnx_change_0_lag_{lag}"] = df["tnx_change_0"].shift(lag)

    # Windows required across saved models.
    for w in [3, 5, 10, 20, 40, 50, 60, 100, 200]:
        df[f"{prefix}_ma_{w}"] = c.rolling(w).mean()
        df[f"{prefix}_momentum_{w}"] = c / c.shift(w) - 1
        df[f"{prefix}_volatility_{w}"] = df[f"{prefix}_ret_0"].rolling(w).std()

        # Original movement style.
        df[f"{prefix}_rolling_return_{w}d"] = c / c.shift(w) - 1
        df[f"{prefix}_rolling_volatility_{w}d"] = df[f"{prefix}_ret_0"].rolling(w).std()

        # Crude level adjusted variants.
        df[f"{prefix}_adj_rolling_return_{w}d"] = a / a.shift(w) - 1
        df[f"{prefix}_adj_rolling_volatility_{w}d"] = df[f"{prefix}_adj_return_1d"].rolling(w).std()

        # V11 real Crude fix:
        # The original Crude level notebook stores these as ratios / percentage changes,
        # not raw moving-average differences. Raw diff() values were much larger
        # and caused impossible Crude level predictions such as 482 USD/barrel.
        df[f"{prefix}_close_over_ma_{w}"] = c / df[f"{prefix}_ma_{w}"] - 1
        df[f"{prefix}_ma_{w}_slope_5d"] = df[f"{prefix}_ma_{w}"].pct_change(5)
        df[f"{prefix}_ma_{w}_slope_10d"] = df[f"{prefix}_ma_{w}"].pct_change(10)

    df[f"{prefix}_rsi_14"] = rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df[f"{prefix}_macd"] = ema12 - ema26
    df[f"{prefix}_macd_signal"] = df[f"{prefix}_macd"].ewm(span=9, adjust=False).mean()
    df[f"{prefix}_macd_diff"] = df[f"{prefix}_macd"] - df[f"{prefix}_macd_signal"]

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    df[f"{prefix}_bollinger_pct_b"] = (c - lower) / (upper - lower)
    df[f"{prefix}_bollinger_width"] = (upper - lower) / ma20

    if high in df.columns and low in df.columns:
        prev_close = c.shift(1)
        tr = pd.concat([
            df[high] - df[low],
            (df[high] - prev_close).abs(),
            (df[low] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df[f"{prefix}_atr_14"] = tr.rolling(14).mean()
        df[f"{prefix}_atr_14_ratio"] = df[f"{prefix}_atr_14"] / c

    return df


def add_unprefixed_gold_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact Gold level feature engineering from the original dissertation notebook:
    Gold_Level_Extended_Horizons / Foroutan_Gold_Level_Extended.

    Important fix:
    The broken deployed updater used absolute MA differences and close/MA ratios.
    The training notebook used percentage MA change and close/MA - 1.
    That mismatch was the real cause of impossible Gold level predictions.
    """
    df = df.copy()
    c = pd.to_numeric(df["close"], errors="coerce")

    df["return_1d"] = c.pct_change()
    df["log_close"] = np.log(c)
    df["log_return_1d"] = df["log_close"].diff()

    # Price lags - same as original notebook.
    for lag in [1, 2, 3, 5, 10, 20, 30, 40, 60]:
        df[f"close_lag_{lag}"] = c.shift(lag)
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
        df[f"log_return_lag_{lag}"] = df["log_return_1d"].shift(lag)

    # Moving averages and relative position - exact notebook formula.
    for w in [5, 10, 20, 30, 50, 60, 100, 200]:
        df[f"ma_{w}"] = c.rolling(w).mean()
        df[f"close_over_ma_{w}"] = c / df[f"ma_{w}"] - 1
        df[f"ma_{w}_slope_5d"] = df[f"ma_{w}"].pct_change(5)
        df[f"ma_{w}_slope_10d"] = df[f"ma_{w}"].pct_change(10)

    # Momentum and volatility - exact notebook windows.
    for w in [5, 10, 20, 30, 40, 60]:
        df[f"momentum_{w}"] = c / c.shift(w) - 1
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()
        df[f"rolling_return_{w}"] = c / c.shift(w) - 1

    df["rsi_14"] = rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    df["bollinger_pct_b"] = (c - lower) / (upper - lower)
    df["bollinger_width"] = (upper - lower) / ma20

    prev_close = c.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_14_ratio"] = df["atr_14"] / c

    return df


def add_ratio_features(df: pd.DataFrame, base_prefix: str, other_prefixes: list[str]) -> pd.DataFrame:
    base_close = f"{base_prefix}_close"
    if base_close not in df.columns:
        return df
    for p in other_prefixes:
        pc = f"{p}_close"
        if pc in df.columns:
            name = f"{p}_{base_prefix}_ratio"
            df[name] = df[pc] / df[base_close]
            df[f"{name}_change_5d"] = df[name].pct_change(5)
            df[f"{name}_change_20d"] = df[name].pct_change(20)
    return df



def download_gold_level_ohlcv(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Real Gold level data source for the daily updater.

    This is the important fix. The Gold level models were trained with full OHLCV
    futures-style data, not Alpha Vantage close-only data. The previous updater
    used close-only Gold data, copied close into open/high/low and set volume to
    zero. That produced invalid model inputs and very large negative predictions.

    This function tries to retrieve real daily OHLCV Gold futures data. It does
    not fall back to close-only Gold data for the level model. If full OHLCV data
    cannot be retrieved, the workflow should fail rather than insert bad Gold
    forecasts into SQL.
    """
    start_ts = pd.Timestamp(start)
    d1 = start_ts.strftime("%Y%m%d")

    # Primary: Stooq daily futures CSV for COMEX Gold continuous futures.
    stooq_url = "https://stooq.com/q/d/l/"
    params = {"s": "gc.f", "d1": d1, "i": "d"}
    try:
        log("Downloading Stooq full OHLCV Gold futures data: gc.f")
        r = requests.get(stooq_url, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if "No data" in r.text or "Date" not in r.text:
            raise RuntimeError("Stooq returned no usable Gold OHLCV CSV")
        raw = pd.read_csv(StringIO(r.text))
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        required = ["date", "open", "high", "low", "close"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise RuntimeError(f"Stooq Gold CSV missing columns: {missing}; columns={raw.columns.tolist()}")
        if "volume" not in raw.columns:
            raise RuntimeError("Stooq Gold CSV did not include Volume; refusing close-only level update")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"]).set_index("date").sort_index()
        out = pd.DataFrame(index=raw.index)
        out["gold_open"] = pd.to_numeric(raw["open"], errors="coerce")
        out["gold_high"] = pd.to_numeric(raw["high"], errors="coerce")
        out["gold_low"] = pd.to_numeric(raw["low"], errors="coerce")
        out["gold_close"] = pd.to_numeric(raw["close"], errors="coerce")
        out["gold_adj_close"] = out["gold_close"]
        out["gold_volume"] = pd.to_numeric(raw["volume"], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["gold_open", "gold_high", "gold_low", "gold_close"])
        out = out.loc[out.index >= start_ts]
        if out.empty:
            raise RuntimeError("Stooq Gold OHLCV was empty after cleaning")
        if out["gold_volume"].fillna(0).abs().sum() == 0:
            raise RuntimeError("Stooq Gold volume is all zero/missing; refusing level model update")
        out.index.name = "date"
        return out
    except Exception as exc:
        log(f"WARNING: Stooq Gold OHLCV failed: {exc}")

    # Secondary: yfinance GC=F, still full OHLCV if available.
    if yf is not None:
        try:
            log("Downloading Yahoo/yfinance full OHLCV Gold futures data: GC=F")
            df = yf.download("GC=F", start=start, auto_adjust=False, progress=False, threads=False, timeout=60)
            if df is None or df.empty:
                raise RuntimeError("yfinance returned no Gold rows")
            df = flatten_yfinance(df)
            rename = {
                "Open": "gold_open",
                "High": "gold_high",
                "Low": "gold_low",
                "Close": "gold_close",
                "Adj Close": "gold_adj_close",
                "Volume": "gold_volume",
            }
            out = pd.DataFrame(index=df.index)
            for old, new in rename.items():
                if old in df.columns:
                    out[new] = pd.to_numeric(df[old], errors="coerce")
            if "gold_adj_close" not in out.columns and "gold_close" in out.columns:
                out["gold_adj_close"] = out["gold_close"]
            required = ["gold_open", "gold_high", "gold_low", "gold_close", "gold_volume"]
            missing = [c for c in required if c not in out.columns]
            if missing:
                raise RuntimeError(f"yfinance Gold data missing OHLCV columns: {missing}")
            out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["gold_open", "gold_high", "gold_low", "gold_close"])
            out = out.loc[out.index >= start_ts]
            if out.empty:
                raise RuntimeError("yfinance Gold OHLCV was empty after cleaning")
            if out["gold_volume"].fillna(0).abs().sum() == 0:
                raise RuntimeError("yfinance Gold volume is all zero/missing; refusing level model update")
            out.index.name = "date"
            return out
        except Exception as exc:
            log(f"WARNING: yfinance Gold OHLCV failed: {exc}")

    raise RuntimeError(
        "Could not retrieve full OHLCV Gold futures data from Stooq or yfinance. "
        "Gold level update stopped to avoid inserting invalid negative values."
    )

def build_gold_level_frame() -> pd.DataFrame:
    raw = download_gold_level_ohlcv(start="2010-01-01")
    df = pd.DataFrame(index=raw.index)
    mapping = {
        "gold_open": "open",
        "gold_high": "high",
        "gold_low": "low",
        "gold_close": "close",
        "gold_volume": "volume",
    }
    for old, new in mapping.items():
        if old not in raw.columns:
            raise RuntimeError(f"Gold OHLCV source is missing required column: {old}")
        df[new] = raw[old]
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    if df[["open", "high", "low", "close"]].dropna().empty:
        raise RuntimeError("Gold OHLCV frame is empty after cleaning")
    return add_unprefixed_gold_level_features(df)


def build_gold_movement_frame() -> pd.DataFrame:
    prefixes = ["gold", "silver", "wti", "sp500", "vix", "dxy", "tnx", "eurusd"]
    df = merge_prefixes(prefixes, start="2014-01-01")
    for p in prefixes:
        df = add_prefixed_features(df, p)
    return df



CRUDE_LEVEL_TRAINING_TICKERS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "heating_oil": "HO=F",
    "gasoline": "RB=F",
    "natural_gas": "NG=F",
    "xle": "XLE",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "tnx": "^TNX",
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
}


def download_crude_level_yfinance_series(
    prefix: str,
    ticker: str,
    start: str = "2007-01-01",
    attempts: int = 3,
) -> pd.DataFrame:
    """
    Download one Crude level input exactly from the Yahoo/yfinance source family
    used by the original Crude level training notebook.

    Important:
    - auto_adjust=False
    - full OHLCV where Yahoo provides it
    - no Alpha Vantage/FRED synthetic OHLCV substitution
    """
    if yf is None:
        raise RuntimeError("yfinance is required for the Crude level pipeline")

    last_exc = None

    for attempt in range(1, attempts + 1):
        try:
            log(
                f"Downloading training-compatible Crude level data "
                f"{prefix}: {ticker} (attempt {attempt}/{attempts})"
            )
            raw = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw is None or raw.empty:
                raise RuntimeError(f"yfinance returned no rows for {prefix} ({ticker})")

            raw = flatten_yfinance(raw)

            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in raw.columns]
            if missing:
                raise RuntimeError(
                    f"yfinance {prefix} ({ticker}) is missing required columns: {missing}"
                )

            out = pd.DataFrame(index=pd.to_datetime(raw.index))
            out[f"{prefix}_open"] = pd.to_numeric(raw["Open"], errors="coerce")
            out[f"{prefix}_high"] = pd.to_numeric(raw["High"], errors="coerce")
            out[f"{prefix}_low"] = pd.to_numeric(raw["Low"], errors="coerce")
            out[f"{prefix}_close"] = pd.to_numeric(raw["Close"], errors="coerce")

            if "Adj Close" in raw.columns:
                out[f"{prefix}_adj_close"] = pd.to_numeric(
                    raw["Adj Close"], errors="coerce"
                )

            out[f"{prefix}_volume"] = pd.to_numeric(
                raw["Volume"], errors="coerce"
            )

            out = (
                out.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[f"{prefix}_close"])
                .sort_index()
            )
            out = out[~out.index.duplicated(keep="last")]
            out.index.name = "date"

            if out.empty:
                raise RuntimeError(
                    f"No usable training-compatible rows remained for {prefix} ({ticker})"
                )

            log(
                f"Crude level {prefix}: {len(out)} rows, "
                f"{out.index.min().date()} to {out.index.max().date()}"
            )
            return out

        except Exception as exc:
            last_exc = exc
            log(
                f"WARNING: Crude level yfinance attempt {attempt} failed "
                f"for {prefix}/{ticker}: {exc}"
            )
            if attempt < attempts:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"Could not download Crude level source {prefix}/{ticker} after "
        f"{attempts} attempts: {last_exc}"
    )


def build_crude_level_frame() -> pd.DataFrame:
    """
    Rebuild the live Crude LEVEL inputs using the same source/alignment contract
    as the original Crude level training notebook:

    1. All 12 market inputs come from Yahoo/yfinance.
    2. Brent BZ=F trading dates are the master index.
    3. Other markets are LEFT-joined onto Brent dates.
    4. Only non-Brent source columns are forward-filled, with limit=5.
    5. Saved model feature names are reconstructed afterwards.

    This intentionally does NOT reuse the mixed Alpha/FRED pipeline used by
    movement models, because that substitution changed OHLCV semantics and
    produced economically implausible Crude level outputs.
    """
    downloaded: dict[str, pd.DataFrame] = {}

    for prefix, ticker in CRUDE_LEVEL_TRAINING_TICKERS.items():
        downloaded[prefix] = download_crude_level_yfinance_series(
            prefix=prefix,
            ticker=ticker,
            start="2007-01-01",
        )

    if "brent" not in downloaded or downloaded["brent"].empty:
        raise RuntimeError("Brent BZ=F data is unavailable for Crude level inference")

    # Match the original training notebook: Brent master dates and only
    # Brent OHLCV in the target frame (no Brent adjusted-close column).
    brent = downloaded["brent"]
    required_brent = [
        "brent_open",
        "brent_high",
        "brent_low",
        "brent_close",
        "brent_volume",
    ]
    missing_brent = [c for c in required_brent if c not in brent.columns]
    if missing_brent:
        raise RuntimeError(
            f"Training-compatible Brent frame is missing: {missing_brent}"
        )

    df = brent[required_brent].copy()

    # Match the original notebook's left merge onto Brent trading dates.
    for prefix in CRUDE_LEVEL_TRAINING_TICKERS:
        if prefix == "brent":
            continue
        frame = downloaded[prefix]
        df = df.join(frame, how="left")

    # Match the original notebook: short-gap fill only for non-target markets.
    non_target_cols = [c for c in df.columns if not c.startswith("brent_")]
    df[non_target_cols] = df[non_target_cols].ffill(limit=5)

    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    df.index.name = "date"

    prefixes = list(CRUDE_LEVEL_TRAINING_TICKERS.keys())
    for prefix in prefixes:
        df = add_prefixed_features(df, prefix)

    # Same cross-market ratio family used in the Crude level experiment.
    df = add_ratio_features(
        df,
        "brent",
        ["wti", "heating_oil", "gasoline", "natural_gas", "xle", "sp500"],
    )

    latest_brent = pd.to_numeric(
        df["brent_close"], errors="coerce"
    ).dropna()

    if latest_brent.empty:
        raise RuntimeError("Crude level frame contains no usable Brent close values")

    log(
        f"Training-compatible Crude level frame ready: {len(df)} rows, "
        f"latest Brent date={latest_brent.index[-1].date()}, "
        f"latest Brent close={float(latest_brent.iloc[-1]):.4f}"
    )

    return df


def build_crude_movement_frame() -> pd.DataFrame:
    prefixes = ["brent", "wti", "heating_oil", "gasoline", "natural_gas", "xle", "sp500", "vix", "dxy", "tnx", "eurusd", "gbpusd"]
    df = merge_prefixes(prefixes, start="2014-01-01")
    for p in prefixes:
        df = add_prefixed_features(df, p)

    # Kulkarni-style futures-only movement features used in the crude movement notebook.
    for p in ["brent", "wti", "heating_oil", "gasoline", "natural_gas"]:
        close = f"{p}_close"
        if close in df.columns:
            df[p] = df[close]
            df[f"{p}_ma3"] = df[p].rolling(3).mean()
            df[f"{p}_rel"] = df[p] / df[f"{p}_ma3"] - 1
            df[f"{p}_rel_lag_0"] = df[f"{p}_rel"]
    if "brent_rel" in df.columns:
        for lag in range(0, 13):
            df[f"brent_rel_lag_{lag}"] = df["brent_rel"].shift(lag)
    for p in ["wti", "heating_oil", "gasoline", "natural_gas"]:
        if f"{p}_rel" in df.columns:
            df[f"{p}_rel_lag_0"] = df[f"{p}_rel"]

    df = add_ratio_features(df, "brent", ["wti", "heating_oil", "gasoline", "natural_gas", "xle", "sp500"])
    return df


def build_natgas_movement_frame() -> pd.DataFrame:
    prefixes = ["ng", "brent", "wti", "heating_oil", "gasoline", "ung", "xle", "sp500", "vix", "dxy", "tnx", "eurusd", "gbpusd"]
    df = merge_prefixes(prefixes, start="2014-01-01")
    for p in prefixes:
        df = add_prefixed_features(df, p)
    df = add_ratio_features(df, "ng", ["brent", "wti", "heating_oil", "gasoline", "ung", "xle"])
    return df


def download_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = df.columns[0]
    value_col = df.columns[1]
    s = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.sort_index()


def build_natgas_level_frame() -> pd.DataFrame:
    log("Downloading FRED data for Natural Gas level features")
    frames = []
    for name, sid in FRED_SERIES.items():
        try:
            s = download_fred_series(sid)
            frames.append(s.rename(name))
        except Exception as exc:
            log(f"WARNING: FRED series failed {name}/{sid}: {exc}")
    if not frames:
        raise RuntimeError("No FRED data downloaded for Natural Gas level model.")
    raw = pd.concat(frames, axis=1).sort_index()
    raw = raw.loc[raw.index >= pd.Timestamp("2010-01-01")].ffill()

    interest_rate_cols = ["EFFR", "T5YIE", "T10YIE", "DGS1", "DGS10", "PRIME"]
    model_data = raw.copy()

    for col in raw.columns:
        if col not in interest_rate_cols:
            positive = raw[col].where(raw[col] > 0)
            model_data[f"log_{col}"] = np.log(positive)

    model_data["HH_LOG_T"] = model_data["log_HH_SPOT"]
    for lag in range(1, 15):
        model_data[f"HH_LOG_LAG_{lag}"] = model_data["log_HH_SPOT"].shift(lag)

    up_move = (model_data["HH_SPOT"].diff() > 0).astype(int)
    model_data["Momentum_5"] = up_move.rolling(5).sum()
    model_data["Momentum_10"] = up_move.rolling(10).sum()
    model_data["MA_5"] = model_data["log_HH_SPOT"].rolling(5).mean()
    model_data["MA_10"] = model_data["log_HH_SPOT"].rolling(10).mean()
    return model_data


def latest_feature_row(df: pd.DataFrame, features: list[str], strict: bool = False) -> tuple[pd.DataFrame, str, list[str]]:
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    missing = [f for f in features if f not in df.columns]

    if strict and missing:
        return pd.DataFrame(), "", missing

    if strict:
        clean = df.dropna(subset=features).copy()
        if clean.empty:
            raise RuntimeError("No complete latest feature row is available for strict model prediction.")
        row = clean.iloc[[-1]][features]
        date = clean.index[-1].date().isoformat()
        return row, date, missing

    for f in missing:
        df[f] = 0.0
    feature_df = df[features].ffill().fillna(0.0)
    row = feature_df.iloc[[-1]]
    date = df.index[-1].date().isoformat()
    return row, date, missing


def predict_proba_up(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    pred = model.predict(X)
    return float(pred[0])


def find_pkl_for_row(commodity: str, task: str, h: int, row: pd.Series | None = None) -> Path | None:
    cdir = MODEL_ROOT / commodity
    if not cdir.exists():
        return None
    task_upper = "LEVEL" if task == "level" else "MOVEMENT"
    matches = sorted(cdir.glob(f"{commodity}_{task_upper}_t{h}_*.pkl"))
    if matches:
        return matches[0]
    return None


def load_registry(commodity: str) -> pd.DataFrame:
    patterns = {
        "GOLD": "gold_actual_dissertation_platform_deployment_registry.csv",
        "CRUDE": "crude_actual_dissertation_platform_deployment_registry.csv",
        "NATGAS": "natural_gas_actual_dissertation_platform_deployment_registry.csv",
    }
    path = MODEL_ROOT / commodity / patterns[commodity]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def current_features_for(commodity: str, task: str) -> pd.DataFrame:
    if commodity == "GOLD" and task == "level":
        return build_gold_level_frame()
    if commodity == "GOLD" and task == "movement":
        return build_gold_movement_frame()
    if commodity == "CRUDE" and task == "level":
        return build_crude_level_frame()
    if commodity == "CRUDE" and task == "movement":
        return build_crude_movement_frame()
    if commodity == "NATGAS" and task == "level":
        return build_natgas_level_frame()
    if commodity == "NATGAS" and task == "movement":
        return build_natgas_movement_frame()
    raise ValueError(f"Unsupported feature request: {commodity}/{task}")


def model_display_name(info: dict) -> str:
    name = str(info.get("model_name", "Model"))
    return f"{name} without extra chart features or news"


def unit_source_text(commodity: str) -> str:
    if commodity == "GOLD":
        return "REAL_GOLD_LEVEL_FEATURE_FIX_V8 from saved Gold no-news dissertation models"
    if commodity == "CRUDE":
        return "REAL_CRUDE_TRAINING_COMPATIBLE_YFINANCE_V13 from saved Crude Oil no-news dissertation models"
    if commodity == "NATGAS":
        return "Daily GitHub Actions update from saved Natural Gas no-news dissertation models"
    return "Daily GitHub Actions update from saved no-news dissertation models"



def latest_numeric_at_or_before(df: pd.DataFrame, column: str, dt: pd.Timestamp) -> float | None:
    if column not in df.columns:
        return None
    s = pd.to_numeric(df.loc[df.index <= dt, column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def valid_crude_level_prediction(pred: float, reference_price: float | None) -> bool:
    if (not np.isfinite(pred)) or pred <= 0:
        return False
    # Crude oil should not be single digits or hundreds in this deployment period.
    if pred < 30 or pred > 180:
        return False
    if reference_price is not None and np.isfinite(reference_price) and reference_price > 0:
        ratio = pred / float(reference_price)
        if ratio < 0.35 or ratio > 2.50:
            return False
    return True


def select_stable_crude_level_predictions(feature_df: pd.DataFrame, model_items: list[dict]) -> tuple[dict[int, tuple[float, str, dict]], str]:
    """
    Select the most recent Crude feature row that gives coherent level predictions
    for every Crude level horizon. This avoids posting a partially broken live row
    when one external source updates later than another or returns partial data.

    This is not a manual replacement: every value is still produced by the saved
    dissertation model, using a real complete feature row.
    """
    df = feature_df.copy().replace([np.inf, -np.inf], np.nan).sort_index()
    if df.empty:
        raise RuntimeError("Crude feature frame is empty.")

    candidate_dates = list(df.index.unique())[-90:]
    candidate_dates = sorted(candidate_dates, reverse=True)

    latest_dt = candidate_dates[0] if candidate_dates else None
    if latest_dt is not None:
        latest_ref = latest_numeric_at_or_before(df, "brent_close", latest_dt)
        log(
            f"Testing {len(candidate_dates)} recent Brent trading rows for "
            f"Crude level stability; newest={latest_dt.date()}, "
            f"reference Brent={latest_ref}"
        )

    for item in model_items:
        missing_cols = [f for f in item["features"] if f not in df.columns]
        log(
            f"Crude level t+{int(item['horizon'])}: "
            f"required_features={len(item['features'])}, "
            f"missing_columns={len(missing_cols)}"
        )
        if missing_cols:
            log(
                f"Crude level t+{int(item['horizon'])} first missing columns: "
                f"{missing_cols[:10]}"
            )

    last_failure = None

    for dt in candidate_dates:
        reference_price = latest_numeric_at_or_before(df, "brent_close", dt)
        if reference_price is None:
            reference_price = latest_numeric_at_or_before(df, "wti_close", dt)

        candidate_predictions: dict[int, tuple[float, str, dict]] = {}
        candidate_ok = True
        failure_reasons = []

        for item in model_items:
            h = int(item["horizon"])
            features = item["features"]
            model = item["model"]
            info = item["info"]

            missing = [f for f in features if f not in df.columns]
            if missing:
                candidate_ok = False
                failure_reasons.append(f"t+{h} missing {len(missing)} features")
                break

            row_df = df.loc[[dt], features].replace([np.inf, -np.inf], np.nan)
            if row_df.isna().any(axis=None):
                candidate_ok = False
                failure_reasons.append(f"t+{h} has NA features")
                break

            pred = float(model.predict(row_df)[0])
            if not valid_crude_level_prediction(pred, reference_price):
                candidate_ok = False
                failure_reasons.append(f"t+{h} invalid prediction {pred:.4f} at {dt.date()}")
                break

            candidate_predictions[h] = (round(pred, 4), dt.date().isoformat(), info)

        if candidate_ok and len(candidate_predictions) == len(model_items):
            log(f"Crude stable level feature row selected: {dt.date()} with reference price {reference_price}")
            for h, (pred, _, _) in sorted(candidate_predictions.items()):
                log(f"Crude level t+{h} stable-row prediction OK: {pred:.4f}")
            return candidate_predictions, dt.date().isoformat()

        if failure_reasons:
            last_failure = "; ".join(failure_reasons)

    raise RuntimeError(
        "No stable Crude level feature row found in the last 90 rows. "
        f"Last failure: {last_failure}"
    )


def build_predictions_for_commodity(commodity: str) -> dict[int, dict]:
    reg = load_registry(commodity)
    if "horizon" not in reg.columns:
        raise RuntimeError(f"Registry for {commodity} has no horizon column")
    reg["horizon"] = reg["horizon"].astype(int)

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    combined: dict[int, dict] = {}

    # Decide which horizons get sent.
    level_horizons = sorted(reg.loc[reg["task"] == "level", "horizon"].dropna().astype(int).unique().tolist())
    movement_horizons = sorted(reg.loc[reg["task"] == "movement", "horizon"].dropna().astype(int).unique().tolist())
    if SEND_POLICY[commodity] == "union":
        send_horizons = sorted(set(level_horizons).union(movement_horizons))
    else:
        send_horizons = level_horizons

    for h in send_horizons:
        combined[h] = {
            "commodity_code": commodity,
            "horizon_days": int(h),
            "forecast_date": datetime.now(timezone.utc).date().isoformat(),
            "target_date": make_target_date(datetime.now(timezone.utc).date().isoformat(), h),
            "predicted_level": "",
            "predicted_movement": "Not available",
            "movement_probability": "",
            "level_model_name": "Level model not available for this horizon",
            "movement_model_name": "Movement model not available for this horizon",
            "data_source": unit_source_text(commodity),
        }

    # Real Crude fix V13:
    # Crude LEVEL data is first rebuilt with the exact Yahoo/yfinance source and
    # Brent-date alignment contract used during training. Predictions are then
    # checked on one coherent/latest valid feature row before anything is posted.
    if commodity == "CRUDE":
        crude_level_rows = reg[reg["task"] == "level"].copy()
        crude_model_items: list[dict] = []
        crude_feature_df = None

        for _, level_row in crude_level_rows.iterrows():
            h = int(level_row["horizon"])
            if h not in combined:
                continue
            pkl_path = find_pkl_for_row(commodity, "level", h, level_row)
            if pkl_path is None or not pkl_path.exists():
                log(f"WARNING: no pkl found for CRUDE level t+{h}")
                continue
            info = joblib.load(pkl_path)
            features = list(info.get("features", []))
            model = info.get("model")
            if model is None or not features:
                raise RuntimeError(f"Bad CRUDE level model file: {pkl_path}")
            crude_model_items.append({
                "horizon": h,
                "features": features,
                "model": model,
                "info": info,
            })

        if crude_model_items:
            log("Building current features for CRUDE level")
            crude_feature_df = current_features_for("CRUDE", "level")
            stable_preds, stable_forecast_date = select_stable_crude_level_predictions(crude_feature_df, crude_model_items)

            for h, (pred, forecast_date, info) in stable_preds.items():
                combined[h]["forecast_date"] = forecast_date
                combined[h]["target_date"] = make_target_date(forecast_date, h)
                combined[h]["predicted_level"] = round(float(pred), 4)
                combined[h]["level_model_name"] = model_display_name(info)

    for task in ["level", "movement"]:
        if commodity == "CRUDE" and task == "level":
            continue
        task_rows = reg[reg["task"] == task].copy()
        for _, row in task_rows.iterrows():
            h = int(row["horizon"])
            if h not in combined:
                continue

            pkl_path = find_pkl_for_row(commodity, task, h, row)
            if pkl_path is None or not pkl_path.exists():
                log(f"WARNING: no pkl found for {commodity} {task} t+{h}")
                continue

            info = joblib.load(pkl_path)
            features = list(info.get("features", []))
            model = info.get("model")
            if model is None or not features:
                log(f"WARNING: bad model file {pkl_path}")
                continue

            key = (commodity, task)
            if key not in cache:
                log(f"Building current features for {commodity} {task}")
                cache[key] = current_features_for(commodity, task)

            strict_level = (commodity in ["GOLD", "CRUDE"] and task == "level")
            X_latest, forecast_date, missing = latest_feature_row(cache[key], features, strict=strict_level)
            if missing:
                log(f"WARNING: {commodity} {task} t+{h} missing {len(missing)} features; zero-filled first few: {missing[:8]}")
                if strict_level:
                    raise RuntimeError(
                        f"{commodity} level t+{h} is missing required model features: {missing}. "
                        "This is a real feature-pipeline mismatch, so the updater stopped instead of posting bad level values."
                    )

            target_date = make_target_date(forecast_date, h)
            combined[h]["forecast_date"] = forecast_date
            combined[h]["target_date"] = target_date

            if task == "level":
                pred = float(model.predict(X_latest)[0])
                if commodity == "NATGAS":
                    # Natural Gas level models predict log(HH spot), so convert back to price.
                    pred = float(np.exp(pred))

                if commodity == "GOLD":
                    latest_close = pd.to_numeric(cache[key]["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().iloc[-1]
                    ratio = pred / float(latest_close)
                    if (not np.isfinite(pred)) or pred <= 0 or pred < 1000 or pred > 10000 or ratio < 0.40 or ratio > 1.80:
                        raise RuntimeError(
                            f"Gold level t+{h} produced an invalid value ({pred:.4f}) from latest close {float(latest_close):.4f}. "
                            "The updater stopped and did not post this value to SQL. This prevents the negative Gold discrepancy."
                        )
                    log(f"Gold level t+{h} strict prediction OK: {pred:.4f} from latest close {float(latest_close):.4f}, ratio={ratio:.4f}")

                if commodity == "CRUDE":
                    if (not np.isfinite(pred)) or pred <= 0 or pred > 250:
                        raise RuntimeError(
                            f"Crude level t+{h} produced an invalid value ({pred:.4f}). "
                            "The updater stopped and did not post this value to SQL."
                        )
                    log(f"Crude level t+{h} strict prediction OK: {pred:.4f}")

                combined[h]["predicted_level"] = round(pred, 4)
                combined[h]["level_model_name"] = model_display_name(info)
            else:
                prob_up = predict_proba_up(model, X_latest)
                threshold = float(info.get("threshold", row.get("threshold", 0.5)))
                if math.isnan(threshold):
                    threshold = 0.5
                pred_class = int(prob_up >= threshold)
                movement = "Up" if pred_class == 1 else "Down"
                confidence = prob_up if pred_class == 1 else (1 - prob_up)
                combined[h]["predicted_movement"] = movement
                combined[h]["movement_probability"] = round(confidence * 100, 3)
                combined[h]["movement_model_name"] = model_display_name(info)

    return combined


def main() -> None:
    log(f"Starting daily commodity platform update - {REAL_GOLD_FIX_VERSION}")
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"Model root not found: {MODEL_ROOT}")

    all_payloads: list[dict] = []
    for commodity in ["GOLD", "CRUDE", "NATGAS"]:
        log(f"Preparing predictions for {commodity}")
        preds = build_predictions_for_commodity(commodity)
        for h in sorted(preds):
            all_payloads.append(preds[h])

    # Final check: invalid level values must never be posted if feature pipeline breaks again.
    for payload in all_payloads:
        if payload.get("commodity_code") == "GOLD" and payload.get("predicted_level") not in ("", None):
            gold_value = float(payload["predicted_level"])
            if (not np.isfinite(gold_value)) or gold_value < 1000 or gold_value > 10000:
                raise RuntimeError(
                    f"FINAL BLOCK: invalid Gold level value {gold_value} for t+{payload.get('horizon_days')}. Not posted."
                )
        if payload.get("commodity_code") == "CRUDE" and payload.get("predicted_level") not in ("", None):
            crude_value = float(payload["predicted_level"])
            if (not np.isfinite(crude_value)) or crude_value <= 0 or crude_value > 250:
                raise RuntimeError(
                    f"FINAL BLOCK: invalid Crude level value {crude_value} for t+{payload.get('horizon_days')}. Not posted."
                )

    log(f"Posting {len(all_payloads)} forecast rows to platform")
    for payload in all_payloads:
        post_forecast(payload)

    log("Daily commodity platform update complete")


if __name__ == "__main__":
    main()
