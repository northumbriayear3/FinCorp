
"""
Daily Commodity Forecast Update
--------------------------------
This script loads the saved dissertation models, rebuilds the latest no-news features
from public market data, predicts the latest level/movement outputs, and posts the
combined forecast rows to the PHP/MySQL platform.

Required GitHub secrets:
- PLATFORM_API_KEY: your ingest API key from config.php
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
import yfinance as yf
import joblib
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "final_platform_actual_dissertation_models"
FORECAST_API_URL = (os.getenv("PLATFORM_FORECAST_API_URL") or "http://commodity.fin-corp.uk/api/ingest_forecast.php").strip()
API_KEY = os.getenv("PLATFORM_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("Missing PLATFORM_API_KEY GitHub secret.")

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


def download_yahoo(prefix: str, start: str = "2014-01-01") -> pd.DataFrame:
    ticker = YAHOO_TICKERS[prefix]
    log(f"Downloading Yahoo data: {prefix} = {ticker}")
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No Yahoo data downloaded for {prefix} ({ticker})")
    df = flatten_yfinance(df)

    rename = {
        "Open": f"{prefix}_open",
        "High": f"{prefix}_high",
        "Low": f"{prefix}_low",
        "Close": f"{prefix}_close",
        "Adj Close": f"{prefix}_adj_close",
        "Volume": f"{prefix}_volume",
    }
    out = pd.DataFrame(index=df.index)
    for old, new in rename.items():
        if old in df.columns:
            out[new] = pd.to_numeric(df[old], errors="coerce")
    if f"{prefix}_adj_close" not in out.columns and f"{prefix}_close" in out.columns:
        out[f"{prefix}_adj_close"] = out[f"{prefix}_close"]
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
    close = f"{prefix}_close"
    adj = f"{prefix}_adj_close"
    high = f"{prefix}_high"
    low = f"{prefix}_low"

    if close not in df.columns:
        return df

    c = df[close]
    df[f"{prefix}_ret_0"] = c.pct_change()
    if adj in df.columns:
        df[f"{prefix}_adj_ret_0"] = df[adj].pct_change()
    else:
        df[f"{prefix}_adj_ret_0"] = df[f"{prefix}_ret_0"]

    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20, 30, 40, 60]:
        df[f"{prefix}_ret_0_lag_{lag}"] = df[f"{prefix}_ret_0"].shift(lag)
        df[f"{prefix}_adj_ret_0_lag_{lag}"] = df[f"{prefix}_adj_ret_0"].shift(lag)

    if prefix == "tnx":
        df["tnx_change_0"] = c.diff()
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"tnx_change_0_lag_{lag}"] = df["tnx_change_0"].shift(lag)

    for w in [3, 5, 10, 20, 50, 60, 100, 200]:
        df[f"{prefix}_ma_{w}"] = c.rolling(w).mean()
        df[f"{prefix}_momentum_{w}"] = c / c.shift(w) - 1
        df[f"{prefix}_volatility_{w}"] = df[f"{prefix}_ret_0"].rolling(w).std()
        df[f"{prefix}_rolling_return_{w}d"] = c / c.shift(w) - 1
        df[f"{prefix}_rolling_volatility_{w}d"] = df[f"{prefix}_ret_0"].rolling(w).std()
        df[f"{prefix}_close_over_ma_{w}"] = c / df[f"{prefix}_ma_{w}"]
        df[f"{prefix}_ma_{w}_slope_5d"] = df[f"{prefix}_ma_{w}"].diff(5)
        df[f"{prefix}_ma_{w}_slope_10d"] = df[f"{prefix}_ma_{w}"].diff(10)

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
    # Input columns: open, high, low, close, volume
    c = df["close"]
    df["return_1d"] = c.pct_change()
    df["log_close"] = np.log(c)
    df["log_return_1d"] = df["log_close"].diff()

    for lag in [1, 2, 3, 5, 10, 20, 30, 40, 60]:
        df[f"close_lag_{lag}"] = c.shift(lag)
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
        df[f"log_return_lag_{lag}"] = df["log_return_1d"].shift(lag)

    for w in [5, 10, 20, 30, 50, 100, 200]:
        df[f"ma_{w}"] = c.rolling(w).mean()
        df[f"close_over_ma_{w}"] = c / df[f"ma_{w}"]
        df[f"ma_{w}_slope_5d"] = df[f"ma_{w}"].diff(5)
        df[f"ma_{w}_slope_10d"] = df[f"ma_{w}"].diff(10)
        df[f"momentum_{w}"] = c / c.shift(w) - 1
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()
        df[f"rolling_return_{w}d"] = c / c.shift(w) - 1

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


def build_gold_level_frame() -> pd.DataFrame:
    raw = download_yahoo("gold", start="2010-01-01")
    df = pd.DataFrame(index=raw.index)
    mapping = {
        "gold_open": "open",
        "gold_high": "high",
        "gold_low": "low",
        "gold_close": "close",
        "gold_volume": "volume",
    }
    for old, new in mapping.items():
        if old in raw.columns:
            df[new] = raw[old]
    df = df.ffill()
    return add_unprefixed_gold_level_features(df)


def build_gold_movement_frame() -> pd.DataFrame:
    prefixes = ["gold", "silver", "wti", "sp500", "vix", "dxy", "tnx", "eurusd"]
    df = merge_prefixes(prefixes, start="2014-01-01")
    for p in prefixes:
        df = add_prefixed_features(df, p)
    return df


def build_crude_frame() -> pd.DataFrame:
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


def latest_feature_row(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, str, list[str]]:
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    missing = [f for f in features if f not in df.columns]
    for f in missing:
        df[f] = 0.0
    feature_df = df[features].ffill().fillna(0.0)
    # Use latest row where all requested features are finite after filling.
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
    if commodity == "CRUDE":
        return build_crude_frame()
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
        return "Daily GitHub Actions update from saved Gold no-news dissertation models"
    if commodity == "CRUDE":
        return "Daily GitHub Actions update from saved Crude Oil no-news dissertation models"
    if commodity == "NATGAS":
        return "Daily GitHub Actions update from saved Natural Gas no-news dissertation models"
    return "Daily GitHub Actions update from saved no-news dissertation models"


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

    for task in ["level", "movement"]:
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

            X_latest, forecast_date, missing = latest_feature_row(cache[key], features)
            if missing:
                log(f"WARNING: {commodity} {task} t+{h} missing {len(missing)} features; zero-filled first few: {missing[:8]}")

            target_date = make_target_date(forecast_date, h)
            combined[h]["forecast_date"] = forecast_date
            combined[h]["target_date"] = target_date

            if task == "level":
                pred = float(model.predict(X_latest)[0])
                if commodity == "NATGAS":
                    # Natural Gas level models predict log(HH spot), so convert back to price.
                    pred = float(np.exp(pred))
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
    log("Starting daily commodity platform update")
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"Model root not found: {MODEL_ROOT}")

    all_payloads: list[dict] = []
    for commodity in ["GOLD", "CRUDE", "NATGAS"]:
        log(f"Preparing predictions for {commodity}")
        preds = build_predictions_for_commodity(commodity)
        for h in sorted(preds):
            all_payloads.append(preds[h])

    log(f"Posting {len(all_payloads)} forecast rows to platform")
    for payload in all_payloads:
        post_forecast(payload)

    log("Daily commodity platform update complete")


if __name__ == "__main__":
    main()
