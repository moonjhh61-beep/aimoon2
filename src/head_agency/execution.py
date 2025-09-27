"""Helper functions for executing research runner code blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DATA_DIR = PROJECT_ROOT / "binance_futures_data" / "data"
_PROCESSED_DIR = _DATA_DIR / "processed"
_TYPE_TO_FOLDER = {
    "klines": "klines",
    "mark_price": "markPriceKlines",
    "index_price": "indexPriceKlines",
    "premium_index": "premiumIndexKlines",
    "funding_rate": "fundingRate",
}
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _normalise_columns(df: pd.DataFrame, *, folder_name: str) -> pd.DataFrame:
    """Ensure timestamp column exists for klines-like datasets."""

    if "timestamp" not in df.columns:
        if folder_name in {
            "klines",
            "markPriceKlines",
            "indexPriceKlines",
            "premiumIndexKlines",
        }:
            df = df.copy()
            df.columns = _KLINE_COLUMNS[: len(df.columns)]
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        elif folder_name == "fundingRate":
            df = df.copy()
            df.columns = [
                "funding_time",
                "funding_rate",
                "mark_price",
                "index_price",
            ][: len(df.columns)]
            df["timestamp"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
        else:
            raise ValueError("Unable to infer timestamp column for dataset")
    else:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def load_dataset(
    *,
    data_type: str,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load processed Binance futures data for the given configuration."""

    folder_name = _TYPE_TO_FOLDER.get(data_type)
    if not folder_name:
        raise ValueError(f"Unsupported data_type '{data_type}'")

    symbols = list(symbols)
    if not symbols:
        raise ValueError("symbols must contain at least one entry")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol_dir = _PROCESSED_DIR / folder_name / symbol
        if not symbol_dir.exists():
            continue
        for file_path in sorted(symbol_dir.glob("*.parquet")):
            try:
                raw_df = pd.read_parquet(file_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Failed to read {file_path}: {exc}")
                continue

            df = _normalise_columns(raw_df, folder_name=folder_name)
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            df = df[mask]
            if df.empty:
                continue
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            else:
                df["symbol"] = df["symbol"].fillna(symbol)
            frames.append(df)

    if not frames:
        raise ValueError(
            f"No data returned for type={data_type}, symbols={symbols}, start={start_date}, end={end_date}."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["timestamp", "symbol"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


__all__ = ["load_dataset", "np", "pd"]
