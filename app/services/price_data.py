"""
日度行情读取服务。

按股票代码懒加载 market_data_full_raw/daily_by_symbol 下的 CSV，避免
Streamlit 启动时读入全量行情。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import MARKET_DATA_DIR


def normalize_symbol(raw: Any) -> str:
    """把不同市场前后缀格式统一成 6 位证券代码。"""
    if raw is None:
        return ""
    text = str(raw).strip().upper()
    if not text or text == "NAN":
        return ""

    digit_groups = re.findall(r"\d+", text)
    if not digit_groups:
        return ""
    digits = max(digit_groups, key=len)
    return digits[-6:].zfill(6)


def _find_price_file(code: str) -> Path | None:
    if not MARKET_DATA_DIR.exists():
        return None

    exact = MARKET_DATA_DIR / f"{code}.csv"
    if exact.exists():
        return exact

    stripped = code.lstrip("0") or "0"
    alt = MARKET_DATA_DIR / f"{stripped}.csv"
    if alt.exists():
        return alt

    return None


@lru_cache(maxsize=4096)
def _read_price_rows(code: str) -> tuple[tuple[str, float], ...]:
    file_path = _find_price_file(code)
    if file_path is None:
        return ()

    try:
        df = pd.read_csv(file_path, usecols=["date", "close"], encoding="utf-8-sig")
    except ValueError:
        return ()

    df["date"] = df["date"].astype(str).str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[(df["date"] != "") & df["close"].notna() & (df["close"] > 0)]
    if df.empty:
        return ()

    df = df.sort_values("date", kind="stable")
    return tuple((row.date, float(row.close)) for row in df.itertuples(index=False))


def get_historical_prices(
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """读取指定股票的未复权收盘价。"""
    price_map: dict[str, list[dict[str, Any]]] = {}
    missing_symbols: list[str] = []
    warnings: list[str] = []

    normalized_codes = sorted({normalize_symbol(symbol) for symbol in symbols})
    for code in normalized_codes:
        if not code:
            continue

        all_rows = _read_price_rows(code)
        if not all_rows:
            missing_symbols.append(code)
            continue

        rows = [
            {"date": date_val, "symbol": code, "close": close}
            for date_val, close in all_rows
            if (start_date is None or date_val >= start_date)
            and (end_date is None or date_val <= end_date)
        ]

        if rows:
            price_map[code] = rows
        else:
            missing_symbols.append(code)
            warnings.append(f"{code}: no valid price rows in selected date range")

    return {
        "priceMap": price_map,
        "missingSymbols": missing_symbols,
        "warnings": warnings,
    }
