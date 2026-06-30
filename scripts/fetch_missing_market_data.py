"""
Download missing daily market data used by the Streamlit trend curves.

Examples:
    python scripts/fetch_missing_market_data.py
    python scripts/fetch_missing_market_data.py --symbols 000002,510500
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MARKET_DATA_DIR, STATS_DATA_DIR, STRATEGY_DATA_DIR
from app.services.excel_strategy_loader import load_excel_strategies
from app.services.price_data import normalize_symbol
from app.services.trend_service import standardize_trades


PRICE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "pctChg,turn,tradestatus,isST"
)


def baostock_code(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def market_label(symbol: str) -> str:
    return "SH" if symbol.startswith(("5", "6", "9")) else "SZ"


def exchange_symbol(symbol: str) -> str:
    prefix = "SHSE" if market_label(symbol) == "SH" else "SZSE"
    return f"{prefix}.{symbol}"


def collect_symbols(include_app_uploads: bool = False) -> set[str]:
    """Collect symbols used by bundled strategy and sample user data."""
    symbols: set[str] = set()

    for dir_path in sorted(STRATEGY_DATA_DIR.iterdir()):
        trades_file = dir_path / "trades.csv"
        if not trades_file.exists():
            continue
        trades = standardize_trades(pd.read_csv(trades_file), dir_path.name)
        symbols.update(trades.loc[
            trades["side"].isin(["buy", "sell", "allotment_in", "allotment_out"]),
            "symbol",
        ])

    excel_trades, _ = load_excel_strategies(STATS_DATA_DIR)
    for strategy_name, trades_df in excel_trades.items():
        trades = standardize_trades(trades_df, strategy_name)
        symbols.update(trades.loc[
            trades["side"].isin(["buy", "sell", "allotment_in", "allotment_out"]),
            "symbol",
        ])

    for path in STATS_DATA_DIR.glob("模拟账户*的记录.xlsx"):
        trades = standardize_trades(pd.read_excel(path), path.name)
        symbols.update(trades.loc[
            trades["side"].isin(["buy", "sell", "allotment_in", "allotment_out"]),
            "symbol",
        ])

    if include_app_uploads:
        for path in (ROOT / "app" / "data" / "trades").glob("*.csv"):
            trades = standardize_trades(pd.read_csv(path), path.name)
            symbols.update(trades.loc[
                trades["side"].isin(["buy", "sell", "allotment_in", "allotment_out"]),
                "symbol",
            ])

    return {normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)}


def find_missing_symbols(symbols: set[str]) -> list[str]:
    return sorted(
        symbol
        for symbol in symbols
        if symbol and not (MARKET_DATA_DIR / f"{symbol}.csv").exists()
    )


def normalize_price_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["code"] = symbol
    out["symbol"] = exchange_symbol(symbol)
    out["name"] = ""
    out["market"] = market_label(symbol)
    out["asset_type"] = ""
    out["date"] = raw["date"]
    out["open"] = raw["open"]
    out["high"] = raw["high"]
    out["low"] = raw["low"]
    out["close"] = raw["close"]
    out["preclose"] = raw["preclose"]
    out["volume"] = raw["volume"]
    out["amount"] = raw["amount"]
    out["pct_chg"] = raw["pctChg"]
    out["turnover"] = raw["turn"]
    out["trade_status"] = raw["tradestatus"]
    out["is_st"] = raw["isST"]
    out["adjust"] = "none"
    out["provider"] = "baostock"
    out["source_function"] = "query_history_k_data_plus"
    return out


def fetch_symbol(symbol: str, start_date: str, end_date: str) -> tuple[bool, str]:
    code = baostock_code(symbol)
    rs = bs.query_history_k_data_plus(
        code,
        PRICE_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        return False, rs.error_msg

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return False, "no rows"

    raw = pd.DataFrame(rows, columns=rs.fields)
    out = normalize_price_frame(raw, symbol)
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MARKET_DATA_DIR / f"{symbol}.csv", index=False)
    return True, f"{len(out)} rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", help="Comma-separated symbols to fetch. Defaults to scanning bundled data.")
    parser.add_argument("--start-date", default="2019-12-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--include-app-uploads", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.symbols:
        symbols = {normalize_symbol(item) for item in args.symbols.split(",")}
        missing = sorted(symbol for symbol in symbols if symbol)
    else:
        symbols = collect_symbols(include_app_uploads=args.include_app_uploads)
        missing = find_missing_symbols(symbols)

    print(f"symbols to fetch: {len(missing)}")
    if not missing:
        return 0

    login = bs.login()
    if getattr(login, "error_code", "0") != "0":
        print(f"baostock login failed: {login.error_msg}")
        return 1

    ok = 0
    failures: list[tuple[str, str]] = []
    try:
        for index, symbol in enumerate(missing, start=1):
            try:
                success, message = fetch_symbol(symbol, args.start_date, args.end_date)
            except Exception as exc:  # noqa: BLE001 - script should continue per symbol.
                success, message = False, str(exc)

            if success:
                ok += 1
                print(f"[{index}/{len(missing)}] {symbol}: OK ({message})")
            else:
                failures.append((symbol, message))
                print(f"[{index}/{len(missing)}] {symbol}: FAILED ({message})")

            time.sleep(args.sleep)
    finally:
        bs.logout()

    print(f"done: {ok} downloaded, {len(failures)} failed")
    if failures:
        print("failed symbols:")
        for symbol, message in failures:
            print(f"  {symbol}: {message}")
    return 0 if ok or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
