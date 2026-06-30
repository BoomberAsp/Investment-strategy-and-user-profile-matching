"""
收益曲线计算与 Plotly 图表构建。

核心口径:
    totalAsset_t = cash_t + sum(position_i_t * close_i_t)
    cumulativeReturn_t = (totalAsset_t / initialAsset - 1) * 100
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.services.price_data import get_historical_prices, normalize_symbol


DEFAULT_INITIAL_CAPITAL = 1_000_000.0


def _pick_column(
    df: pd.DataFrame,
    candidates: list[str],
    fallback_index: int | None = None,
) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        hit = normalized.get(name.lower())
        if hit is not None:
            return hit
    if fallback_index is not None and len(df.columns) > fallback_index:
        return df.columns[fallback_index]
    return None


def _parse_date_value(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _map_side(value: Any) -> str | None:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None

    if text in {"BUY", "B"} or "买入" in text or "证券买入" in text:
        return "buy"
    if text in {"SELL", "S"} or "卖出" in text or "证券卖出" in text:
        return "sell"
    if "配售" in text or "新股入账" in text:
        return "allotment_in"
    if "中签股份下账" in text:
        return "allotment_out"
    if "银证转入" in text or "股息入账" in text or "现金转入" in text:
        return "cash_in"
    if "银证转出" in text or "现金转出" in text:
        return "cash_out"

    return None


def _numeric(series: pd.Series | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(default, index=[])
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _extract_initial_capital_from_columns(df: pd.DataFrame) -> float | None:
    for col in df.columns:
        match = re.search(r"(\d+(?:\.\d+)?)万", str(col))
        if match:
            return float(match.group(1)) * 10000
    return None


def standardize_trades(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """标准化客户或策略交易记录为收益曲线计算 schema。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "symbol", "side", "quantity", "price", "amount"])

    date_col = _pick_column(
        df,
        ["trade_date", "trade_time", "datetime", "date", "交易日期", "交收日期"],
        fallback_index=0,
    )
    side_col = _pick_column(
        df,
        ["side", "_side", "action", "action_type", "btype", "操作类型", "业务标示"],
        fallback_index=1,
    )
    symbol_col = _pick_column(df, ["symbol", "stock_code", "证券代码"], fallback_index=2)
    price_col = _pick_column(df, ["price", "vwap", "成交价格", "order_price"], fallback_index=4)
    qty_col = _pick_column(df, ["qty", "quantity", "volume", "成交数量"], fallback_index=5)
    amount_col = _pick_column(df, ["amount", "成交金额"])

    if date_col is None or side_col is None:
        return pd.DataFrame(columns=["date", "symbol", "side", "quantity", "price", "amount"])

    raw_price = _numeric(df[price_col]) if price_col is not None else pd.Series(0.0, index=df.index)
    raw_qty = _numeric(df[qty_col]) if qty_col is not None else pd.Series(0.0, index=df.index)
    raw_amount = (
        _numeric(df[amount_col])
        if amount_col is not None
        else (raw_price * raw_qty)
    )

    out = pd.DataFrame(index=df.index)
    out["date"] = df[date_col].apply(_parse_date_value)
    out["side"] = df[side_col].apply(_map_side)
    out["symbol"] = (
        df[symbol_col].apply(normalize_symbol)
        if symbol_col is not None
        else ""
    )
    out["quantity"] = raw_qty.abs()
    out["price"] = raw_price
    out["amount"] = raw_amount.abs()

    cash_mask = out["side"].isin(["cash_in", "cash_out"])
    out.loc[cash_mask & (out["price"] <= 0), "price"] = out.loc[cash_mask, "amount"]
    out.loc[cash_mask & (out["quantity"] <= 0), "quantity"] = 1.0

    out = out[out["date"].notna() & out["side"].notna()].copy()
    position_mask = out["side"].isin(["buy", "sell", "allotment_in", "allotment_out"])
    out = out[
        (~position_mask)
        | ((out["symbol"] != "") & (out["quantity"] > 0))
    ].copy()
    out = out[
        (~out["side"].isin(["buy", "sell"]))
        | (out["price"] > 0)
    ].copy()

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out.attrs["source"] = source
    return out


def _infer_initial_positions(trades: pd.DataFrame) -> tuple[dict[str, float], set[str], list[str]]:
    initial_positions: dict[str, float] = {}
    position_check: dict[str, float] = {}
    symbol_side_types: dict[str, set[str]] = {}
    warnings: list[str] = []

    for row in trades.itertuples(index=False):
        sym = str(row.symbol)
        side = str(row.side)
        qty = float(row.quantity)
        if not sym:
            continue
        symbol_side_types.setdefault(sym, set()).add(side)

        if side == "buy":
            position_check[sym] = position_check.get(sym, 0.0) + qty
        elif side == "sell":
            position_check[sym] = position_check.get(sym, 0.0) - qty
            if position_check[sym] < 0:
                shortfall = -position_check[sym]
                initial_positions[sym] = initial_positions.get(sym, 0.0) + shortfall
                position_check[sym] = 0.0
        elif side == "allotment_in":
            position_check[sym] = position_check.get(sym, 0.0) + qty
        elif side == "allotment_out":
            position_check[sym] = position_check.get(sym, 0.0) - qty

    ipo_only: set[str] = set()
    for sym, types in symbol_side_types.items():
        if types and types <= {"allotment_in", "allotment_out"} and abs(position_check.get(sym, 0.0)) < 0.01:
            ipo_only.add(sym)

    if initial_positions:
        preview = ", ".join(f"{sym}={qty:.0f}股" for sym, qty in sorted(initial_positions.items())[:8])
        warnings.append(f"检测到首个交易日前持仓，已推断期初持仓：{preview}")
    if ipo_only:
        warnings.append(f"检测到 {len(ipo_only)} 个仅配售/下账代码，已从行情需求中排除。")

    return initial_positions, ipo_only, warnings


def compute_portfolio_trend(
    trades_df: pd.DataFrame,
    initial_capital: float,
    label: str,
) -> dict[str, Any]:
    """从标准化交易和日度行情计算组合累计收益曲线。"""
    trades = standardize_trades(trades_df, label)
    warnings: list[str] = []

    if trades.empty:
        return {
            "label": label,
            "trend": [],
            "meta": {
                "initialCapital": initial_capital,
                "initialAsset": initial_capital,
                "startDate": None,
                "endDate": None,
                "finalReturn": None,
                "dataQuality": "insufficient_trade_data",
                "missingSymbols": [],
                "fallbackSymbols": [],
                "coverageRate": 0.0,
                "warnings": ["没有可用于计算收益曲线的有效交易记录。"],
            },
        }

    initial_positions, ipo_only, inferred_warnings = _infer_initial_positions(trades)
    warnings.extend(inferred_warnings)

    trade_start = trades["date"].iloc[0].date().isoformat()
    trade_end = trades["date"].iloc[-1].date().isoformat()

    position_symbols = sorted({
        str(row.symbol)
        for row in trades.itertuples(index=False)
        if row.side in {"buy", "sell", "allotment_in", "allotment_out"}
        and str(row.symbol)
        and str(row.symbol) not in ipo_only
    } | set(initial_positions.keys()))

    price_result = get_historical_prices(position_symbols, trade_start, trade_end)
    price_map = price_result["priceMap"]
    missing_symbols = price_result["missingSymbols"]
    warnings.extend(price_result["warnings"])

    total_needed = len(position_symbols)
    missing_ratio = len(missing_symbols) / total_needed if total_needed else 0.0
    coverage_rate = ((total_needed - len(missing_symbols)) / total_needed * 100) if total_needed else 100.0

    fallback_prices: dict[str, float] = {}
    fallback_symbols: list[str] = []
    if missing_symbols:
        missing_set = set(missing_symbols)
        for row in trades.itertuples(index=False):
            sym = str(row.symbol)
            if sym in missing_set and sym not in fallback_prices and float(row.price) > 0:
                fallback_prices[sym] = float(row.price)
                fallback_symbols.append(sym)

    price_lookup: dict[str, dict[str, float]] = {}
    all_price_dates: set[str] = set()
    for sym, rows in price_map.items():
        lookup = {row["date"]: float(row["close"]) for row in rows}
        price_lookup[sym] = lookup
        all_price_dates.update(lookup.keys())

    if not all_price_dates and not fallback_prices:
        return {
            "label": label,
            "trend": [],
            "meta": {
                "initialCapital": initial_capital,
                "initialAsset": initial_capital,
                "startDate": trade_start,
                "endDate": trade_end,
                "finalReturn": None,
                "dataQuality": "insufficient_price_data",
                "missingSymbols": missing_symbols,
                "fallbackSymbols": fallback_symbols,
                "coverageRate": round(coverage_rate, 1),
                "warnings": ["没有可用行情数据。"] + warnings,
            },
        }

    initial_holding_value = 0.0
    for sym, qty in initial_positions.items():
        if sym in fallback_prices:
            initial_holding_value += qty * fallback_prices[sym]
        elif sym in price_lookup and price_lookup[sym]:
            first_date = min(price_lookup[sym].keys())
            initial_holding_value += qty * price_lookup[sym][first_date]

    initial_cash = float(initial_capital or DEFAULT_INITIAL_CAPITAL)
    initial_asset = initial_cash + initial_holding_value
    if initial_holding_value > 0:
        warnings.append(
            f"期初持仓估值 {initial_holding_value:,.0f}，初始总资产 {initial_asset:,.0f}。"
        )

    if fallback_symbols:
        warnings.append(
            f"缺失行情代码 {len(fallback_symbols)} 个，已使用首次成交价固定估值，曲线为近似结果："
            f"{', '.join(fallback_symbols[:8])}{'...' if len(fallback_symbols) > 8 else ''}"
        )

    trades_by_date: dict[str, list[Any]] = {}
    for row in trades.itertuples(index=False):
        day = row.date.date().isoformat()
        trades_by_date.setdefault(day, []).append(row)

    cash = initial_cash
    positions: dict[str, float] = dict(initial_positions)
    last_prices: dict[str, float] = {}

    for sym in initial_positions:
        if sym in price_lookup and price_lookup[sym]:
            first_date = min(price_lookup[sym].keys())
            last_prices[sym] = price_lookup[sym][first_date]

    start_dt = date.fromisoformat(trade_start)
    end_dt = date.fromisoformat(max(all_price_dates)) if all_price_dates else date.fromisoformat(trade_end)
    trend: list[dict[str, Any]] = []
    current = start_dt

    while current <= end_dt:
        day_str = current.isoformat()

        for row in trades_by_date.get(day_str, []):
            sym = str(row.symbol)
            side = str(row.side)
            qty = float(row.quantity)
            price = float(row.price)

            if side == "buy":
                cash -= price * qty
                positions[sym] = positions.get(sym, 0.0) + qty
            elif side == "sell":
                cash += price * qty
                positions[sym] = positions.get(sym, 0.0) - qty
            elif side == "allotment_in":
                positions[sym] = positions.get(sym, 0.0) + qty
            elif side == "allotment_out":
                positions[sym] = positions.get(sym, 0.0) - qty
            elif side == "cash_in":
                cash += price
            elif side == "cash_out":
                cash -= price

        total_asset = cash
        for sym, qty in positions.items():
            if abs(qty) < 1e-9:
                continue
            if sym in fallback_prices:
                total_asset += qty * fallback_prices[sym]
                continue
            day_price = price_lookup.get(sym, {}).get(day_str)
            if day_price is not None:
                last_prices[sym] = day_price
                total_asset += qty * day_price
            elif sym in last_prices:
                total_asset += qty * last_prices[sym]

        cumulative_return = (total_asset / initial_asset - 1.0) * 100
        trend.append({
            "date": day_str,
            "totalAsset": round(total_asset, 2),
            "cumulativeReturn": round(cumulative_return, 6),
        })
        current += timedelta(days=1)

    return {
        "label": label,
        "trend": trend,
        "meta": {
            "initialCapital": round(initial_cash, 2),
            "initialAsset": round(initial_asset, 2),
            "startDate": trade_start,
            "endDate": end_dt.isoformat(),
            "finalReturn": trend[-1]["cumulativeReturn"] if trend else None,
            "dataQuality": "partial_missing_prices" if fallback_symbols else "ok",
            "missingSymbols": missing_symbols,
            "fallbackSymbols": fallback_symbols,
            "coverageRate": round(coverage_rate, 1),
            "warnings": warnings,
        },
    }


def compute_strategy_trend(strategy_id: str, trades_df: pd.DataFrame) -> dict[str, Any]:
    return compute_portfolio_trend(
        trades_df=trades_df,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        label=strategy_id,
    )


def compute_user_trend(
    user_id: str,
    trades_df: pd.DataFrame,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    inferred = _extract_initial_capital_from_columns(trades_df)
    capital = initial_capital if initial_capital and initial_capital > 0 else inferred
    return compute_portfolio_trend(
        trades_df=trades_df,
        initial_capital=capital or DEFAULT_INITIAL_CAPITAL,
        label=f"用户 {user_id}",
    )


def build_trend_plot(
    customer_trend: dict[str, Any] | None,
    strategy_trends: dict[str, dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()

    if customer_trend and customer_trend.get("trend"):
        df_customer = pd.DataFrame(customer_trend["trend"])
        fig.add_trace(go.Scatter(
            x=df_customer["date"],
            y=df_customer["cumulativeReturn"],
            mode="lines",
            name="我的账户",
            line={"width": 3, "color": "#2563eb"},
            hovertemplate="%{x}<br>我的账户: %{y:.2f}%<extra></extra>",
        ))

    for strategy_name, result in strategy_trends.items():
        if not result.get("trend"):
            continue
        df_strategy = pd.DataFrame(result["trend"])
        fig.add_trace(go.Scatter(
            x=df_strategy["date"],
            y=df_strategy["cumulativeReturn"],
            mode="lines",
            name=strategy_name,
            line={"width": 2},
            hovertemplate=f"%{{x}}<br>{strategy_name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(
        height=430,
        margin={"l": 8, "r": 8, "t": 16, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis_title="日期",
        yaxis_title="累计收益率（%）",
        template="plotly_white",
    )
    return fig
