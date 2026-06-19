"""
Experiment 1 — Synthetic Trade Simulator

Generates synthetic stock prices (Geometric Brownian Motion) and investor
trading records based on 6 predefined behavior types.  The simulation uses
independent behavioral parameters — NOT feature-space perturbations — so
ground truth labels do not share a causal mechanism with the similarity
metrics under test.

Output:
    experiments/exp1_synthetic/output/synthetic_trades.csv   — all trades
    experiments/exp1_synthetic/output/investor_labels.csv    — type labels
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────
SEED_BASE = 42
N_TRADING_DAYS = 504          # ~2 years
TRADING_DAYS_PER_MONTH = 21
START_DATE = "2024-01-01"

N_STOCKS = 50
N_ETF_STOCKS = 10
# ETF codes must start with 5 to be detectable by _pick_stock / feature extraction
def _make_etf_codes(n: int) -> list[str]:
    return [f"51{i:04d}" for i in range(n)]

# ── investor-type definitions ───────────────────────────────────────────────

@dataclass
class InvestorType:
    type_id: str       # "T1"–"T6"
    label: str         # Chinese label
    buys_per_month: float   # λ — expected new positions per month
    avg_hold_days: float    # mean target holding period
    take_profit: float      # +gain fraction that triggers sell
    stop_loss: float        # –loss fraction that triggers sell (stored positive)
    n_holdings: int         # target concurrent holdings
    momentum: str           # "strong" / "medium" / "weak"

INVESTOR_TYPES = [
    InvestorType("T1", "长线分散",  0.5,  90,  0.30, 0.15, 15, "weak"),
    InvestorType("T2", "长线集中",  0.5,  90,  0.30, 0.15,  3, "weak"),
    InvestorType("T3", "短线高频",  3.0,   5,  0.05, 0.03, 10, "strong"),
    InvestorType("T4", "短线投机",  5.0,   2,  0.03, 0.02,  2, "strong"),
    InvestorType("T5", "中线均衡",  1.0,  30,  0.15, 0.10,  8, "medium"),
    InvestorType("T6", "被动ETF",   0.3, 120,  9.99, 0.20, 15, "weak"),
]
N_PER_TYPE = 50


# ── stock-price simulation (GBM) ────────────────────────────────────────────

def generate_stock_prices(n_days: int = N_TRADING_DAYS,
                          seed: int = SEED_BASE) -> pd.DataFrame:
    """Simulate daily closing prices via correlated Geometric Brownian Motion.

    Returns DataFrame  (n_days × n_stocks)  with 6-digit string column labels.
    """
    rng = np.random.default_rng(seed)
    n_stocks_total = N_STOCKS + N_ETF_STOCKS

    # market factor → moderate cross-stock correlation
    market_annual_ret = 0.08
    market_annual_vol = 0.20
    dt = 1 / 252
    market_r = rng.normal(market_annual_ret * dt,
                          market_annual_vol * np.sqrt(dt),
                          size=n_days)

    # idiosyncratic returns
    idio_annual_ret = rng.uniform(-0.05, 0.20, size=n_stocks_total)
    idio_annual_vol = rng.uniform(0.20, 0.50, size=n_stocks_total)
    idio_r = rng.normal(idio_annual_ret * dt,
                        idio_annual_vol * np.sqrt(dt),
                        size=(n_days, n_stocks_total))

    # blend: ρ ≈ 0.5 between any two stocks
    alpha = 0.5
    log_ret = alpha * market_r[:, None] + np.sqrt(1 - alpha ** 2) * idio_r
    prices = 50 * np.exp(np.cumsum(log_ret, axis=0))

    # build column names
    stock_codes = [f"{600001 + i:06d}" for i in range(N_STOCKS)]
    etf_codes = _make_etf_codes(N_ETF_STOCKS)
    return pd.DataFrame(prices, columns=stock_codes + etf_codes)


# ── trade simulation ────────────────────────────────────────────────────────

def _pick_stock(candidates: list[str],
                prices: pd.DataFrame,
                day_idx: int,
                momentum: str,
                rng: np.random.Generator) -> str:
    """Select a stock filtered by momentum bias."""
    if day_idx < 6:
        return rng.choice(candidates)

    rets = {}
    for s in candidates:
        p_today = prices.iloc[day_idx][s]
        p_5ago = prices.iloc[day_idx - 5][s]
        if p_5ago > 0:
            rets[s] = (p_today - p_5ago) / p_5ago
    if not rets:
        return rng.choice(candidates)

    sorted_stocks = sorted(rets, key=rets.get)
    n = len(sorted_stocks)
    if momentum == "strong":
        pool = sorted_stocks[n // 2 :]
    elif momentum == "weak":
        pool = sorted_stocks[: n // 2]
    else:
        pool = candidates
    return rng.choice(pool if pool else candidates)


def simulate_one_investor(inv_id: str,
                          inv_type: InvestorType,
                          prices: pd.DataFrame,
                          etf_codes: list[str],
                          dates: pd.DatetimeIndex,
                          seed: int) -> pd.DataFrame:
    """Generate BUY/SELL trades for a single investor."""
    rng = np.random.default_rng(seed)
    n_days = len(dates)

    if inv_type.type_id == "T6":
        pool = etf_codes
        lot_base = 100
    else:
        pool = [c for c in prices.columns if c not in etf_codes]
        lot_base = 1

    daily_buy_prob = inv_type.buys_per_month / TRADING_DAYS_PER_MONTH
    # cap so extremely frequent types don't buy multiple times every day
    daily_buy_prob = min(daily_buy_prob, 0.5)

    positions: list[dict] = []   # {symbol, buy_day, buy_price, qty, target_hold}
    rows: list[dict] = []

    for day in range(n_days):
        # ── 1. check sell triggers ──
        to_close: list[int] = []
        for i, pos in enumerate(positions):
            px = prices.iloc[day][pos["symbol"]]
            ret = (px - pos["buy_price"]) / pos["buy_price"] if pos["buy_price"] > 0 else 0
            held = day - pos["buy_day"]

            trigger = False
            if ret >= inv_type.take_profit:
                trigger = True
            elif ret <= -inv_type.stop_loss:
                trigger = True
            elif held >= pos["target_hold"]:
                trigger = True

            if trigger:
                to_close.append(i)
                rows.append(dict(datetime=dates[day], symbol=pos["symbol"],
                                 action="SELL", price=px, quantity=pos["qty"],
                                 amount=px * pos["qty"]))
        positions = [p for i, p in enumerate(positions) if i not in to_close]

        # ── 2. decide to buy ──
        if len(positions) < inv_type.n_holdings * 2 and rng.random() < daily_buy_prob:
            stock = _pick_stock(pool, prices, day, inv_type.momentum, rng)
            px = prices.iloc[day][stock]
            qty = max(lot_base, int(rng.uniform(2000, 30000) / px))
            target_hold = max(2, int(rng.normal(inv_type.avg_hold_days,
                                                inv_type.avg_hold_days * 0.3)))

            positions.append(dict(symbol=stock, buy_day=day, buy_price=px,
                                  qty=qty, target_hold=target_hold))
            rows.append(dict(datetime=dates[day], symbol=stock, action="BUY",
                             price=px, quantity=qty, amount=px * qty))

    # ── 3. force-close remaining at period end ──
    last_day = n_days - 1
    for pos in positions:
        px = prices.iloc[last_day][pos["symbol"]]
        rows.append(dict(datetime=dates[last_day], symbol=pos["symbol"],
                         action="SELL", price=px, quantity=pos["qty"],
                         amount=px * pos["qty"]))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "symbol", "action",
                                     "price", "quantity", "amount"])
    df["investor_id"] = inv_id
    df["type_id"] = inv_type.type_id
    df["type_label"] = inv_type.label
    return df.sort_values("datetime").reset_index(drop=True)


# ── main entry point ────────────────────────────────────────────────────────

def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data and return (trades_df, labels_df)."""
    print("Generating stock prices (GBM) …")
    prices = generate_stock_prices()
    dates = pd.bdate_range(START_DATE, periods=N_TRADING_DAYS)
    etf_codes = [c for c in prices.columns if c.startswith("5")]

    all_trades: list[pd.DataFrame] = []
    labels: list[dict] = []
    seed = SEED_BASE + 1

    for t in INVESTOR_TYPES:
        for i in range(N_PER_TYPE):
            inv_id = f"{t.type_id}_{i + 1:02d}"
            df = simulate_one_investor(inv_id, t, prices, etf_codes, dates, seed)
            seed += 1
            all_trades.append(df)
            labels.append(dict(investor_id=inv_id, type_id=t.type_id, type_label=t.label))

    trades = pd.concat(all_trades, ignore_index=True)
    labels_df = pd.DataFrame(labels)

    trades.to_csv(OUTPUT_DIR / "synthetic_trades.csv", index=False)
    labels_df.to_csv(OUTPUT_DIR / "investor_labels.csv", index=False)

    n = len(trades)
    n_inv = trades["investor_id"].nunique()
    print(f"Simulation done: {n_inv} investors, {n} trades → {OUTPUT_DIR}")
    return trades, labels_df


if __name__ == "__main__":
    run_simulation()
