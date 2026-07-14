"""
Phase 1 — Strategy Behaviour Profile Extraction.

Extract 12-dim feature vectors from 37 strategies' cleaned trading records
using the SAME feature-extraction pipeline as the main project (pipeline.py).

Output:
    data/strategy_features.csv     — 37 × 12 feature matrix
    data/strategy_meta.csv         — strategy names, trade counts, date spans
"""
import numpy as np
import pandas as pd

from experiments.expB1_user_strategy.utils import (
    MATCH_FEATURES, DATA_DIR,
    STRATEGIES_CSV, extract_features, prepare_inplace,
)


def main():
    print("=" * 60)
    print("  Phase 1 — Strategy Behaviour Profile Extraction")
    print("=" * 60)

    # ── load all strategies ──
    raw = pd.read_csv(STRATEGIES_CSV, dtype={"stock_code": str, "action": str})
    raw["action"] = raw["action"].astype(str).str.strip().str.upper()
    strategy_names = sorted(raw["strategy_name"].unique())
    print(f"\n  Strategies: {len(strategy_names)}")

    # ── extract per-strategy features ──
    features_list: list[dict] = []
    meta_rows: list[dict] = []

    for sname in strategy_names:
        s_trades_raw = raw[raw["strategy_name"] == sname].copy()
        s_trades = prepare_inplace(s_trades_raw)
        feat = extract_features(s_trades)
        feat["strategy_name"] = sname
        features_list.append(feat)

        meta_rows.append({
            "strategy_name": sname,
            "n_trades": len(s_trades),
            "date_start": s_trades["trade_date"].min().strftime("%Y-%m-%d"),
            "date_end": s_trades["trade_date"].max().strftime("%Y-%m-%d"),
        })

    # ── build feature matrix ──
    fnames = ["strategy_name"] + MATCH_FEATURES
    feat_df = pd.DataFrame(features_list)[fnames]
    feat_df = feat_df.set_index("strategy_name")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta_df = pd.DataFrame(meta_rows)

    # ── check data quality ──
    print(f"\n  Feature matrix: {feat_df.shape}")
    print(f"  Features with NaN: {feat_df.isna().sum().sum()}")
    print(f"  Features with Inf:  {np.isinf(feat_df.values).sum()}")

    # basic stats
    print(f"\n  Per-feature stats:")
    for f in MATCH_FEATURES:
        vals = feat_df[f].values
        print(f"    {f:24s}  mean={vals.mean():10.4f}  std={vals.std():10.4f}  "
              f"min={vals.min():10.4f}  max={vals.max():10.4f}")

    # ── save ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(DATA_DIR / "strategy_features.csv", encoding="utf-8-sig")
    meta_df.to_csv(DATA_DIR / "strategy_meta.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Saved: {DATA_DIR / 'strategy_features.csv'}")
    print(f"  Saved: {DATA_DIR / 'strategy_meta.csv'}")

    # trade count distribution
    print(f"\n  Trade count distribution:")
    print(f"    min={meta_df['n_trades'].min()}, median={meta_df['n_trades'].median():.0f}, "
          f"max={meta_df['n_trades'].max()}")
    tiny = meta_df[meta_df["n_trades"] < 100]
    if len(tiny) > 0:
        print(f"    Strategies with <100 trades ({len(tiny)}):")
        for _, row in tiny.iterrows():
            print(f"      {row['strategy_name']}: {row['n_trades']} trades")

    # ── strategy-level L2 std (for Phase 3 quality check threshold) ──
    strat_arr = feat_df[MATCH_FEATURES].values
    strat_l2_std = float(np.linalg.norm(strat_arr.std(axis=0)))
    print(f"\n  Strategy-level L2 std: {strat_l2_std:.2f}  "
          f"(quality-check threshold for Phase 3: 2× = {2 * strat_l2_std:.2f})")

    print("\n  Done.")


if __name__ == "__main__":
    main()
