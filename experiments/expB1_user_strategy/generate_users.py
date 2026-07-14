"""
Phase 3 — Compatible User Generation.

For each of the 37 strategies, generate 20 "compatible users" by applying
5-level perturbation at the TRADING-RECORD level (NOT the feature-vector level).

This preserves the causal chain:  trading records → feature extraction (with noise)
so that the perturbation source and the metric-under-test reside in separate spaces,
avoiding circular reasoning.

Output:
    data/user_features.csv            — 740 × 12 feature matrix
    data/user_labels.csv              — user_id → source_strategy mapping
    data/synthetic_users/*.csv        — per-user perturbed trading records
    output/user_generation_stats.png  — quality check visualisation
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.expB1_user_strategy.utils import (
    MATCH_FEATURES, DATA_DIR, OUTPUT_DIR,
    STRATEGIES_CSV, extract_features, prepare_inplace,
)

N_USERS_PER_STRATEGY = 20
SEED = 42


def perturb_trades(trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply 5 perturbation operations at the trading-record level.

    Operations (applied in order):
      1. Bootstrap resample: 85-95% of original trades (with replacement)
      2. Amount jitter:     multiply by exp(N(0, 0.15^2))
      3. Date jitter:       offset trade_date by N(0, 2^2) business days
      4. Random drop:       remove 3-8% of trades
      5. Noise insertion:   insert 5-10% extra random-ish trades
    """
    t = trades.copy()
    n_orig = len(t)

    # ── 1. Bootstrap resample ──
    ratio = rng.uniform(0.85, 0.95)
    n_sample = max(int(n_orig * ratio), 15)
    idx = rng.choice(n_orig, size=n_sample, replace=True)
    t = t.iloc[idx].reset_index(drop=True)

    # ── 2. Amount jitter ──
    t["amount"] = t["amount"].abs() * np.exp(rng.normal(0.0, 0.15, size=len(t)))

    # ── 3. Date jitter (Normal(0, 2²) per plan spec) ──
    offsets = np.rint(rng.normal(0, 2, size=len(t))).astype(int)
    t["trade_date"] = t["trade_date"] + pd.to_timedelta(offsets, unit="D")
    t = t.sort_values("trade_date").reset_index(drop=True)

    # ── 4. Random drop ──
    drop_ratio = rng.uniform(0.03, 0.08)
    n_keep = max(int(len(t) * (1.0 - drop_ratio)), 10)
    keep_idx = rng.choice(len(t), size=n_keep, replace=False)
    t = t.iloc[np.sort(keep_idx)].reset_index(drop=True)

    # ── 5. Noise insertion ──
    n_noise = max(int(len(t) * rng.uniform(0.05, 0.10)), 1)
    # pick random templates from existing trades
    noise_rows = []
    for _ in range(n_noise):
        template = t.iloc[rng.integers(0, len(t))]
        row = {
            "trade_date": template["trade_date"] + pd.Timedelta(days=int(rng.integers(-5, 6))),
            "symbol": template["symbol"],
            "action": rng.choice(["BUY", "SELL"]),
            "price": template["price"] * np.exp(rng.normal(0.0, 0.05)),
            "amount": abs(template["amount"]) * np.exp(rng.normal(0.0, 0.30)),
            "quantity": 0.0,
            "is_buy": False,
            "is_sell": False,
        }
        row["quantity"] = row["amount"] / max(row["price"], 1e-8)
        noise_rows.append(row)

    noise_df = pd.DataFrame(noise_rows)
    noise_df["is_buy"] = noise_df["action"] == "BUY"
    noise_df["is_sell"] = noise_df["action"] == "SELL"

    t = pd.concat([t, noise_df], ignore_index=True)
    t = t.sort_values("trade_date").reset_index(drop=True)

    return t


def main():
    print("=" * 60)
    print("  Phase 3 — Compatible User Generation")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users_dir = DATA_DIR / "synthetic_users"
    users_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load strategies ──
    raw = pd.read_csv(STRATEGIES_CSV, dtype={"stock_code": str, "action": str})
    raw["action"] = raw["action"].astype(str).str.strip().str.upper()
    strategy_names = sorted(raw["strategy_name"].unique())
    print(f"\n  Strategies: {len(strategy_names)}")
    print(f"  Users per strategy: {N_USERS_PER_STRATEGY}")
    print(f"  Total users to generate: {len(strategy_names) * N_USERS_PER_STRATEGY}")

    # ── generate ──
    all_features: list[dict] = []
    all_labels: list[dict] = []

    for s_idx, sname in enumerate(strategy_names):
        s_trades_raw = raw[raw["strategy_name"] == sname].copy()
        s_trades = prepare_inplace(s_trades_raw)
        n_s_trades = len(s_trades)

        for u_i in range(N_USERS_PER_STRATEGY):
            user_id = f"U_{s_idx:02d}_{u_i:02d}"
            seed_u = SEED * 10000 + s_idx * 100 + u_i
            local_rng = np.random.default_rng(seed_u)

            # apply perturbations
            perturbed = perturb_trades(s_trades, local_rng)

            # extract features
            feat = extract_features(perturbed)
            feat["user_id"] = user_id
            all_features.append(feat)

            all_labels.append({
                "user_id": user_id,
                "source_strategy": sname,
                "source_strategy_idx": s_idx,
                "n_trades_original": n_s_trades,
                "n_trades_perturbed": len(perturbed),
            })

            # save perturbed trades (only first 2 per strategy to save disk)
            if u_i < 2:
                perturbed_out = perturbed.copy()
                perturbed_out["datetime"] = perturbed_out["trade_date"].dt.strftime(
                    "%Y-%m-%d %H:%M:%S")
                out_cols = ["datetime", "symbol", "action", "price", "amount", "quantity"]
                perturbed_out[out_cols].to_csv(
                    users_dir / f"{user_id}_trades.csv", index=False, encoding="utf-8-sig")

        if (s_idx + 1) % 10 == 0:
            print(f"  [{s_idx + 1}/{len(strategy_names)}] strategies done")

    print(f"  Generated {len(all_features)} users")

    # ── build feature matrix ──
    feat_df = pd.DataFrame(all_features)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    user_ids = feat_df.pop("user_id")
    feat_cols = [c for c in MATCH_FEATURES if c in feat_df.columns]
    feat_matrix = feat_df[feat_cols]

    labels_df = pd.DataFrame(all_labels)

    # ── quality check: user mean vs strategy vector ──
    print(f"\n  Quality check: ‖μ_users - s_k‖ per strategy")
    strat_feats = pd.read_csv(DATA_DIR / "strategy_features.csv", index_col=0)
    strat_arr = strat_feats[MATCH_FEATURES].values.astype(float)
    strat_l2_std = float(np.linalg.norm(strat_arr.std(axis=0)))
    print(f"  Strategy-level L2 std: {strat_l2_std:.2f}  (threshold: 2× = {2 * strat_l2_std:.2f})")
    distances = []
    for s_idx, sname in enumerate(strategy_names):
        s_vec = strat_feats.loc[sname, MATCH_FEATURES].values.astype(float)
        user_mask = labels_df["source_strategy"] == sname
        user_vecs = feat_matrix.loc[user_mask, MATCH_FEATURES].values
        u_mean = user_vecs.mean(axis=0)
        dist = np.linalg.norm(u_mean - s_vec)
        distances.append({"strategy": sname, "distance": dist, "n_trades": len(user_vecs)})
        if dist > 2 * strat_l2_std:  # flag large deviations
            print(f"    [!] {sname}: dist = {dist:.2f}  (large — may need review)")

    dist_vals = [d["distance"] for d in distances]
    print(f"    Mean dist: {np.mean(dist_vals):.2f}, "
          f"Median: {np.median(dist_vals):.2f}, "
          f"Max: {np.max(dist_vals):.2f}")

    # ── save ──
    feat_df_out = pd.concat([pd.Series(user_ids, name="user_id"), feat_matrix], axis=1)
    feat_df_out.to_csv(DATA_DIR / "user_features.csv", index=False, encoding="utf-8-sig")
    labels_df.to_csv(DATA_DIR / "user_labels.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Saved: {DATA_DIR / 'user_features.csv'}  ({feat_matrix.shape})")
    print(f"  Saved: {DATA_DIR / 'user_labels.csv'}  ({len(labels_df)} rows)")
    print(f"  Saved: {users_dir}/  (sample perturbed trades, 2 per strategy)")

    # ── quick stats ──
    print(f"\n  Perturbed trade count: "
          f"min={labels_df['n_trades_perturbed'].min()}, "
          f"median={labels_df['n_trades_perturbed'].median():.0f}, "
          f"max={labels_df['n_trades_perturbed'].max()}")

    # ── visualisation: feature distribution comparison ──
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for d, fname in enumerate(MATCH_FEATURES):
        ax = axes[d]
        s_vals = strat_feats[fname].values.astype(float)
        u_vals = feat_matrix[fname].values
        ax.hist(s_vals, bins=20, alpha=0.5, label="Strategies (n=37)", density=True)
        ax.hist(u_vals, bins=40, alpha=0.5, label=f"Users (n={len(feat_matrix)})", density=True)
        ax.set_title(fname, fontsize=8)
        ax.tick_params(labelsize=7)
        if d >= 8:
            ax.legend(fontsize=6)
    fig.suptitle("Feature Distribution: Strategies vs Generated Users",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "user_generation_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'user_generation_stats.png'}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
