"""
Phase 2 — Noise Calibration from Real Accounts.

Use 3 real accounts' trading records with Bootstrap resampling to estimate
per-feature perturbation magnitudes. The intuition:

  "If we observe slightly different subsets of the SAME account's trades,
   how much do the extracted features vary?"

This within-account feature variability is used as the calibrated noise level
for user generation (Phase 3). It reflects the natural "same-behaviour"
variation that arises from finite sampling of trading records — not arbitrary.

Output:
    data/calibrated_noise.json    — σ_calibrated[12] + metadata
    output/noise_calibration.png  — per-feature σ across accounts
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.expB1_user_strategy.utils import (
    MATCH_FEATURES, DATA_DIR, OUTPUT_DIR,
    ACCOUNTS_CSV, extract_features, features_to_array, prepare_inplace,
)

N_BOOTSTRAP = 100
BOOTSTRAP_RATIO = 0.80
SEED = 42


def main():
    print("=" * 60)
    print("  Phase 2 — Noise Calibration from Real Accounts")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    accounts_df = prepare_inplace(pd.read_csv(ACCOUNTS_CSV, dtype={"stock_code": str}))
    account_ids = sorted(accounts_df["account_id"].unique())
    print(f"\n  Accounts: {account_ids}")
    for aid in account_ids:
        n = len(accounts_df[accounts_df["account_id"] == aid])
        print(f"    Account {aid}: {n} trades")

    # ── per-account bootstrap ──
    # σ_a[d] = std of feature d across 100 bootstrap subsamples of account a
    all_sigmas: dict[str, np.ndarray] = {}

    for aid in account_ids:
        a_trades = accounts_df[accounts_df["account_id"] == aid].copy()
        n_sample = max(int(len(a_trades) * BOOTSTRAP_RATIO), 20)

        boot_feats = []
        for b in range(N_BOOTSTRAP):
            idx = rng.choice(len(a_trades), size=n_sample, replace=True)
            sample = a_trades.iloc[idx].reset_index(drop=True)
            feat = extract_features(sample)
            boot_feats.append(feat)

        feat_arr = features_to_array(boot_feats)  # (100, 12)
        sigma_a = feat_arr.std(axis=0)             # (12,)
        all_sigmas[aid] = sigma_a

        print(f"\n  Account {aid} (bootstrap n={n_sample}):")
        for d, fname in enumerate(MATCH_FEATURES):
            print(f"    {fname:24s}  σ = {sigma_a[d]:.4f}")

    # ── per-feature median across accounts ──
    sigma_stack = np.stack(list(all_sigmas.values()))  # (3, 12)
    sigma_calibrated = np.median(sigma_stack, axis=0)   # (12,)

    print(f"\n  Calibrated σ (median across accounts):")
    for d, fname in enumerate(MATCH_FEATURES):
        vals = [all_sigmas[a][d] for a in account_ids]
        print(f"    {fname:24s}  σ_cal = {sigma_calibrated[d]:.4f}  "
              f"(per-account: {[f'{v:.4f}' for v in vals]})")

    # ── cross-reference: strategy-level σ as upper bound ──
    strat_feat_df = pd.read_csv(DATA_DIR / "strategy_features.csv", index_col=0)
    strat_std = strat_feat_df[MATCH_FEATURES].std().values
    print(f"\n  Strategy-level σ (upper bound reference, n=37):")
    for d, fname in enumerate(MATCH_FEATURES):
        ratio = sigma_calibrated[d] / strat_std[d] if strat_std[d] > 0 else 0
        print(f"    {fname:24s}  σ_strat = {strat_std[d]:.4f}  "
              f"σ_cal/σ_strat = {ratio:.3f}")

    # ── save ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibrated = {
        "description": (
            "Per-feature calibrated noise σ. Computed as per-account bootstrap "
            f"std (median across {len(account_ids)} accounts), {N_BOOTSTRAP} "
            f"iterations per account, {BOOTSTRAP_RATIO:.0%} subsample ratio."
        ),
        "bootstrap_iterations": N_BOOTSTRAP,
        "bootstrap_ratio": BOOTSTRAP_RATIO,
        "account_ids": list(account_ids),
        "sigma_calibrated": {f: float(v) for f, v in zip(MATCH_FEATURES, sigma_calibrated)},
        "sigma_by_account": {
            a: {f: float(v) for f, v in zip(MATCH_FEATURES, all_sigmas[a])}
            for a in account_ids
        },
        "sigma_strategy_level": {f: float(v) for f, v in zip(MATCH_FEATURES, strat_std)},
        "noise_levels": {
            "low":  {f: float(v * 0.5) for f, v in zip(MATCH_FEATURES, sigma_calibrated)},
            "mid":  {f: float(v * 1.0) for f, v in zip(MATCH_FEATURES, sigma_calibrated)},
            "high": {f: float(v * 2.0) for f, v in zip(MATCH_FEATURES, sigma_calibrated)},
        },
    }
    with open(DATA_DIR / "calibrated_noise.json", "w", encoding="utf-8") as f:
        json.dump(calibrated, f, ensure_ascii=False, indent=2)

    # ── visualisation ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(MATCH_FEATURES))
    width = 0.25
    for i, aid in enumerate(account_ids):
        ax.bar(x + i * width, all_sigmas[aid], width, label=f"Account {aid}", alpha=0.7)
    ax.plot(x + width, sigma_calibrated, "D-", color="black", markersize=8,
            linewidth=2, label="σ_calibrated (median)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(MATCH_FEATURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Per-feature std (bootstrap)")
    ax.set_title(f"Noise Calibration: Per-Feature Bootstrap σ ({N_BOOTSTRAP} iterations)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved: {DATA_DIR / 'calibrated_noise.json'}")
    print(f"  Saved: {OUTPUT_DIR / 'noise_calibration.png'}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
