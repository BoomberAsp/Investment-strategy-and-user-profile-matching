"""
Experiment 1.4 — Hierarchical Bayesian Radial Penalty (Empirical Bayes)

Derivation:  bayesian_radial_theory.md
Plan:        experiment_plan.md  §1.4

1. Load synthetic trades & extract 12-dim PCA features
2. Train/test split  (30 / 20 per type)
3. Estimate per-type Gamma posterior for λ_k  (closed-form conjugate update)
4. Bayesian predictive penalty:  (b_k / (b_k + d))^a_k
5. k-NN classification — hard (known type) and soft (inferred type)
6. Compare vs fixed-λ baselines
7. Figures: λ posteriors, penalty-curve comparison, accuracy bar chart
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gamma as gamma_dist, kruskal

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (                                         # noqa: E402
    MATCH_FEATURES,
    extract_user_behavior_features,
    extract_user_asset_pref_features,
    extract_user_risk_proxy_features,
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ── gamma hyper-priors ───────────────────────────────────────────────────────
A0, B0 = 1.0, 1.0          # E[λ] = 1.0, Var = 1.0  (weak prior aligned with default λ)
FIXED_LAMBDAS = [0.25, 0.5, 1.0]

TYPE_IDS = ["T1", "T2", "T3", "T4", "T5", "T6"]
TYPE_LABELS = {"T1": "长线分散", "T2": "长线集中", "T3": "短线高频",
               "T4": "短线投机", "T5": "中线均衡", "T6": "被动ETF"}


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  (shared with run.py)
# ══════════════════════════════════════════════════════════════════════════════

def simulated_to_pipeline_format(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    df["trade_date"] = pd.to_datetime(df["datetime"])
    df["is_buy"] = df["action"] == "BUY"
    df["is_sell"] = df["action"] == "SELL"
    return df.sort_values("trade_date").reset_index(drop=True)


def extract_investor_features(investor_trades: pd.DataFrame) -> dict:
    if investor_trades.empty:
        return {f: 0.0 for f in MATCH_FEATURES}
    t = investor_trades.copy()
    return {**extract_user_behavior_features(t),
            **extract_user_asset_pref_features(t),
            **extract_user_risk_proxy_features(t)}


def build_feature_matrix(trades_df: pd.DataFrame,
                         labels_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Return  (X, y, investor_ids)."""
    rows, ids, y_arr = [], [], []
    for inv_id in sorted(trades_df["investor_id"].unique()):
        inv_trades = simulated_to_pipeline_format(
            trades_df[trades_df["investor_id"] == inv_id])
        feat = extract_investor_features(inv_trades)
        rows.append([feat.get(f, 0.0) for f in MATCH_FEATURES])
        ids.append(inv_id)
        lb = labels_df[labels_df["investor_id"] == inv_id]
        y_arr.append(lb["type_id"].values[0] if not lb.empty else "?")
    return (np.nan_to_num(np.array(rows), nan=0.0),
            np.array(y_arr), ids)


# ══════════════════════════════════════════════════════════════════════════════
#  Gamma posterior estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_posteriors(X: np.ndarray, y: np.ndarray,
                        a0: float = A0, b0: float = B0) -> dict:
    """Return  {type_id: {"a": a_k, "b": b_k, "mean": mean, "std": std, ...}}."""
    posteriors = {}
    for tid in TYPE_IDS:
        mask = y == tid
        X_k = X[mask]
        n = len(X_k)
        norms = np.linalg.norm(X_k, axis=1)

        # all positive-pair radial distances within this type
        d_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                if norms[i] > 0 and norms[j] > 0:
                    d_vals.append(abs(np.log(norms[i] / norms[j])))
        d_vals = np.array(d_vals)

        a_k = a0 + len(d_vals)
        b_k = b0 + d_vals.sum()
        post_mean = a_k / b_k
        post_std = np.sqrt(a_k) / b_k

        posteriors[tid] = {
            "a": a_k, "b": b_k,
            "mean": post_mean, "std": post_std,
            "n_investors": n,
            "n_pairs": len(d_vals),
            "d_mean": float(d_vals.mean()) if len(d_vals) > 0 else 0,
            "d_std": float(d_vals.std()) if len(d_vals) > 0 else 0,
        }
    return posteriors


# ══════════════════════════════════════════════════════════════════════════════
#  Bayesian similarity
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_radial_cosine_hard(x_query: np.ndarray, X_train: np.ndarray,
                                y_train: np.ndarray, posteriors: dict,
                                query_type: str) -> np.ndarray:
    """Hard classification: use the query's own type's posterior."""
    a_k = posteriors[query_type]["a"]
    b_k = posteriors[query_type]["b"]

    cos = np.dot(X_train, x_query)   # unnormalised dot
    n_q = np.linalg.norm(x_query)
    n_t = np.linalg.norm(X_train, axis=1)
    denom = n_t * n_q
    denom = np.clip(denom, 1e-12, None)
    cosine = cos / denom

    dr = np.abs(np.log(np.clip(n_t, 1e-12, None) / np.clip(n_q, 1e-12, None)))
    penalty = (b_k / (b_k + dr)) ** a_k
    return cosine * penalty


def bayesian_radial_cosine_soft(x_query: np.ndarray, X_train: np.ndarray,
                                posteriors: dict, centers: dict,
                                temperature: float = 1.0) -> np.ndarray:
    """Soft classification: infer type membership from distance to type centres,
    then mix per-type penalties weighted by membership probability."""
    cos = np.dot(X_train, x_query)
    n_q = np.linalg.norm(x_query)
    n_t = np.linalg.norm(X_train, axis=1)
    denom = np.clip(n_t * n_q, 1e-12, None)
    cosine = cos / denom
    dr = np.abs(np.log(np.clip(n_t, 1e-12, None) / np.clip(n_q, 1e-12, None)))

    # membership probabilities  π_ik ∝ exp(-dist to centre_k / temperature)
    dists = {}
    for tid in TYPE_IDS:
        dists[tid] = np.linalg.norm(x_query - centers[tid])
    inv_temp = -1.0 / max(temperature, 1e-8)
    weights = np.array([np.exp(inv_temp * dists[t]) for t in TYPE_IDS])
    pi = weights / weights.sum()

    # blend penalties
    penalty = np.zeros_like(dr)
    for idx, tid in enumerate(TYPE_IDS):
        a_k = posteriors[tid]["a"]
        b_k = posteriors[tid]["b"]
        penalty += pi[idx] * (b_k / (b_k + dr)) ** a_k

    return cosine * penalty


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_knn_from_sim(sim_matrix: np.ndarray,
                          y_train: np.ndarray,
                          y_test: np.ndarray, k: int = 3) -> float:
    """k-NN accuracy: each row of sim_matrix is a test sample;
    columns index training samples."""
    correct = 0
    uniq = sorted(set(y_train))
    for i in range(len(y_test)):
        order = np.argsort(-sim_matrix[i])
        neighbours = order[:k]
        votes = y_train[neighbours]
        pred = uniq[np.bincount([uniq.index(v) for v in votes]).argmax()]
        correct += int(pred == y_test[i])
    return correct / len(y_test)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_posteriors(posteriors: dict, save_path: Path):
    """Gamma posterior density curves for each type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.linspace(0, 8, 300)
    colours = plt.cm.tab10.colors
    for idx, tid in enumerate(TYPE_IDS):
        p = posteriors[tid]
        ys = gamma_dist.pdf(xs, a=p["a"], scale=1 / p["b"])
        ax.plot(xs, ys, color=colours[idx], linewidth=2,
                label=f'{tid} {TYPE_LABELS[tid]}  (E[λ]={p["mean"]:.2f})')
    ax.axvline(1.0, color="grey", linestyle="--", alpha=0.5, label="prior mean = 1.0")
    ax.set_xlabel("λ")
    ax.set_ylabel("Posterior density")
    ax.set_title("Per-Type Gamma Posteriors for λ_k  (a₀=1, b₀=1)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_penalty_comparison(posteriors: dict, save_path: Path):
    """Bayesian penalty curves vs fixed-λ exp(-λ d)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    ds = np.linspace(0, 3, 200)
    colours = plt.cm.tab10.colors
    for idx, tid in enumerate(TYPE_IDS):
        ax = axes[idx // 3][idx % 3]
        p = posteriors[tid]
        bayes_pen = (p["b"] / (p["b"] + ds)) ** p["a"]
        ax.plot(ds, bayes_pen, color=colours[idx], linewidth=2.5,
                label=f'Bayes  λ~Γ(a={p["a"]:.0f},b={p["b"]:.1f})')
        for lam in FIXED_LAMBDAS:
            ax.plot(ds, np.exp(-lam * ds), linestyle="--", linewidth=1,
                    alpha=0.5, label=f"λ={lam}")
        ax.set_title(f"{tid}  {TYPE_LABELS[tid]}", fontsize=9)
        if idx >= 3:
            ax.set_xlabel("Radial distance  d = |log(‖u‖/‖s‖)|")
        if idx % 3 == 0:
            ax.set_ylabel("Penalty")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Bayesian Penalty vs Fixed-λ Penalty by Investor Type",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(acc_fixed: dict, acc_hard: float, acc_soft: float,
                             save_path: Path):
    """Grouped bar chart: fixed-λ vs Bayesian (hard) vs Bayesian (soft)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"λ={lam}" for lam in FIXED_LAMBDAS] + ["Bayes (hard)", "Bayes (soft)"]
    values = [acc_fixed[f"λ={lam}"] for lam in FIXED_LAMBDAS] + [acc_hard, acc_soft]
    colours = ["#DD8452"] * len(FIXED_LAMBDAS) + ["#55A868", "#4C72B0"]
    bars = ax.bar(labels, values, color=colours)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11)
    ax.set_ylabel("k-NN Accuracy (k=3)")
    ax.set_title("Fixed λ vs Hierarchical Bayesian λ — k=3 Classification")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Experiment 1.4 - Hierarchical Bayesian Radial Penalty")
    print("=" * 64)

    # ── load ──
    trades_path = OUTPUT_DIR / "synthetic_trades.csv"
    labels_path = OUTPUT_DIR / "investor_labels.csv"
    trades_df = pd.read_csv(trades_path)
    labels_df = pd.read_csv(labels_path)
    X_all, y_all, ids_all = build_feature_matrix(trades_df, labels_df)

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=0.90)
    pca.fit(X_scaled)
    scaler_ctr = StandardScaler(with_std=False)
    X_pca = scaler_ctr.fit_transform(X_all) @ pca.components_.T

    # ── train / test split  (stratified per type) ──
    n_train = 30
    train_idx: list[int] = []
    test_idx: list[int] = []
    for tid in TYPE_IDS:
        idx_t = np.where(y_all == tid)[0]
        perm = rng.permutation(len(idx_t))
        train_idx.extend(idx_t[perm[:n_train]].tolist())
        test_idx.extend(idx_t[perm[n_train:]].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train, y_train = X_pca[train_idx], y_all[train_idx]
    X_test, y_test = X_pca[test_idx], y_all[test_idx]
    print(f"Train: {len(X_train)}  ({n_train}/type),  "
          f"Test: {len(X_test)}  ({50 - n_train}/type)")

    # ── estimate posteriors from training set ──
    print("\n[1] Estimating per-type Gamma posteriors …")
    posteriors = estimate_posteriors(X_train, y_train)
    for tid in TYPE_IDS:
        p = posteriors[tid]
        print(f"  {tid} {TYPE_LABELS[tid]:8s}  "
              f"a={p['a']:.0f}  b={p['b']:.2f}  "
              f"E[λ]={p['mean']:.4f}  σ={p['std']:.4f}  "
              f"n_pairs={p['n_pairs']}  d_mean={p['d_mean']:.4f}")

    # ── type centres (for soft classification) ──
    centres = {}
    for tid in TYPE_IDS:
        centres[tid] = X_train[y_train == tid].mean(axis=0)

    # ── fixed-λ baselines ──
    print("\n[2] Fixed-λ baselines …")
    acc_fixed = {}
    for lam in FIXED_LAMBDAS:
        sim = np.zeros((len(X_test), len(X_train)))
        for i, xq in enumerate(X_test):
            cos = np.dot(X_train, xq)
            n_q = np.linalg.norm(xq)
            n_t = np.linalg.norm(X_train, axis=1)
            denom = np.clip(n_t * n_q, 1e-12, None)
            dr = np.abs(np.log(np.clip(n_t, 1e-12, None) / np.clip(n_q, 1e-12, None)))
            sim[i] = (cos / denom) * np.exp(-lam * dr)
        acc = evaluate_knn_from_sim(sim, y_train, y_test, k=3)
        acc_fixed[f"λ={lam}"] = acc
        print(f"  λ={lam:.2f}  k=3 accuracy = {acc:.4f}")

    # ── Bayesian hard ──
    print("\n[3] Bayesian (hard) — known type …")
    sim_hard = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        sim_hard[i] = bayesian_radial_cosine_hard(
            X_test[i], X_train, y_train, posteriors, y_test[i])
    acc_hard = evaluate_knn_from_sim(sim_hard, y_train, y_test, k=3)
    print(f"  k=3 accuracy = {acc_hard:.4f}")

    # ── Bayesian soft ──
    print("\n[4] Bayesian (soft) — inferred type …")
    sim_soft = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        sim_soft[i] = bayesian_radial_cosine_soft(
            X_test[i], X_train, posteriors, centres)
    acc_soft = evaluate_knn_from_sim(sim_soft, y_train, y_test, k=3)
    print(f"  k=3 accuracy = {acc_soft:.4f}")

    # ── Kruskal-Wallis test ──
    print("\n[5] Kruskal-Wallis test on λ_k posteriors …")
    samples = []
    for tid in TYPE_IDS:
        p = posteriors[tid]
        s = gamma_dist.rvs(a=p["a"], scale=1 / p["b"], size=2000, random_state=42)
        samples.append(s)
    h_stat, p_val = kruskal(*samples)
    print(f"  H = {h_stat:.2f},  p = {p_val:.6f}  "
          f"{'*** p<0.001' if p_val < 0.001 else ''}")

    # ── figures ──
    print("\n[6] Generating figures …")
    plot_posteriors(posteriors, OUTPUT_DIR / "bayesian_lambda_posteriors.png")
    plot_penalty_comparison(posteriors, OUTPUT_DIR / "bayesian_penalty_comparison.png")
    plot_accuracy_comparison(acc_fixed, acc_hard, acc_soft,
                             OUTPUT_DIR / "bayesian_accuracy_comparison.png")

    # ── serialise ──
    report = {
        "a0": A0, "b0": B0,
        "posteriors": posteriors,
        "accuracy": {"fixed": acc_fixed, "bayesian_hard": acc_hard,
                     "bayesian_soft": acc_soft},
        "kruskal_wallis": {"H": float(h_stat), "p": float(p_val)},
    }
    with open(OUTPUT_DIR / "bayesian_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    # ── verdict ──
    print("\n" + "=" * 64)
    best_fixed = max(acc_fixed.values())
    print(f"  Best fixed λ:       {best_fixed:.4f}")
    print(f"  Bayesian (hard):    {acc_hard:.4f}  "
          f"({'[PASS]' if acc_hard >= best_fixed else '[vs]'} vs best fixed)")
    print(f"  Bayesian (soft):    {acc_soft:.4f}  "
          f"({'[PASS]' if acc_soft >= best_fixed else '[vs]'} vs best fixed)")
    print(f"  Kruskal-Wallis p = {p_val:.6f}  "
          f"{'-> types have significantly different lambda' if p_val < 0.05 else ''}")


if __name__ == "__main__":
    main()
