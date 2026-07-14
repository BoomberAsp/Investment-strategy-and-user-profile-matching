"""
Phase 5 — Ranking Evaluation.

Evaluate whether the unified feature space correctly ranks a user's
source strategy above unrelated strategies. Three metric families:
  - Pure cosine similarity
  - Euclidean distance (as 1/(1+d))
  - Radial-penalty cosine with multiple lambda values

Includes: full-set PRA, Recall@N, MRR, adversarial-subset analysis,
McNemar test, bootstrap confidence intervals, stratified analysis.

Output:
    output/results.json              — full numerical results
    output/recall_curve.png          — Recall@N vs N curves
    output/adversarial_comparison.png — hard-negative PRA comparison
    output/metric_heatmap.png        — per-strategy PRA heatmap
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from experiments.expB1_user_strategy.utils import (
    MATCH_FEATURES, DATA_DIR, OUTPUT_DIR,
)

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BOOTSTRAP_ITER = 1000
SEED = 42


def compute_radial_similarity(user_mat: np.ndarray, strat_mat: np.ndarray,
                               lam: float) -> np.ndarray:
    """User-strategy radial-penalty cosine: cos(u,s) * exp(-lam * |log(||u||/||s||)|)."""
    cos_sim = cosine_similarity(user_mat, strat_mat)
    user_norms = np.linalg.norm(user_mat, axis=1, keepdims=True)
    strat_norms = np.linalg.norm(strat_mat, axis=1, keepdims=True).T
    user_norms = np.clip(user_norms, 1e-10, None)
    strat_norms = np.clip(strat_norms, 1e-10, None)
    log_ratio = np.abs(np.log(user_norms / strat_norms))
    return cos_sim * np.exp(-lam * log_ratio)


def compute_all_similarities(user_mat: np.ndarray, strat_mat: np.ndarray
                              ) -> dict[str, np.ndarray]:
    """Return dict of similarity matrices for all metric variants."""
    dist_euc = cdist(user_mat, strat_mat, metric="euclidean")
    results = {
        "cosine": cosine_similarity(user_mat, strat_mat),
        "euclidean": 1.0 / (1.0 + dist_euc),
    }
    for lam in LAMBDAS:
        key = f"radial_{lam}"
        results[key] = compute_radial_similarity(user_mat, strat_mat, lam)
    return results


def compute_pra(sim_mat: np.ndarray, user_labels: np.ndarray) -> float:
    """Pairwise Ranking Accuracy: P(sim(u, pos) > sim(u, neg)) over all (u, neg) pairs."""
    n_users, n_strategies = sim_mat.shape
    correct = 0
    total = 0
    for u in range(n_users):
        pos = user_labels[u]
        pos_score = sim_mat[u, pos]
        for j in range(n_strategies):
            if j != pos:
                if pos_score > sim_mat[u, j]:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def compute_recall_at_n(sim_mat: np.ndarray, user_labels: np.ndarray,
                         n_values: tuple = (1, 3, 5)) -> dict[int, float]:
    """Recall@N: fraction of users whose source strategy ranks in top N."""
    n_users, _ = sim_mat.shape
    ranks = np.argsort(-sim_mat, axis=1)  # descending similarity
    results = {}
    for n in n_values:
        top_n = ranks[:, :n]
        hits = np.any(top_n == user_labels[:, None], axis=1)
        results[n] = float(hits.mean())
    return results


def compute_mrr(sim_mat: np.ndarray, user_labels: np.ndarray) -> float:
    """Mean Reciprocal Rank of the source strategy."""
    n_users, _ = sim_mat.shape
    ranks = np.argsort(-sim_mat, axis=1)
    reciprocal_ranks = []
    for u in range(n_users):
        pos = user_labels[u]
        rank = np.where(ranks[u] == pos)[0][0] + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def compute_adversarial_pra(sim_mat: np.ndarray,
                            adv_df: pd.DataFrame,
                            strategy_index: dict) -> dict:
    """Adversarial PRA: fraction where sim(u, pos) > sim(u, hard_neg)."""
    correct = 0
    total = 0
    for _, row in adv_df.iterrows():
        u_idx = row["_user_idx"]
        pos_idx = strategy_index[row["pos_strategy"]]
        hn_idx = strategy_index[row["hard_neg_strategy"]]
        if sim_mat[u_idx, pos_idx] > sim_mat[u_idx, hn_idx]:
            correct += 1
        total += 1
    return {
        "adversarial_pra": correct / total if total > 0 else 0.0,
        "n_pairs": total,
    }


def mcnemar_test(sim_a: np.ndarray, sim_b: np.ndarray,
                 user_labels: np.ndarray) -> dict:
    """McNemar test: does metric B outperform metric A on the same (u, neg) pairs?"""
    n_users, n_strats = sim_a.shape
    a_wins_b_loses = 0  # A correct, B wrong
    b_wins_a_loses = 0  # B correct, A wrong
    for u in range(n_users):
        pos = user_labels[u]
        pa, pb = sim_a[u, pos], sim_b[u, pos]
        for j in range(n_strats):
            if j == pos:
                continue
            a_correct = pa > sim_a[u, j]
            b_correct = pb > sim_b[u, j]
            if a_correct and not b_correct:
                a_wins_b_loses += 1
            elif b_correct and not a_correct:
                b_wins_a_loses += 1

    n_discordant = a_wins_b_loses + b_wins_a_loses
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}

    # McNemar with continuity correction
    chi2 = (abs(a_wins_b_loses - b_wins_a_loses) - 1) ** 2 / n_discordant
    p_val = 1.0 - stats.chi2.cdf(chi2, 1)
    return {
        "statistic": float(chi2),
        "p_value": float(p_val),
        "n_discordant": n_discordant,
        "a_wins_b_loses": a_wins_b_loses,
        "b_wins_a_loses": b_wins_a_loses,
    }


def bootstrap_ci(metric_fn, sim_mat: np.ndarray, user_labels: np.ndarray,
                 n_iter: int = BOOTSTRAP_ITER, alpha: float = 0.05,
                 rng: np.random.Generator = None) -> dict:
    """Bootstrap 95% CI for a scalar metric computed on (sim_mat, user_labels)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_users = sim_mat.shape[0]
    values = []
    for _ in range(n_iter):
        idx = rng.choice(n_users, size=n_users, replace=True)
        val = metric_fn(sim_mat[idx], user_labels[idx])
        values.append(val)
    values = np.array(values)
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(values)),
    }


def main():
    print("=" * 60)
    print("  Phase 5 — Ranking Evaluation")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    strat = pd.read_csv(DATA_DIR / "strategy_features.csv", index_col=0)
    users = pd.read_csv(DATA_DIR / "user_features.csv")
    labels = pd.read_csv(DATA_DIR / "user_labels.csv")
    adv_df = pd.read_csv(DATA_DIR / "adversarial_test_set.csv")

    strategy_names = list(strat.index)
    strategy_index = {name: i for i, name in enumerate(strategy_names)}
    n_strategies = len(strategy_names)

    feat_cols = [c for c in MATCH_FEATURES if c in users.columns]
    user_ids = users["user_id"].tolist()
    user_mat = users[feat_cols].values.astype(float)
    strat_mat = strat[feat_cols].values.astype(float)

    # Map each user to its source strategy index
    user_labels = np.array([strategy_index[labels.loc[i, "source_strategy"]]
                             for i in range(len(labels))])

    # Map adversarial rows to user indices
    adv_df["_user_idx"] = adv_df["user_id"].apply(lambda uid: user_ids.index(uid))

    n_users = len(user_ids)
    print(f"\n  Users: {n_users}  |  Strategies: {n_strategies}")
    print(f"  Adversarial test triples: {len(adv_df)}")

    # ── compute all similarity matrices ──
    print("\n  Computing similarity matrices ...")
    sims = compute_all_similarities(user_mat, strat_mat)

    # ── full-set evaluation ──
    print("\n" + "=" * 60)
    print("  Full-Set Evaluation")
    print("=" * 60)

    full_results = {}
    for metric_name, sim_mat in sims.items():
        pra = compute_pra(sim_mat, user_labels)
        recall = compute_recall_at_n(sim_mat, user_labels)
        mrr = compute_mrr(sim_mat, user_labels)
        full_results[metric_name] = {
            "PRA": round(pra, 4),
            "Recall@1": round(recall[1], 4),
            "Recall@3": round(recall[3], 4),
            "Recall@5": round(recall[5], 4),
            "MRR": round(mrr, 4),
        }
        print(f"\n  {metric_name}:")
        print(f"    PRA = {pra:.4f}   (random baseline = 0.5000)")
        print(f"    Recall@1 = {recall[1]:.4f}  @3 = {recall[3]:.4f}  @5 = {recall[5]:.4f}")
        print(f"    MRR = {mrr:.4f}   (random baseline ~ 0.11)")

    # ── bootstrap CIs for key metrics ──
    print("\n  Bootstrap 95% CIs (PRA):")
    for metric_name in ["cosine", "euclidean", "radial_1.0"]:
        if metric_name in sims:
            ci = bootstrap_ci(compute_pra, sims[metric_name], user_labels, rng=rng)
            full_results[metric_name]["PRA_bootstrap"] = ci
            print(f"    {metric_name:22s}  {ci['mean']:.4f}  [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # ── McNemar tests: radial vs cosine, radial vs euclidean ──
    print("\n  McNemar tests (radial λ=1.0 vs baselines):")
    ref = sims["radial_1.0"]
    for other in ["cosine", "euclidean"]:
        result = mcnemar_test(ref, sims[other], user_labels)
        winner = "radial" if result["a_wins_b_loses"] > result["b_wins_a_loses"] else other
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "n.s."
        print(f"    radial vs {other:10s}: chi2={result['statistic']:.1f}, "
              f"p={result['p_value']:.4f} ({sig}), winner={winner}, "
              f"n_discordant={result['n_discordant']}")
        full_results[f"mcnemar_radial_vs_{other}"] = result

    # ── adversarial subset ──
    print("\n" + "=" * 60)
    print("  Adversarial Subset Evaluation")
    print("=" * 60)

    adv_results = {}
    for metric_name, sim_mat in sims.items():
        adv_pra = compute_adversarial_pra(sim_mat, adv_df, strategy_index)
        adv_results[metric_name] = adv_pra
        print(f"    {metric_name:22s}  Adversarial PRA = {adv_pra['adversarial_pra']:.4f}")

    # ── stratified analysis by trade count ──
    print("\n" + "=" * 60)
    print("  Stratified Analysis (by original trade count)")
    print("=" * 60)

    bins = [(0, 200, "≤200"), (200, 500, "201-500"), (500, 2000, "501-2000"),
            (2000, 99999, ">2000")]
    stratified = {}
    for metric_name in ["cosine", "euclidean", "radial_1.0"]:
        sim_mat = sims[metric_name]
        strat_out = {}
        for lo, hi, label in bins:
            mask = labels["n_trades_original"].between(lo, hi)
            if mask.sum() == 0:
                continue
            idx = np.where(mask.values)[0]
            sub_sim = sim_mat[idx]
            sub_labels = user_labels[idx]
            pra = compute_pra(sub_sim, sub_labels)
            recall = compute_recall_at_n(sub_sim, sub_labels)
            mrr = compute_mrr(sub_sim, sub_labels)
            strat_out[label] = {
                "n_users": int(mask.sum()),
                "PRA": round(pra, 4),
                "Recall@3": round(recall[3], 4),
                "MRR": round(mrr, 4),
            }
            print(f"    {metric_name:22s}  {label:8s}  n={mask.sum():4d}  "
                  f"PRA={pra:.4f}  R@3={recall[3]:.4f}  MRR={mrr:.4f}")
        stratified[metric_name] = strat_out

    # ── save results ──
    results = {
        "description": "Experiment B.1 Phase 5: Ranking evaluation of user-strategy matching.",
        "n_users": n_users,
        "n_strategies": n_strategies,
        "n_adversarial_triples": len(adv_df),
        "lambda_values": LAMBDAS,
        "random_baselines": {
            "PRA": 0.5,
            "Recall@1": round(1 / n_strategies, 4),
            "Recall@3": round(3 / n_strategies, 4),
            "Recall@5": round(5 / n_strategies, 4),
            "MRR": round(np.mean([1 / k for k in range(1, n_strategies + 1)]), 4),
        },
        "full_results": full_results,
        "adversarial_results": adv_results,
        "stratified_analysis": stratified,
    }
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x))
    print(f"\n  Saved: {OUTPUT_DIR / 'results.json'}")

    # ── visualisation 1: Recall@N curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall curve
    ax = axes[0]
    n_vals = list(range(1, n_strategies + 1))
    metric_styles = {
        "cosine": ("o-", "#2196F3", "Cosine"),
        "euclidean": ("s-", "#4CAF50", "Euclidean"),
        "radial_1.0": ("D-", "#F44336", "Radial (λ=1.0)"),
    }
    for key, (style, color, label) in metric_styles.items():
        if key in sims:
            recall_curve = [compute_recall_at_n(sims[key], user_labels, (n,))[n]
                            for n in n_vals]
            ax.plot(n_vals, recall_curve, style, color=color, label=label, markersize=4)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("N (Top-N)")
    ax.set_ylabel("Recall@N")
    ax.set_title("Recall@N Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda sensitivity
    ax = axes[1]
    lam_labels = [k for k in sims if k.startswith("radial_")]
    lam_values = []
    lam_pras = []
    for k in lam_labels:
        lam = float(k.replace("radial_", "").replace("_", "."))
        lam_values.append(lam)
        lam_pras.append(full_results[k]["PRA"])
    order = np.argsort(lam_values)
    lam_values = np.array(lam_values)[order]
    lam_pras = np.array(lam_pras)[order]
    ax.plot(lam_values, lam_pras, "D-", color="#F44336", markersize=8, linewidth=2)
    ax.axhline(y=full_results.get("cosine", {}).get("PRA", 0), color="#2196F3",
               linestyle="--", label="Cosine baseline")
    ax.axhline(y=0.5, color="gray", linestyle=":", label="Random baseline")
    ax.set_xlabel("λ")
    ax.set_ylabel("PRA")
    ax.set_title("λ Sensitivity (PRA)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'recall_curve.png'}")

    # ── visualisation 2: adversarial comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_names = []
    adv_pras = []
    colors = []
    for key, label, color in [
        ("cosine", "Cosine", "#2196F3"),
        ("euclidean", "Euclidean", "#4CAF50"),
        ("radial_1.0", "Radial\n(λ=1.0)", "#F44336"),
    ]:
        if key in adv_results:
            metric_names.append(label)
            adv_pras.append(adv_results[key]["adversarial_pra"])
            colors.append(color)
    bars = ax.bar(metric_names, adv_pras, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, label="Random baseline")
    for bar, val in zip(bars, adv_pras):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Adversarial PRA")
    ax.set_title("Hard-Negative PRA: Direction-Similar, Norm-Different Pairs")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "adversarial_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'adversarial_comparison.png'}")

    # ── visualisation 3: per-strategy PRA heatmap ──
    per_strat_pra = {}
    for s_idx, sname in enumerate(strategy_names):
        mask = labels["source_strategy"] == sname
        u_idx = np.where(mask.values)[0]
        if len(u_idx) == 0:
            continue
        sub_sim = sims["radial_1.0"][u_idx]
        sub_labels = user_labels[u_idx]
        pra = compute_pra(sub_sim, sub_labels)
        n_trades = labels.loc[mask, "n_trades_original"].iloc[0]
        per_strat_pra[sname] = {"PRA": pra, "n_trades": n_trades}

    pra_vals = [v["PRA"] for v in per_strat_pra.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_items = sorted(per_strat_pra.items(), key=lambda x: x[1]["PRA"])
    names_sort = [item[0] for item in sorted_items]
    pra_sort = [item[1]["PRA"] for item in sorted_items]
    colors_bar = ["#F44336" if p < 0.7 else "#FF9800" if p < 0.85 else "#4CAF50"
                  for p in pra_sort]
    ax.barh(range(len(names_sort)), pra_sort, color=colors_bar, alpha=0.85, edgecolor="black")
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1.5, label="Random baseline")
    ax.set_yticks(range(len(names_sort)))
    ax.set_yticklabels(names_sort, fontsize=7)
    ax.set_xlabel("PRA (Radial λ=1.0)")
    ax.set_title("Per-Strategy Pairwise Ranking Accuracy")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_strategy_pra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'per_strategy_pra.png'}")

    # ── final summary ──
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    r = full_results["radial_1.0"]
    c = full_results.get("cosine", {})
    e = full_results.get("euclidean", {})

    # Success criteria checks
    print(f"\n  Success Criteria:")
    check1 = r["PRA"] > 0.75
    print(f"    [{'PASS' if check1 else 'FAIL'}] PRA (radial λ=1.0) > 0.75:  {r['PRA']:.4f}")
    delta_pra = r["PRA"] - c.get("PRA", 0)
    check2 = delta_pra >= 0.05
    print(f"    [{'PASS' if check2 else 'FAIL'}] PRA(radial) > PRA(cosine) + 5pp:  "
          f"Δ = {delta_pra:+.4f} ({'+' if delta_pra >= 0 else ''}{delta_pra * 100:.1f} pp)")
    check3 = r["Recall@3"] > 0.30
    print(f"    [{'PASS' if check3 else 'FAIL'}] Recall@3 (radial) > 0.30:  {r['Recall@3']:.4f}")
    adv_delta = adv_results.get("radial_1.0", {}).get("adversarial_pra", 0) - \
                adv_results.get("cosine", {}).get("adversarial_pra", 0)
    check4 = adv_delta >= 0.08
    print(f"    [{'PASS' if check4 else 'FAIL'}] Adversarial PRA(radial) > PRA(cosine) + 8pp:  "
          f"Δ = {adv_delta:+.4f} ({'+' if adv_delta >= 0 else ''}{adv_delta * 100:.1f} pp)")
    check6 = abs(full_results.get("radial_0.0", {}).get("PRA", 0) -
                 full_results.get("cosine", {}).get("PRA", 0)) < 0.001
    print(f"    [{'PASS' if check6 else 'FAIL'}] λ=0 degeneracy check: "
          f"PRA(λ=0) = PRA(cosine) within 0.001")

    all_checks = [check1, check2, check3, check4, check6]
    passed = sum(all_checks)
    print(f"\n    {passed}/{len(all_checks)} checks passed.")
    results["success_criteria"] = {
        "check1_pra_gt_075": bool(check1),
        "check2_radial_vs_cosine_5pp": bool(check2),
        "check3_recall3_gt_030": bool(check3),
        "check4_adversarial_8pp": bool(check4),
        "check6_lambda0_degeneracy": bool(check6),
        "passed": passed,
        "total": len(all_checks),
    }

    # Re-save with success criteria
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x))

    print("\n  Done.")


if __name__ == "__main__":
    main()
