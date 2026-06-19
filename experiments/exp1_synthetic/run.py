"""
Experiment 1 — Main Runner

1. Load / generate synthetic trades
2. Extract 12-dim features per investor  (reusing pipeline.py logic)
3. PCA projection
4. Compute similarity matrices  (cosine / euclidean / radial-penalty-cosine)
5. Evaluate: k-NN classification accuracy, Silhouette Score, confusion matrices
6. λ sensitivity analysis
7. Generate visualisations

Output written to  experiments/exp1_synthetic/output/
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

# ── import from project pipeline ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (                                      # noqa: E402
    fifo_pair_trades,
    extract_user_behavior_features,
    extract_user_asset_pref_features,
    extract_user_risk_proxy_features,
    MATCH_FEATURES, BEHAVIOR_FEATURES,
    ASSET_PREF_FEATURES, RISK_PROXY_FEATURES,
    FEATURE_GROUPS,
    apply_beta_weighting, build_feature_matrix, apply_pca,
    compute_radial_penalty_cosine, compute_similarity,
)
from experiments.exp1_synthetic.simulator import run_simulation  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BETA = 0.5
LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K_VALUES = [1, 3, 5]
TYPE_LABELS = {t.type_id: t.label for t in [
    type("_", (), dict(type_id="T1", label="长线分散")),
    type("_", (), dict(type_id="T2", label="长线集中")),
    type("_", (), dict(type_id="T3", label="短线高频")),
    type("_", (), dict(type_id="T4", label="短线投机")),
    type("_", (), dict(type_id="T5", label="中线均衡")),
    type("_", (), dict(type_id="T6", label="被动ETF")),
]}


# ── step 0: load / generate data ────────────────────────────────────────────

def load_or_generate(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = OUTPUT_DIR / "synthetic_trades.csv"
    labels_path = OUTPUT_DIR / "investor_labels.csv"
    if trades_path.exists() and labels_path.exists() and not force:
        return pd.read_csv(trades_path), pd.read_csv(labels_path)
    return run_simulation()


# ── step 1: convert to pipeline-compatible format ────────────────────────────

def simulated_to_pipeline_format(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Convert simulated trades to the format expected by pipeline.py user-
    feature extraction functions."""
    df = trades_df.copy()
    df["trade_date"] = pd.to_datetime(df["datetime"])
    df["action_type"] = df["action"].map({"BUY": "证券买入", "SELL": "证券卖出"})
    df["is_buy"] = df["action"] == "BUY"
    df["is_sell"] = df["action"] == "SELL"
    return df.sort_values("trade_date").reset_index(drop=True)


# ── step 2: extract features ────────────────────────────────────────────────

def extract_investor_features(investor_trades: pd.DataFrame) -> dict[str, float]:
    """Extract 12-dim feature vector from one investor's simulated trades."""
    if investor_trades.empty:
        return {f: 0.0 for f in MATCH_FEATURES}
    t = investor_trades.copy()
    behavior = extract_user_behavior_features(t)
    asset = extract_user_asset_pref_features(t)
    risk = extract_user_risk_proxy_features(t)
    return {**behavior, **asset, **risk}


def build_investor_feature_matrix(trades_df: pd.DataFrame,
                                  labels_df: pd.DataFrame,
                                  ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build investor-only feature matrix with type labels."""
    feature_rows = []
    ids = []
    y_labels = []

    for inv_id in sorted(trades_df["investor_id"].unique()):
        inv_trades = simulated_to_pipeline_format(
            trades_df[trades_df["investor_id"] == inv_id]
        )
        feat = extract_investor_features(inv_trades)
        features = [feat.get(f, 0.0) for f in MATCH_FEATURES]
        feature_rows.append(features)
        ids.append(inv_id)
        label_row = labels_df[labels_df["investor_id"] == inv_id]
        y_labels.append(label_row["type_id"].values[0] if not label_row.empty else "?")

    X = np.array(feature_rows)
    y = np.array(y_labels)
    return X, ids, y


# ── step 3: similarity & metrics ─────────────────────────────────────────────

def compute_radial_cosine_matrix(X: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Pairwise radial-penalty cosine similarity."""
    cos = cosine_similarity(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    log_ratio = np.abs(np.log(norms / norms.T))
    return cos * np.exp(-lam * log_ratio)


def compute_similarity_matrices(X: np.ndarray,
                                lam: float = 1.0) -> dict[str, np.ndarray]:
    """Return three pairwise similarity matrices."""
    dist_euc = cdist(X, X, metric="euclidean")
    return {
        "cosine":       cosine_similarity(X),
        "euclidean":    1.0 / (1.0 + dist_euc),
        "radial_cosine": compute_radial_cosine_matrix(X, lam=lam),
    }


def evaluate_knn(sim_matrix: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-out k-NN accuracy from a similarity matrix."""
    n = len(y)
    correct = 0
    for i in range(n):
        order = np.argsort(-sim_matrix[i])
        neighbours = [idx for idx in order if idx != i][:k]
        votes = y[neighbours]
        pred = np.bincount([list(set(y)).index(v) for v in votes]).argmax()
        correct += int(list(set(y))[pred] == y[i])
    return correct / n


# ── step 4: λ sensitivity ───────────────────────────────────────────────────

def run_lambda_sensitivity(X: np.ndarray, y: np.ndarray,
                           k_values: list[int] = K_VALUES,
                           lambdas: list[float] = LAMBDA_VALUES,
                           ) -> pd.DataFrame:
    """Evaluate k-NN accuracy for each λ × k combination."""
    rows = []
    for lam in lambdas:
        sim = compute_radial_cosine_matrix(X, lam=lam)
        for k in k_values:
            acc = evaluate_knn(sim, y, k)
            rows.append(dict(lambda_val=lam, k=k, accuracy=acc))
    return pd.DataFrame(rows)


# ── step 5: visualisation ───────────────────────────────────────────────────

def plot_pca_scatter(X_pca: np.ndarray, y: np.ndarray, save_path: Path):
    """PCA scatter of investors coloured by behaviour type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    type_ids = sorted(set(y))
    cmap = plt.cm.tab10
    for j, tid in enumerate(type_ids):
        mask = y == tid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=TYPE_LABELS[tid],
                   color=cmap(j), s=30, alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection of 300 Synthetic Investors")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette(results: dict[str, float], save_path: Path):
    """Bar chart comparing Silhouette Scores across metrics."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(results.keys())
    values = [results[k] for k in labels]
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Quality by Similarity Metric")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(cm_dict: dict[str, np.ndarray],
                            type_ids: list[str], save_path: Path):
    """3-panel confusion matrix (cosine / euclidean / radial_cosine)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, cm) in zip(axes, cm_dict.items()):
        im = ax.matshow(cm, cmap="YlOrRd", vmin=0, vmax=cm.max())
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(type_ids)))
        ax.set_xticklabels(type_ids, rotation=45, ha="left", fontsize=7)
        ax.set_yticks(range(len(type_ids)))
        ax.set_yticklabels(type_ids, fontsize=7)
        for i in range(len(type_ids)):
            for j in range(len(type_ids)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.suptitle("Confusion Matrices (k-NN, k=3, Leave-One-Out)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_sensitivity(df: pd.DataFrame, save_path: Path):
    """Line plot: λ vs k-NN accuracy for k=1, 3, 5."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, grp in df.groupby("k"):
        ax.plot(grp["lambda_val"], grp["accuracy"], marker="o", label=f"k={k}")
    ax.set_xlabel("λ")
    ax.set_ylabel("k-NN Accuracy")
    ax.set_title("λ Sensitivity of Radial-Penalty Cosine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norm_boxplot(X: np.ndarray, y: np.ndarray, save_path: Path):
    """Boxplot of feature-vector norms by investor type."""
    norms = np.linalg.norm(X, axis=1)
    type_ids = sorted(set(y))
    data = [norms[y == tid] for tid in type_ids]
    labels = [TYPE_LABELS[tid] for tid in type_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colours = plt.cm.tab10.colors
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("Feature Vector Norm  (‖x‖)")
    ax.set_title("Norm Distribution by Investor Type")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Experiment 1 — Synthetic Behaviour-Group Validation")
    print("=" * 64)

    # ── 0. load / generate ──
    print("\n[0] Loading synthetic trades …")
    trades_df, labels_df = load_or_generate()
    print(f"    {trades_df['investor_id'].nunique()} investors, "
          f"{len(trades_df)} trades")

    # ── 1. feature extraction ──
    print("\n[1] Extracting 12-dim features …")
    X, ids, y = build_investor_feature_matrix(trades_df, labels_df)
    X = np.nan_to_num(X, nan=0.0)
    print(f"    shape {X.shape},  type distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 2. PCA ──
    print("\n[2] PCA projection …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.90)
    pca.fit(X_scaled)
    scaler_ctr = StandardScaler(with_std=False)
    X_ctr = scaler_ctr.fit_transform(X)
    X_pca = X_ctr @ pca.components_.T
    print(f"    {pca.n_components_} components, "
          f"cumulative variance {np.cumsum(pca.explained_variance_ratio_)[-1]:.1%}")

    plot_pca_scatter(X_pca, y, OUTPUT_DIR / "pca_scatter.png")

    # ── 3. similarity matrices ──
    print("\n[3] Computing similarity matrices (cosine / euclidean / radial) …")
    sim = compute_similarity_matrices(X_pca, lam=1.0)

    # ── 4. k-NN evaluation ──
    print("\n[4] k-NN leave-one-out classification …")
    type_ids = sorted(set(y))
    results_summary: dict = {}

    for name, s in sim.items():
        print(f"\n  --- {name} ---")
        for k in K_VALUES:
            acc = evaluate_knn(s, y, k)
            print(f"    k={k}: accuracy = {acc:.4f}")

        # confusion matrix (k=3)
        cm = confusion_matrix(
            y,
            np.array([type_ids[
                np.bincount([list(type_ids).index(y[j]) for j in
                             np.argsort(-s[i])[1:4]]).argmax()
            ] for i in range(len(y))]),
            labels=type_ids,
        )
        results_summary[name] = dict(
            accuracy_k1=evaluate_knn(s, y, 1),
            accuracy_k3=evaluate_knn(s, y, 3),
            accuracy_k5=evaluate_knn(s, y, 5),
            silhouette=silhouette_score(X_pca, y, metric="euclidean"),
        )

    # ── 5. Silhouette ──
    sil_results = {}
    # Use PCA-space distances for Silhouette (precomputed not directly supported
    # with arbitrary similarity; we use the feature matrix in PCA space)
    for name in sim:
        # For Silhouette we need a distance metric, not similarity.
        # Use Euclidean distance in PCA space as the underlying metric for all
        # three — Silhouette compares cluster assignments against the SAME
        # distance space.  The fair comparison is: do the three similarity
        # metrics induce _different clusterings_ whose Silhouette on the SAME
        # distance metric differs?
        pass   # see below — we compute per-metric k-NN clusterings instead

    # Build clusterings from each metric's k-NN (k=1) classification result
    for name, s_mat in sim.items():
        preds = []
        for i in range(len(y)):
            order = np.argsort(-s_mat[i])
            neighbour = [idx for idx in order if idx != i][0]
            preds.append(y[neighbour])
        preds_arr = np.array(preds)
        sil = silhouette_score(X_pca, preds_arr, metric="euclidean") if len(set(preds_arr)) > 1 else 0.0
        sil_results[name] = sil
        print(f"    {name} Silhouette = {sil:.4f}")

    # ── 6. confusion matrices ──
    cm_dict = {}
    for name, s_mat in sim.items():
        preds = []
        for i in range(len(y)):
            order = np.argsort(-s_mat[i])
            neighbours = [idx for idx in order if idx != i][:3]
            votes = y[neighbours]
            preds.append(type_ids[np.bincount([list(type_ids).index(v) for v in votes]).argmax()])
        cm_dict[name] = confusion_matrix(y, np.array(preds), labels=type_ids)

    # ── 7. λ sensitivity ──
    print("\n[5] λ sensitivity analysis …")
    df_lambda = run_lambda_sensitivity(X_pca, y)
    for lam in LAMBDA_VALUES:
        row = df_lambda[df_lambda["lambda_val"] == lam]
        accs = ", ".join(f"k={int(r['k'])}:{r['accuracy']:.4f}" for _, r in row.iterrows())
        print(f"    λ={lam:.2f}  {accs}")

    # ── 8. norm analysis ──
    print("\n[6] Norm analysis …")
    norms = np.linalg.norm(X_pca, axis=1)
    for tid in type_ids:
        mask = y == tid
        print(f"    {TYPE_LABELS[tid]:8s}  norm μ={norms[mask].mean():.2f}  "
              f"σ={norms[mask].std():.2f}")

    # ── 9. visualisation ──
    print("\n[7] Generating figures …")
    plot_silhouette(sil_results, OUTPUT_DIR / "silhouette_comparison.png")
    plot_confusion_matrices(cm_dict, type_ids, OUTPUT_DIR / "confusion_matrices.png")
    plot_lambda_sensitivity(df_lambda, OUTPUT_DIR / "lambda_sensitivity.png")
    plot_norm_boxplot(X_pca, y, OUTPUT_DIR / "norm_boxplot.png")
    print(f"    all figures → {OUTPUT_DIR}")

    # ── 10. serialise results ──
    report = {
        "accuracy": results_summary,
        "silhouette": sil_results,
        "pca_n_components": int(pca.n_components_),
        "pca_cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "lambda_sensitivity": df_lambda.to_dict(orient="records"),
        "norm_by_type": {
            tid: {"mean": float(norms[y == tid].mean()),
                  "std": float(norms[y == tid].std())}
            for tid in type_ids
        },
    }
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 11. verdict ──
    print("\n" + "=" * 64)
    print("  Verdict")
    print("=" * 64)
    acc_cos = results_summary["cosine"]["accuracy_k3"]
    acc_rad = results_summary["radial_cosine"]["accuracy_k3"]
    delta = (acc_rad - acc_cos) * 100
    print(f"  cosine        k=3 accuracy: {acc_cos:.4f}")
    print(f"  radial_cosine k=3 accuracy: {acc_rad:.4f}  (Δ = {delta:+.1f} pp)")
    if delta >= 5:
        print(f"  [PASS] radial-penalty cosine significantly outperforms pure cosine")
    elif delta >= 0:
        print(f"  [TIE]  radial-penalty cosine marginally outperforms pure cosine")
    else:
        print(f"  [FAIL] radial-penalty cosine underperforms — investigate")

    sil_cos = sil_results["cosine"]
    sil_rad = sil_results["radial_cosine"]
    print(f"  Silhouette    cosine: {sil_cos:.4f}  "
          f"radial: {sil_rad:.4f}  euclidean: {sil_results['euclidean']:.4f}")


if __name__ == "__main__":
    main()
