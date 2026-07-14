"""
Phase 4 — Hard Negative Sample Construction.

Select 10 strategy pairs where cos_sim > 0.7 AND norm_ratio > 1.5
("similar direction, different magnitude"). For each pair, users from
strategy A face strategy B as a hard negative — this is where radial
penalty cosine should outperform pure cosine by leveraging norm differences.

Output:
    data/adversarial_pairs.json       — selected 10 pairs + selection metadata
    data/adversarial_test_set.csv     — user_id, pos_strategy, hard_neg_strategy, pair_id
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from experiments.expB1_user_strategy.utils import (
    MATCH_FEATURES, DATA_DIR,
)

N_PAIRS = 10
COS_THRESHOLD = 0.7
NORM_RATIO_THRESHOLD = 1.5


def main():
    print("=" * 60)
    print("  Phase 4 — Hard Negative Sample Construction")
    print("=" * 60)

    strat = pd.read_csv(DATA_DIR / "strategy_features.csv", index_col=0)
    arr = strat[MATCH_FEATURES].values.astype(float)
    names = list(strat.index)
    norms = np.linalg.norm(arr, axis=1)
    cos_mat = cosine_similarity(arr)
    n = len(names)

    # ── find all candidate pairs ──
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_val = float(cos_mat[i, j])
            nr = norms[i] / norms[j] if norms[i] > norms[j] else norms[j] / norms[i]
            if cos_val > COS_THRESHOLD and nr > NORM_RATIO_THRESHOLD:
                candidates.append({
                    "idx_i": i, "idx_j": j,
                    "name_i": names[i], "name_j": names[j],
                    "cos_sim": round(cos_val, 4),
                    "norm_ratio": round(nr, 4),
                    "norm_i": round(float(norms[i]), 2),
                    "norm_j": round(float(norms[j]), 2),
                })
    print(f"\n  Candidate pairs (cos > {COS_THRESHOLD}, "
          f"norm_ratio > {NORM_RATIO_THRESHOLD}): {len(candidates)}")

    # ── greedy diverse selection: prefer strategies not already picked ──
    candidates.sort(key=lambda x: (x["cos_sim"], x["norm_ratio"]), reverse=True)
    selected = []
    used = set()
    for c in candidates:
        # Prefer pairs where neither strategy has been used
        score = (c["name_i"] not in used) + (c["name_j"] not in used)
        c["novelty"] = score
        selected.append(c)

    # Sort by novelty (fresh strategies first), then cos_sim, then norm_ratio
    selected.sort(key=lambda x: (x["novelty"], x["cos_sim"], x["norm_ratio"]),
                  reverse=True)
    selected = selected[:N_PAIRS]

    print(f"\n  Selected {len(selected)} diverse pairs:")
    for i, c in enumerate(selected):
        print(f"    [{i}] cos={c['cos_sim']:.4f}  norm_ratio={c['norm_ratio']:.2f}")
        print(f"        {c['name_i']}  (norm={c['norm_i']:.1f})")
        print(f"        {c['name_j']}  (norm={c['norm_j']:.1f})")

    # ── build adversarial test set ──
    labels = pd.read_csv(DATA_DIR / "user_labels.csv")
    rows = []
    for pair_id, c in enumerate(selected):
        for s_name, hn_name in [(c["name_i"], c["name_j"]),
                                 (c["name_j"], c["name_i"])]:
            users = labels[labels["source_strategy"] == s_name]
            for _, u_row in users.iterrows():
                rows.append({
                    "user_id": u_row["user_id"],
                    "pos_strategy": s_name,
                    "hard_neg_strategy": hn_name,
                    "pair_id": pair_id,
                    "pair_cos_sim": c["cos_sim"],
                    "pair_norm_ratio": c["norm_ratio"],
                })

    adv_df = pd.DataFrame(rows)
    print(f"\n  Adversarial test set: {len(adv_df)} user-strategy triples")
    print(f"    Unique users: {adv_df['user_id'].nunique()}")
    print(f"    Unique pairs: {adv_df['pair_id'].nunique()}")

    # ── save ──
    pairs_out = []
    for i, c in enumerate(selected):
        pairs_out.append({
            "pair_id": i,
            "strategy_a": c["name_i"],
            "strategy_b": c["name_j"],
            "cos_sim": c["cos_sim"],
            "norm_ratio": c["norm_ratio"],
            "norm_a": c["norm_i"],
            "norm_b": c["norm_j"],
        })

    output = {
        "description": (
            f"Hard negative strategy pairs: cos_sim > {COS_THRESHOLD} AND "
            f"norm_ratio > {NORM_RATIO_THRESHOLD}. Selected {N_PAIRS} diverse "
            "pairs via greedy novelty-first selection."
        ),
        "n_candidate_pairs": len(candidates),
        "n_selected": len(selected),
        "pairs": pairs_out,
    }
    with open(DATA_DIR / "adversarial_pairs.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    adv_df.to_csv(DATA_DIR / "adversarial_test_set.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Saved: {DATA_DIR / 'adversarial_pairs.json'}")
    print(f"  Saved: {DATA_DIR / 'adversarial_test_set.csv'}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
