"""
融合后端：统计方法 + LSTM 加权融合

融合公式（来自 ML_INTEGRATION_GUIDE.md）：
    final_score = α * stat_norm + (1-α) * ml_norm

默认 α = 0.7（统计方法为主，LSTM 为序列风格辅助项）。
Min-Max 归一化到 0-1，避免量纲不同。
"""

from sklearn.preprocessing import MinMaxScaler

from app.services.matching_backend import MatchingBackend
from app.config import FUSION_ALPHA


class FusionBackend(MatchingBackend):
    """
    统计 + LSTM 加权融合后端

    需要同时注册并 fit 了 StatisticalBackend 和 LSTMBackend。
    """

    def __init__(self, stat_backend, lstm_backend, alpha: float = FUSION_ALPHA):
        self._stat_backend = stat_backend
        self._lstm_backend = lstm_backend
        self._alpha = alpha
        self._is_fitted = False
        self._strategy_ids: list[str] = []
        self._user_to_account: dict[str, str] = {}

    def name(self) -> str:
        return "fusion"

    def fit(self, strategy_features: dict, strategy_nav: dict | None = None):
        """
        确保 stat_backend 和 lstm_backend 都已 fit。
        记录共同的策略 ID 集合。

        如果两边没有共同的策略 ID，仍标记为 fitted，
        但 predict() 会降级为纯统计推荐。
        """
        stat_ids = set(self._stat_backend.get_strategy_ids())
        lstm_ids = set(self._lstm_backend.get_strategy_ids())
        self._strategy_ids = sorted(stat_ids & lstm_ids)
        self._has_overlap = len(self._strategy_ids) > 0

        # 不 raise，即使没有共同策略 —— predict 中降级处理
        self._is_fitted = True

    def predict(
        self, user_features: dict[str, float], beta: float = 0.5, top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        """
        1. 调用 StatisticalBackend.predict() 得到统计得分
        2. 调用 LSTMBackend.predict() 得到 LSTM 得分
        3. Min-Max 归一化
        4. 加权融合
        5. 返回 Top-N + 双源归因
        """
        if not self._is_fitted:
            raise RuntimeError("FusionBackend not fitted. Call fit() first.")

        # 无共同策略时降级为纯统计
        if not self._has_overlap:
            stat_result = self._stat_backend.predict(user_features, beta, top_n=top_n, industry_vector=industry_vector)
            return {
                "top3": stat_result.get("top3", [])[:top_n],
                "explanation": {
                    **stat_result.get("explanation", {}),
                    "fusion_note": "统计后端和 LSTM 后端使用不同策略集合，仅展示统计方法结果",
                },
                "metric_used": f"Fusion (stat-only fallback, no common strategies)",
                "all_sims": stat_result.get("all_sims", {}),
                "phase1_rank": {},
                "phase2_rank": {},
                "stat_score": {s: float(v) for s, v in stat_result.get("all_sims", {}).items()},
                "ml_score": {},
            }

        # Step 1: 统计得分
        stat_result = self._stat_backend.predict(user_features, beta, top_n=999, industry_vector=industry_vector)
        stat_all_sims = stat_result.get("all_sims", {})

        # Step 2: LSTM 得分
        # 需要先找到用户对应的 LSTM 账户名
        user_id = user_features.get("_user_id", "")
        lstm_result = self._lstm_backend.predict(user_features, beta, top_n=999, industry_vector=industry_vector)
        ml_all_sims = lstm_result.get("all_sims", {})

        if not ml_all_sims:
            # LSTM 无映射账户 → 降级为纯统计推荐
            return {
                "top3": stat_result.get("top3", [])[:top_n],
                "explanation": {
                    **stat_result.get("explanation", {}),
                    "fusion_note": "LSTM 账户未分配，仅使用统计方法结果",
                },
                "metric_used": f"Fusion (α={self._alpha}, stat-only fallback)",
                "all_sims": stat_all_sims,
                "phase1_rank": {},
                "phase2_rank": {},
                "stat_score": {s: float(v) for s, v in stat_all_sims.items()},
                "ml_score": {},
            }

        # Step 3: 对齐共同策略
        common_strategies = set(stat_all_sims.keys()) & set(ml_all_sims.keys())
        if not common_strategies:
            raise ValueError("统计得分和 LSTM 得分没有共同的策略，无法融合。")

        # Step 4: Min-Max 归一化
        stat_values = [[stat_all_sims[s]] for s in common_strategies]
        ml_values = [[ml_all_sims[s]] for s in common_strategies]

        stat_norm_flat = MinMaxScaler().fit_transform(stat_values).flatten()
        ml_norm_flat = MinMaxScaler().fit_transform(ml_values).flatten()

        stat_norm = dict(zip(common_strategies, stat_norm_flat))
        ml_norm = dict(zip(common_strategies, ml_norm_flat))

        # Step 5: 加权融合
        final_scores = {}
        for s in common_strategies:
            final_scores[s] = self._alpha * stat_norm[s] + (1 - self._alpha) * ml_norm[s]

        # Step 6: 排序取 Top-N
        sorted_strategies = sorted(final_scores.items(), key=lambda x: -x[1])

        top3 = []
        for rank_pos, (strategy, score) in enumerate(sorted_strategies[:top_n]):
            top3.append({
                "strategy": strategy,
                "similarity": float(score),
                "rank": rank_pos + 1,
            })

        # Phase1/Phase2 排名
        phase1_rank = lstm_result.get("phase1_rank", {})
        phase2_rank = lstm_result.get("phase2_rank", {})

        # 构建融合归因
        explanation = {
            "fusion_alpha": self._alpha,
            "stat_contribution": f"{self._alpha * 100:.0f}%",
            "ml_contribution": f"{(1 - self._alpha) * 100:.0f}%",
            "most_similar_dimensions": stat_result.get("explanation", {}).get("most_similar_dimensions", []),
            "most_different_dimensions": stat_result.get("explanation", {}).get("most_different_dimensions", []),
            "lstm_shap_top_features": lstm_result.get("explanation", {}).get("lstm_shap_top_features", []),
        }

        return {
            "top3": top3,
            "explanation": explanation,
            "metric_used": f"Fusion (α={self._alpha}: {self._alpha * 100:.0f}% stat + {(1 - self._alpha) * 100:.0f}% LSTM)",
            "all_sims": final_scores,
            "phase1_rank": phase1_rank,
            "phase2_rank": phase2_rank,
            "stat_score": stat_norm,
            "ml_score": ml_norm,
        }

    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        """返回融合后的度量（同时包含原始分量）"""
        if not self._is_fitted:
            return {"fusion": {"similarity": None, "metric_name": "Fusion (not fitted)"}}

        # 获取统计侧的多种度量
        stat_metrics = self._stat_backend.get_all_metrics(user_features, beta)
        stat_metrics["fusion"] = {
            "similarity": None,
            "metric_name": f"Fusion (α={self._alpha})",
        }
        return stat_metrics

    def set_alpha(self, alpha: float):
        """动态调整融合权重"""
        self._alpha = alpha

    def get_strategy_ids(self) -> list[str]:
        return list(self._strategy_ids)
