"""
PCA + 径向惩罚余弦相似度（当前主后端）

直接从 pipeline.py 导入函数，不修改原代码。
"""

import numpy as np

from app.services.matching_backend import MatchingBackend
from app.config import DEFAULT_BETA, DEFAULT_LAMBDA


class StatisticalBackend(MatchingBackend):
    """
    统计匹配后端：PCA 降维 + 径向惩罚余弦相似度
    """

    def __init__(self):
        self._strategy_features = {}
        self._strategy_ids = []
        self._pca = None
        self._scaler_std = None
        self._strategy_pca_coords = None
        self._n_strategies = 0
        self._is_fitted = False
        self._beta = DEFAULT_BETA
        self._lam = DEFAULT_LAMBDA

    def name(self) -> str:
        return "statistical"

    def fit(self, strategy_features: dict, strategy_nav: dict | None = None):
        """
        预计算 PCA 模型。
        strategy_features: {strategy_id: {feature: value}}
        """
        from pipeline import (
            apply_beta_weighting, build_feature_matrix, apply_pca,
            MATCH_FEATURES, FEATURE_GROUPS,
        )

        self._strategy_features = strategy_features
        self._strategy_ids = sorted(strategy_features.keys())
        self._n_strategies = len(self._strategy_ids)
        self._beta = DEFAULT_BETA
        self._lam = DEFAULT_LAMBDA

        # 构建一个空的"占位用户"来使 PCA 拟合在策略+用户联合空间上
        # 由于用户在运行时才提供，我们仅在策略特征上拟合 PCA
        strategy_weighted = {
            sid: apply_beta_weighting(strategy_features[sid], DEFAULT_BETA)
            for sid in self._strategy_ids
        }

        # 构建策略特征矩阵
        S = np.array([
            [strategy_weighted[sid][f] for f in MATCH_FEATURES]
            for sid in self._strategy_ids
        ])
        S = np.nan_to_num(S, nan=0.0, posinf=10.0, neginf=-10.0)

        # PCA 拟合（90% 方差）
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler_std = StandardScaler()
        X_scaled = scaler_std.fit_transform(S)

        pca = PCA(n_components=0.90)
        pca.fit(X_scaled)

        # 投影：去均值但不缩放
        scaler_center = StandardScaler(with_std=False)
        X_centered = scaler_center.fit_transform(S)
        X_centered = np.nan_to_num(X_centered, nan=0.0)

        self._strategy_pca_coords = X_centered @ pca.components_.T
        self._pca = pca
        self._scaler_std = scaler_std
        self._scaler_center = scaler_center
        self._is_fitted = True

    def _project_user(self, user_features: dict[str, float], beta: float) -> np.ndarray:
        """将用户特征投影到预计算的 PCA 空间"""
        from pipeline import apply_beta_weighting, MATCH_FEATURES

        weighted = apply_beta_weighting(user_features, beta)
        u = np.array([[weighted[f] for f in MATCH_FEATURES]])
        u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)

        # 去均值但不缩放（与 fit 时一致）
        u_centered = self._scaler_center.transform(u)
        u_centered = np.nan_to_num(u_centered, nan=0.0)

        return u_centered @ self._pca.components_.T

    def predict(
        self, user_features: dict[str, float], beta: float = 0.5, top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        """
        返回 Top-N 推荐。
        """
        from pipeline import compute_radial_penalty_cosine, generate_explanation

        if not self._is_fitted:
            raise RuntimeError("StatisticalBackend not fitted. Call fit() first.")

        user_pca = self._project_user(user_features, beta)

        # 计算径向惩罚余弦
        sim_matrix = compute_radial_penalty_cosine(user_pca, self._strategy_pca_coords, lam=self._lam)

        # 排序
        sims = sim_matrix[0]
        ranked_indices = np.argsort(-sims)

        top3 = []
        for rank_pos, idx in enumerate(ranked_indices[:top_n]):
            top3.append({
                "strategy": self._strategy_ids[idx],
                "similarity": float(sims[idx]),
                "rank": rank_pos + 1,
            })

        # 生成可解释归因
        top_idx = ranked_indices[0]
        top_strategy = self._strategy_ids[top_idx]
        explanation = generate_explanation(
            "", top3[0], user_features, self._strategy_features[top_strategy]
        )

        return {
            "top3": top3,
            "explanation": explanation,
            "metric_used": f"Radial-Penalty Cosine (λ={self._lam})",
            "all_sims": {self._strategy_ids[i]: float(sims[i]) for i in range(self._n_strategies)},
        }

    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        """返回三种度量下的完整推荐结果"""
        from pipeline import (
            apply_beta_weighting, compute_similarity, MATCH_FEATURES,
        )
        from scipy.spatial.distance import cdist
        from sklearn.metrics.pairwise import cosine_similarity

        if not self._is_fitted:
            raise RuntimeError("StatisticalBackend not fitted. Call fit() first.")

        user_pca = self._project_user(user_features, beta)

        strategy_pca = self._strategy_pca_coords
        n_strategies = self._n_strategies

        # 欧式距离
        dist_euclidean = cdist(user_pca, strategy_pca, metric='euclidean')
        sim_euclidean = 1 / (1 + dist_euclidean)

        # 余弦相似度
        sim_cosine = cosine_similarity(user_pca, strategy_pca)

        # 径向惩罚余弦
        from pipeline import compute_radial_penalty_cosine
        sim_rp = compute_radial_penalty_cosine(user_pca, strategy_pca, lam=self._lam)

        return {
            "radial_penalty": {
                "similarity": sim_rp,
                "metric_name": f"Radial-Penalty Cosine (λ={self._lam})",
            },
            "cosine": {
                "similarity": sim_cosine,
                "metric_name": "Cosine Similarity",
            },
            "euclidean": {
                "similarity": sim_euclidean,
                "metric_name": "Euclidean Distance (transformed)",
            },
        }

    def get_strategy_ids(self) -> list[str]:
        return list(self._strategy_ids)

    def get_strategy_features(self) -> dict:
        return dict(self._strategy_features)

    def get_strategy_nav_info(self) -> dict:
        """返回策略净值摘要（用于弹窗话术）"""
        nav_info = {}
        for sid in self._strategy_ids:
            nav_info[sid] = {
                "annual_return": 0.0,  # 需从策略数据中计算
                "max_drawdown": 0.0,
            }
        return nav_info
