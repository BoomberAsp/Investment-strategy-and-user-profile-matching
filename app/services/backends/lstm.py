"""
LSTM 128 维序列风格匹配后端

加载 DLMethod 团队预训练的输出文件：
  - matching_phase2_lstm.csv  → 账户 × 策略 LSTM 相似度矩阵
  - final_recommendations.csv → Top-N 推荐长表（含 Phase1/Phase2 排名）
  - shap_analysis.json        → SHAP 特征归因结果

本质上是"查表式"匹配：用户上传交易数据后，用账户名映射
直接从预计算的相似度矩阵中读取该账户的推荐结果。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.matching_backend import MatchingBackend
from app.config import (
    LSTM_SIMILARITY_MATRIX,
    LSTM_RECOMMENDATIONS,
    LSTM_SHAP_ANALYSIS,
)


class LSTMBackend(MatchingBackend):
    """
    LSTM 序列风格匹配后端（DLMethod 团队输出）

    使用匹配_phase2_lstm.csv 中的预计算相似度矩阵，
    按账户名映射查表返回推荐结果。
    """

    def __init__(self):
        self._is_fitted = False
        self._sim_matrix: pd.DataFrame | None = None      # Account × Strategy
        self._recommendations: pd.DataFrame | None = None  # 推荐长表
        self._shap_report: dict | None = None
        self._strategy_ids: list[str] = []
        self._account_names: list[str] = []
        # 用户 → LSTM 账户名映射（按上传顺序分配）
        self._user_to_account: dict[str, str] = {}
        self._assigned_counter: int = 0

    def name(self) -> str:
        return "lstm"

    def fit(self, strategy_features: dict, strategy_nav: dict | None = None):
        """
        加载预计算的 LSTM 相似度矩阵和推荐表。
        """
        # 加载相似度矩阵
        if not LSTM_SIMILARITY_MATRIX.exists():
            raise FileNotFoundError(
                f"LSTM 相似度矩阵文件不存在: {LSTM_SIMILARITY_MATRIX}\n"
                f"请先运行 DLMethod 流水线 (step1-step7)。"
            )

        self._sim_matrix = pd.read_csv(LSTM_SIMILARITY_MATRIX, index_col=0)
        self._strategy_ids = list(self._sim_matrix.columns)
        self._account_names = list(self._sim_matrix.index)

        # 加载推荐长表
        if LSTM_RECOMMENDATIONS.exists():
            self._recommendations = pd.read_csv(LSTM_RECOMMENDATIONS)

        # 加载 SHAP 归因
        if LSTM_SHAP_ANALYSIS.exists():
            with open(LSTM_SHAP_ANALYSIS, "r", encoding="utf-8") as f:
                self._shap_report = json.load(f)

        self._is_fitted = True

    def assign_account(self, user_id: str) -> str | None:
        """
        为上传了交易数据的用户分配 LSTM 账户名（Account_A/B/C）。
        仅支持前 3 个上传交易数据的用户，超出返回 None。
        """
        if user_id in self._user_to_account:
            return self._user_to_account[user_id]

        if self._assigned_counter >= len(self._account_names):
            return None

        account_name = self._account_names[self._assigned_counter]
        self._user_to_account[user_id] = account_name
        self._assigned_counter += 1
        return account_name

    def get_assigned_accounts(self) -> dict[str, str]:
        """返回已分配的用户 → 账户名映射"""
        return dict(self._user_to_account)

    def predict(
        self, user_features: dict[str, float], beta: float = 0.5, top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        """
        从预计算矩阵中读取用户的 LSTM 推荐。

        需要用户已上传交易数据并已分配 LSTM 账户名。
        注意：user_features 和 beta 参数在此后端中不使用，
        因为匹配结果完全来自预计算的 LSTM 序列风格相似度。
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMBackend not fitted. Call fit() first.")

        # 需要外部调用 assign_account 先分配账户名
        # 这里不自动分配，因为触发时机由上传交易数据逻辑控制
        account_name = None
        for uid, acct in self._user_to_account.items():
            # 匹配最近分配的用户（简化：通过 user_id 模糊匹配）
            if uid == user_features.get("_user_id"):
                account_name = acct
                break

        if account_name is None or account_name not in self._sim_matrix.index:
            return {
                "top3": [],
                "explanation": {},
                "metric_used": "LSTM Cosine Similarity (no account mapped)",
                "note": (
                    "该用户尚未分配 LSTM 账户名。"
                    "请确保用户已上传交易数据，且系统中有可用的 LSTM 账户槽位（最多 3 个）。"
                ),
                "all_sims": {},
                "phase1_rank": {},
                "phase2_rank": {},
            }

        # 从矩阵中取出该账户的相似度向量
        sims = self._sim_matrix.loc[account_name]
        ranked = sims.sort_values(ascending=False)

        top3 = []
        for rank_pos, (strategy, sim) in enumerate(ranked.head(top_n).items()):
            top3.append({
                "strategy": strategy,
                "similarity": float(sim),
                "rank": rank_pos + 1,
            })

        # 从推荐长表中获取 Phase1/Phase2 排名
        phase1_rank = {}
        phase2_rank = {}
        if self._recommendations is not None:
            acct_recs = self._recommendations[self._recommendations["account"] == account_name]
            for _, row in acct_recs.iterrows():
                phase1_rank[row["strategy"]] = int(row["phase1_rank"])
                phase2_rank[row["strategy"]] = int(row["phase2_rank"])

        # 构建 SHAP 归因
        explanation = self._build_explanation(account_name, top3)

        return {
            "top3": top3,
            "explanation": explanation,
            "metric_used": "LSTM Cosine Similarity (128-dim BiLSTM embedding)",
            "all_sims": {s: float(v) for s, v in sims.items()},
            "phase1_rank": phase1_rank,
            "phase2_rank": phase2_rank,
            "lstm_account": account_name,
        }

    def predict_for_account(self, account_name: str, top_n: int = 3) -> dict:
        """
        直接按 LSTM 账户名查询推荐（供 FusionBackend 内部调用）。
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMBackend not fitted. Call fit() first.")

        if account_name not in self._sim_matrix.index:
            return {"all_sims": {}, "top3": [], "phase1_rank": {}, "phase2_rank": {}}

        sims = self._sim_matrix.loc[account_name]
        ranked = sims.sort_values(ascending=False)

        top3 = []
        for rank_pos, (strategy, sim) in enumerate(ranked.head(top_n).items()):
            top3.append({
                "strategy": strategy,
                "similarity": float(sim),
                "rank": rank_pos + 1,
            })

        phase1_rank = {}
        phase2_rank = {}
        if self._recommendations is not None:
            acct_recs = self._recommendations[self._recommendations["account"] == account_name]
            for _, row in acct_recs.iterrows():
                phase1_rank[row["strategy"]] = int(row["phase1_rank"])
                phase2_rank[row["strategy"]] = int(row["phase2_rank"])

        return {
            "all_sims": {s: float(v) for s, v in sims.items()},
            "top3": top3,
            "phase1_rank": phase1_rank,
            "phase2_rank": phase2_rank,
        }

    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        """LSTM 后端只有一种度量，返回 LSTM cosine similarity"""
        if not self._is_fitted:
            return {"lstm": {"similarity": None, "metric_name": "LSTM (not fitted)"}}
        return {
            "lstm": {
                "similarity": None,
                "metric_name": "LSTM Cosine Similarity (128-dim)",
            }
        }

    def _build_explanation(self, account_name: str, top3: list[dict]) -> dict:
        """构建可解释归因，包含 SHAP Top 特征"""
        explanation = {
            "most_similar_dimensions": [],
            "most_different_dimensions": [],
            "lstm_shap_top_features": [],
            "lstm_account": account_name,
        }

        if self._shap_report and "top_features_by_shap" in self._shap_report:
            for feat_name, shap_val in self._shap_report["top_features_by_shap"][:5]:
                explanation["lstm_shap_top_features"].append({
                    "feature": feat_name,
                    "mean_abs_shap": round(float(shap_val), 4),
                })

        return explanation

    def get_strategy_ids(self) -> list[str]:
        return list(self._strategy_ids)
