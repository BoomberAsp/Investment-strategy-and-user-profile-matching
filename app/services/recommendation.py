"""
推荐调度器：通过 MatchingBackend 抽象接口调用具体算法
"""

from datetime import datetime, timezone

from app.services.matching_backend import BackendRegistry
from app.services.popup_generator import PopupGenerator
from app.models.user import MatchingResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecommendationService:
    """
    推荐调度器。通过 MatchingBackend 抽象接口调用具体算法。
    """

    def __init__(self, registry: BackendRegistry, popup_gen: PopupGenerator):
        self.registry = registry
        self.popup_gen = popup_gen
        self._strategy_nav_info = {}

    def set_strategy_nav_info(self, nav_info: dict):
        """
        设置策略净值摘要信息（用于弹窗话术）。
        nav_info: {strategy_id: {"annual_return": float, "max_drawdown": float}}
        """
        self._strategy_nav_info = nav_info

    def recommend(
        self, user_id: str, user_features: dict, profile,
        backend_name: str = "statistical", top_n: int = 3,
    ) -> MatchingResult:
        """
        主推荐入口。

        参数 profile: UserProfile 对象
        """
        backend = self.registry.get(backend_name)
        if backend is None:
            raise ValueError(f"Backend '{backend_name}' not found. Available: {self.registry.list_available()}")

        result = backend.predict(
            user_features=user_features,
            beta=profile.beta,
            top_n=top_n,
            industry_vector=profile.industry_vector if profile.industry_vector else None,
        )

        top3 = result.get("top3", [])
        explanation = result.get("explanation", {})

        # 生成弹窗话术
        if top3:
            top_strategy = top3[0]["strategy"]
            popup_text = self.popup_gen.generate(
                strategy_id=top_strategy,
                similarity=top3[0]["similarity"],
                top_similar_dims=explanation.get("most_similar_dimensions", []),
                top_different_dims=explanation.get("most_different_dimensions", []),
                strategy_nav_info=self._strategy_nav_info.get(top_strategy),
                confidence=profile.confidence_level,
            )
        else:
            popup_text = "暂无匹配策略。"

        return MatchingResult(
            user_id=user_id,
            backend=backend_name,
            timestamp=_now(),
            top_n=top3,
            explanation=explanation,
            popup_text=popup_text,
            confidence=profile.confidence_level,
            metric_used=result.get("metric_used", ""),
        )

    def compare_backends(
        self, user_id: str, user_features: dict, profile, top_n: int = 3,
    ) -> dict[str, MatchingResult]:
        """
        对比所有已激活后端的推荐结果。
        返回: {"statistical": result1, "word2vec": result2, ...}
        """
        results = {}
        for backend_name in self.registry.list_active():
            try:
                results[backend_name] = self.recommend(
                    user_id, user_features, profile,
                    backend_name=backend_name, top_n=top_n,
                )
            except Exception:
                pass
        return results

    def get_all_metrics_recommendation(
        self, user_features: dict, profile, top_n: int = 3,
    ) -> dict:
        """
        返回所有可用度量下的推荐结果（用于可视化对比）。
        """
        backend = self.registry.get("statistical")
        if backend is None:
            return {}
        return backend.get_all_metrics(user_features, beta=profile.beta)
