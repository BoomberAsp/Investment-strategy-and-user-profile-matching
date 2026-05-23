"""
匹配引擎抽象基类 + 注册表
"""

from abc import ABC, abstractmethod


class MatchingBackend(ABC):
    """所有匹配算法的统一接口"""

    _is_fitted: bool = False

    @abstractmethod
    def name(self) -> str:
        """后端名称，如 'statistical' / 'word2vec' / 'transformer'"""
        ...

    @abstractmethod
    def fit(self, strategy_features: dict, strategy_nav: dict | None = None) -> None:
        """
        加载策略数据，预计算匹配模型。
        - 统计后端: 预计算 PCA 模型 + 策略 PCA 坐标
        - Word2Vec 后端: 加载训练好的嵌入模型 + 策略向量
        """
        ...

    @abstractmethod
    def predict(
        self, user_features: dict[str, float], beta: float = 0.5, top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        """
        给定用户特征，返回 Top-N 推荐。
        返回: {"top3": [...], "explanation": {...}, "metric_used": str}
        """
        ...

    @abstractmethod
    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        """返回所有可用度量下的推荐结果（用于可视化对比）"""
        ...


class BackendRegistry:
    """匹配后端注册表，支持运行时切换"""

    def __init__(self):
        self._backends: dict[str, MatchingBackend] = {}

    def register(self, backend: MatchingBackend):
        self._backends[backend.name()] = backend

    def get(self, name: str) -> MatchingBackend | None:
        return self._backends.get(name)

    def list_available(self) -> list[str]:
        return list(self._backends.keys())

    def list_active(self) -> list[str]:
        """返回已 fit 过的后端"""
        return [name for name, b in self._backends.items() if getattr(b, "_is_fitted", False)]
