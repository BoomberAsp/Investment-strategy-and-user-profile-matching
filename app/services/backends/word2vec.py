"""
Word2Vec 嵌入匹配后端（预留占位）

当前仅返回占位结果，Phase 3 实现完整逻辑。
与 generate_simulated_data.py 生成的 token_sequences.txt 对接。
"""

from app.services.matching_backend import MatchingBackend


class Word2VecBackend(MatchingBackend):
    """
    Token 化 + Word2Vec 嵌入（技术储备）

    当前为占位实现，返回 NotImplementedError。
    """

    def __init__(self):
        self._is_fitted = False
        self._strategy_ids = []
        self._strategy_vectors = {}

    def name(self) -> str:
        return "word2vec"

    def fit(self, strategy_features: dict, strategy_nav: dict | None = None):
        """
        加载 Word2Vec 模型并计算策略文档向量。

        Phase 3 实现:
        1. 加载 gensim Word2Vec 模型（从 output/simulated_data/ 目录）
        2. 读取每个策略的 Token 序列
        3. 计算文档向量（Token 嵌入平均）
        """
        self._strategy_ids = sorted(strategy_features.keys())
        # TODO: 加载 Word2Vec 模型
        # from gensim.models import Word2Vec
        # model = Word2Vec.load(...)
        # for sid, tokens in strategy_token_sequences.items():
        #     self._strategy_vectors[sid] = model.wv[tokens].mean(axis=0)
        self._is_fitted = False  # 尚未真正拟合

    def predict(
        self, user_features: dict[str, float], beta: float = 0.5, top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        if not self._is_fitted:
            return {
                "top3": [],
                "explanation": {},
                "metric_used": "Word2Vec (not yet implemented)",
                "note": "Word2Vec backend is a Phase 3 feature. "
                        "Please use the Statistical backend for now.",
            }

        # TODO: 用户 Token 序列 -> 文档向量 -> 余弦相似度
        return {"top3": [], "explanation": {}}

    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        if not self._is_fitted:
            return {"cosine": {"similarity": None, "metric_name": "Word2Vec (not implemented)"}}
        return {}
