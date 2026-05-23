"""
数据模型：User, UserProfile, MatchingResult
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    created_at: str = field(default_factory=_now)
    last_login: str = ""
    onboarding_status: str = "new"  # "new" | "questionnaire_done" | "active"

    @classmethod
    def create(cls, user_id: str, username: str, password: str) -> "User":
        return cls(
            user_id=user_id,
            username=username,
            password_hash=_hash_password(password),
        )

    def verify_password(self, password: str) -> bool:
        return self.password_hash == _hash_password(password)

    def touch_login(self):
        self.last_login = _now()

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "password_hash": self.password_hash,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "onboarding_status": self.onboarding_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "User":
        return cls(**d)


@dataclass
class UserProfile:
    user_id: str

    # 超参数
    beta: float = 0.5
    risk_tolerance: int = 3          # 1-5
    initial_capital: float = 0.0

    # 12 维特征向量
    features: dict = field(default_factory=dict)

    # 行业分布向量（预留）
    industry_vector: dict = field(default_factory=dict)

    # EMA 动量更新参数
    feature_momentum: dict = field(default_factory=dict)
    decay_factor: float = 0.7
    update_count: int = 0

    # 置信度
    confidence_level: str = "low"    # "low" | "medium" | "high"

    # 元数据
    source: str = "questionnaire"    # "questionnaire" | "trade_data" | "hybrid"
    last_updated: str = field(default_factory=_now)
    questionnaire_scores: dict = field(default_factory=dict)
    matching_backend: str = "statistical"

    # 历史快照（记录每次更新后的特征，用于变化轨迹可视化）
    history: list = field(default_factory=list)  # [{"update": int, "features": dict, "timestamp": str}]

    def ema_update(self, new_features: dict[str, float], lr: float = 0.3, base_decay: float = 0.7):
        """
        EMA 动量式画像更新（类比 SGD with Momentum）
        """
        t = self.update_count
        gamma_t = base_decay * (1.0 - 1.0 / (t + 2))

        for k in new_features:
            old_val = self.features.get(k, 0.0)
            g = new_features[k] - old_val

            m_old = self.feature_momentum.get(k, 0.0)
            m_new = gamma_t * m_old + (1.0 - gamma_t) * g
            self.feature_momentum[k] = m_new

            self.features[k] = old_val + lr * m_new

        self.update_count += 1
        self._update_confidence()
        self.last_updated = _now()
        self.source = "hybrid" if self.source == "questionnaire" else "trade_data"

        # 保存历史快照
        self.history.append({
            "update": self.update_count,
            "features": dict(self.features),
            "timestamp": self.last_updated,
        })

    def _update_confidence(self):
        from app.config import CONFIDENCE_MEDIUM_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD
        if self.update_count >= CONFIDENCE_HIGH_THRESHOLD:
            self.confidence_level = "high"
        elif self.update_count >= CONFIDENCE_MEDIUM_THRESHOLD:
            self.confidence_level = "medium"
        else:
            self.confidence_level = "low"

    @property
    def is_confident(self) -> bool:
        return self.confidence_level in ("medium", "high")

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "beta": self.beta,
            "risk_tolerance": self.risk_tolerance,
            "initial_capital": self.initial_capital,
            "features": self.features,
            "industry_vector": self.industry_vector,
            "feature_momentum": self.feature_momentum,
            "decay_factor": self.decay_factor,
            "update_count": self.update_count,
            "confidence_level": self.confidence_level,
            "source": self.source,
            "last_updated": self.last_updated,
            "questionnaire_scores": self.questionnaire_scores,
            "matching_backend": self.matching_backend,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        obj = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        return obj


@dataclass
class MatchingResult:
    user_id: str
    backend: str
    timestamp: str
    top_n: list[dict]          # [{"strategy": str, "similarity": float, "rank": int}, ...]
    explanation: dict
    popup_text: str
    confidence: str
    metric_used: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "backend": self.backend,
            "timestamp": self.timestamp,
            "top_n": self.top_n,
            "explanation": self.explanation,
            "popup_text": self.popup_text,
            "confidence": self.confidence,
            "metric_used": self.metric_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatchingResult":
        return cls(**d)
