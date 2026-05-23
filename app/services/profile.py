"""
用户画像管理：初始化、EMA 更新、查询
"""

from app.services.storage import StorageService
from app.services.feature_extractor import FeatureExtractor
from app.models.user import UserProfile
from app.config import EMA_DEFAULT_DECAY, EMA_DEFAULT_LR
from pipeline import MATCH_FEATURES


class ProfileService:
    """用户画像管理：初始化、EMA 更新、查询"""

    def __init__(self, storage: StorageService, feature_extractor: FeatureExtractor):
        self.storage = storage
        self.extractor = feature_extractor
        self.default_decay = EMA_DEFAULT_DECAY
        self.default_lr = EMA_DEFAULT_LR

    def create_profile_from_questionnaire(
        self, user_id: str, score_result: dict
    ) -> UserProfile:
        """
        从问卷评分结果创建初始画像。

        score_result: QuestionnaireService.score_answers() 的返回值
        """
        features = score_result.get("features", {})
        industry_vector = score_result.get("industry_vector", {})

        profile = UserProfile(
            user_id=user_id,
            beta=score_result.get("beta", 0.5),
            risk_tolerance=score_result.get("risk_tolerance", 3),
            initial_capital=score_result.get("initial_capital", 0.0),
            features=features,
            industry_vector=industry_vector,
            feature_momentum={f: 0.0 for f in MATCH_FEATURES},
            decay_factor=self.default_decay,
            update_count=0,
            confidence_level="low",
            source="questionnaire",
        )

        # 保存初始快照
        profile.history.append({
            "update": 0,
            "features": dict(features),
            "timestamp": profile.last_updated,
        })

        self.storage.save_profile(profile)
        return profile

    def update_profile_with_trades(
        self, user_id: str, trades_df,
        strategy_features: dict | None = None,
        strategy_nav: dict | None = None,
        window_days: int | None = None,
    ) -> UserProfile:
        """
        用交易数据更新画像（EMA 动量方式）。

        1. 从交易数据提取新特征（支持滚动窗口）
        2. 加载当前画像
        3. EMA 更新
        4. 持久化
        """
        # 提取新特征
        if window_days is not None:
            new_features = self.extractor.extract_features_with_window(trades_df, window_days)
        else:
            new_features = self.extractor.extract_user_features(trades_df)

        # 加载当前画像
        profile = self.storage.get_profile(user_id)
        if profile is None:
            # 如果不存在画像，创建一个以交易数据为基础的初始画像
            profile = UserProfile(
                user_id=user_id,
                features=new_features,
                feature_momentum={f: 0.0 for f in MATCH_FEATURES},
                source="trade_data",
                confidence_level="low",
            )
        else:
            # EMA 更新
            profile.ema_update(new_features, lr=self.default_lr, base_decay=profile.decay_factor)

        self.storage.save_profile(profile)
        return profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self.storage.get_profile(user_id)

    def get_confidence(self, user_id: str) -> str:
        profile = self.get_profile(user_id)
        return profile.confidence_level if profile else "none"

    def get_recommendation_ready_features(self, user_id: str) -> dict[str, float] | None:
        """获取可用于推荐的画像特征（若不存在返回 None）"""
        profile = self.get_profile(user_id)
        if profile is None or not profile.features:
            return None
        return profile.features

    def update_beta(self, user_id: str, new_beta: float) -> UserProfile | None:
        """手动调整 β"""
        profile = self.get_profile(user_id)
        if profile is None:
            return None
        profile.beta = max(0.0, min(1.0, new_beta))
        self.storage.save_profile(profile)
        return profile

    def clear_trade_data(self, user_id: str) -> bool:
        """清除用户的交易数据（保留画像）"""
        import os
        from pathlib import Path
        trades_dir = self.storage.trades_dir
        for f in trades_dir.glob(f"{user_id}_*.csv"):
            f.unlink()
        return True
