"""
特征提取服务：从 pipeline.py 中抽象出的特征提取方法
不修改 pipeline.py，仅导入后重新组合。
"""

import pandas as pd
import sys
from pathlib import Path

# 确保 pipeline 模块可以被导入
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (
    parse_user_trades,
    fifo_pair_trades,
    extract_user_behavior_features,
    extract_user_asset_pref_features,
    extract_user_risk_proxy_features,
    extract_strategy_behavior_features,
    extract_strategy_asset_pref_features,
    extract_strategy_risk_proxy_features,
    MATCH_FEATURES,
    BEHAVIOR_FEATURES,
    ASSET_PREF_FEATURES,
    RISK_PROXY_FEATURES,
)


class FeatureExtractor:
    """
    从 pipeline.py 中抽象出的特征提取方法。
    不修改 pipeline.py，而是导入后重新组合。
    """

    def __init__(self):
        self.match_features = MATCH_FEATURES
        self.behavior_features = BEHAVIOR_FEATURES
        self.asset_pref_features = ASSET_PREF_FEATURES
        self.risk_proxy_features = RISK_PROXY_FEATURES

    def extract_user_features(self, trades_df: pd.DataFrame) -> dict[str, float]:
        """
        从用户交易 DataFrame 提取 12 维特征。
        要求 trades_df 与 pipeline.py 中 parse_user_trades 的输入格式一致。
        """
        trades = parse_user_trades(trades_df)
        behavior = extract_user_behavior_features(trades)
        asset_pref = extract_user_asset_pref_features(trades)
        risk = extract_user_risk_proxy_features(trades)
        return {**behavior, **asset_pref, **risk}

    def extract_strategy_features(
        self, nav_df: pd.DataFrame, trades_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        从策略净值 + 交易数据提取 12 维特征。
        """
        behavior = extract_strategy_behavior_features(trades_df, nav_df)
        asset_pref = extract_strategy_asset_pref_features(trades_df)
        risk = extract_strategy_risk_proxy_features(trades_df, nav_df)
        return {**behavior, **asset_pref, **risk}

    def extract_features_with_window(
        self, trades_df: pd.DataFrame, window_days: int
    ) -> dict[str, float]:
        """
        滚动窗口特征提取：仅使用最近 window_days 天的交易。
        """
        if window_days is None:
            return self.extract_user_features(trades_df)

        # 按日期过滤
        date_col = trades_df.columns[0]
        trades_df[date_col] = pd.to_datetime(trades_df[date_col], errors="coerce")
        max_date = trades_df[date_col].max()
        cutoff = max_date - pd.Timedelta(days=window_days)
        filtered = trades_df[trades_df[date_col] >= cutoff]

        if len(filtered) < 5:
            # 窗口内数据太少，回退到全量
            return self.extract_user_features(trades_df)

        return self.extract_user_features(filtered)

    def get_feature_means(self, strategy_features: dict) -> dict[str, float]:
        """
        计算各策略特征均值，用作问卷默认值。
        """
        if not strategy_features:
            return {f: 0.0 for f in self.match_features}

        means = {}
        for f in self.match_features:
            vals = [sf[f] for sf in strategy_features.values() if f in sf]
            means[f] = sum(vals) / len(vals) if vals else 0.0
        return means
