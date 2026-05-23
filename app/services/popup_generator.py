"""
客户端弹窗话术生成器

对应客户需求：当用户登录 APP 交易时，后台自动匹配后弹出提示。
话术格式参考项目计划书中的示例：
  "XXX 策略与您的交易风格匹配度达 70%，行业分布偏好匹配度也超过 70%。
   同时该策略过去半年的收益跑赢 XX%、回撤低 XX%，
   欢迎了解该策略，优化您的投资体验。"
"""

# 特征名中文映射
NAME_MAP = {
    "holding_period": "持仓周期",
    "turnover_rate": "换手率",
    "buy_sell_ratio": "买卖对称性",
    "hhi_concentration": "持仓集中度",
    "disposition_effect": "处置效应",
    "positive_trade_ratio": "胜率",
    "etf_ratio": "ETF偏好",
    "avg_price_preference": "价格偏好",
    "position_uniformity": "分仓均匀度",
    "avg_loss_magnitude": "亏损幅度",
    "vol_preference": "波动偏好",
    "trend_preference": "趋势偏好",
}


class PopupGenerator:
    """推荐弹窗话术生成器"""

    def generate(
        self,
        strategy_id: str,
        similarity: float,
        top_similar_dims: list[dict],
        top_different_dims: list[dict],
        strategy_nav_info: dict | None = None,
        confidence: str = "low",
    ) -> str:
        """
        生成推荐弹窗话术。

        参数:
            strategy_id: 策略名称
            similarity: 匹配度（径向惩罚余弦值）
            top_similar_dims: 最相似维度列表 [{"feature": "持仓周期", "diff": 0.01}, ...]
            top_different_dims: 最大差异维度列表
            strategy_nav_info: {"annual_return": float, "max_drawdown": float}
            confidence: 画像置信度
        """
        sim_pct = similarity * 100

        # 基础话术
        similar_names = [d["feature"] for d in top_similar_dims[:3]]
        different_names = [d["feature"] for d in top_different_dims[:2]]

        parts = []

        if sim_pct >= 0:
            parts.append(
                f"策略 {strategy_id} 与您的投资风格匹配度 {sim_pct:.1f}%。"
            )
        else:
            parts.append(
                f"策略 {strategy_id} 与您的投资风格方向相反（匹配度 {sim_pct:.1f}%），"
                f"可能不太适合您的投资习惯。"
            )

        if similar_names:
            parts.append(
                f"您在 {'、'.join(similar_names)} 方面与该策略风格最为接近，"
                f"建议进一步了解该策略。"
            )

        if different_names and sim_pct >= 0:
            parts.append(
                f"需要注意的是，您在 {'、'.join(different_names)} 方面与该策略有较大差异。"
            )

        # 附加收益/回撤信息（如果可用）
        if strategy_nav_info and strategy_nav_info.get("annual_return"):
            ann_ret = strategy_nav_info["annual_return"]
            parts.append(f"该策略年化收益率约 {ann_ret:.1f}%。")

        if strategy_nav_info and strategy_nav_info.get("max_drawdown"):
            mdd = strategy_nav_info["max_drawdown"]
            parts.append(f"历史最大回撤 {abs(mdd):.1f}%。")

        # 置信度提示
        if confidence == "low":
            parts.append(
                "（当前推荐基于问卷初估，上传交易数据后可获得更精准的匹配。）"
            )

        return " ".join(parts)

    def generate_short(
        self,
        strategy_id: str,
        similarity: float,
        top_similar_dims: list[dict],
    ) -> str:
        """生成简短话术（用于列表展示）"""
        sim_pct = similarity * 100
        similar_names = [d["feature"] for d in top_similar_dims[:2]]

        if sim_pct < 0:
            return f"策略 {strategy_id} 与您的风格方向相反（{sim_pct:.1f}%）"

        text = f"策略 {strategy_id} 与您的投资风格匹配度 {sim_pct:.1f}%。"
        if similar_names:
            text += f"最相似维度：{'、'.join(similar_names)}。"
        return text
