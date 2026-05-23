"""
问卷服务：问卷定义、评分、超参数推导
"""

from app.config import BETA_ROUGH_MAP, DEFAULT_BETA, SW_INDUSTRIES
from app.models.questionnaire import Question, Questionnaire


# ============================================================
# 问卷定义
# ============================================================

def _build_L1() -> Questionnaire:
    return Questionnaire(
        level="L1",
        title="基础风险评估",
        description="5 道基础题，约 2 分钟。必填，用于初始化您的投资画像。",
        estimated_minutes=2,
        questions=[
            Question(
                q_id="L1_Q1",
                text="您的可投资资金规模约为？",
                q_type="single_choice",
                options=["A. 5万以下", "B. 5-20万", "C. 20-50万", "D. 50-100万", "E. 100万以上"],
                target_params=["initial_capital"],
                weights={"A": 3, "B": 12, "C": 35, "D": 75, "E": 150},
            ),
            Question(
                q_id="L1_Q2",
                text="如果投资组合短期下跌 10%，您会？",
                q_type="single_choice",
                options=["A. 立即全部止损", "B. 观望等待", "C. 逢低加仓", "D. 无感，正常操作"],
                target_params=["risk_tolerance", "avg_loss_magnitude"],
            ),
            Question(
                q_id="L1_Q3",
                text="您期望的年化收益率是？",
                q_type="single_choice",
                options=["A. 5-10%", "B. 10-20%", "C. 20-40%", "D. 40%以上"],
                target_params=["risk_tolerance"],
            ),
            Question(
                q_id="L1_Q4",
                text="您平均多长时间进行一次交易？",
                q_type="single_choice",
                options=["A. 每天多次", "B. 每天一次", "C. 每周几次", "D. 每月几次", "E. 很少交易"],
                target_params=["holding_period", "turnover_rate"],
            ),
            Question(
                q_id="L1_Q5",
                text="您更倾向哪种投资方式？",
                q_type="single_choice",
                options=["A. 长期持有少动", "B. 均衡配置", "C. 积极调仓", "D. 高频短线"],
                target_params=["beta"],
                weights=BETA_ROUGH_MAP,
            ),
        ],
    )


def _build_L2() -> Questionnaire:
    return Questionnaire(
        level="L2",
        title="投资偏好深化",
        description="8 道进阶题，约 5 分钟。可选，可提升推荐准确度。",
        estimated_minutes=5,
        questions=[
            Question(
                q_id="L2_Q1",
                text="您偏好的投资品种？（可多选）",
                q_type="multi_select",
                options=["A. ETF基金", "B. 蓝筹股", "C. 小盘股", "D. 创业板/科创板"],
                target_params=["etf_ratio", "avg_price_preference"],
            ),
            Question(
                q_id="L2_Q2",
                text="您通常持仓占您总资金的比例？",
                q_type="slider",
                options=[],
                target_params=["position_uniformity"],
            ),
            Question(
                q_id="L2_Q3",
                text="您对单只股票的亏损容忍度？",
                q_type="slider",
                options=[],
                target_params=["avg_loss_magnitude"],
            ),
            Question(
                q_id="L2_Q4",
                text="您更倾向于？",
                q_type="single_choice",
                options=["A. 抄底（跌了买）", "B. 追涨（涨了买）", "C. 无所谓"],
                target_params=["trend_preference"],
            ),
            Question(
                q_id="L2_Q5",
                text="您买入时通常？",
                q_type="single_choice",
                options=["A. 一次性重仓", "B. 分批建仓", "C. 小额试探"],
                target_params=["buy_sell_ratio", "position_uniformity"],
            ),
            Question(
                q_id="L2_Q6",
                text="您过去交易的胜率大约？",
                q_type="single_choice",
                options=["A. 40%以下", "B. 40-50%", "C. 50-60%", "D. 60-70%", "E. 70%以上"],
                target_params=["positive_trade_ratio"],
            ),
            Question(
                q_id="L2_Q7",
                text="您能接受的最大回撤？",
                q_type="single_choice",
                options=["A. 5%以下", "B. 5-10%", "C. 10-20%", "D. 20%以上"],
                target_params=["risk_tolerance"],
            ),
            Question(
                q_id="L2_Q8",
                text="您的投资经验年限？",
                q_type="single_choice",
                options=["A. 1年以内", "B. 1-3年", "C. 3-5年", "D. 5年以上"],
                target_params=["beta"],
            ),
        ],
    )


def _build_L3() -> Questionnaire:
    return Questionnaire(
        level="L3",
        title="交易行为进阶",
        description="10 道深入题，约 8 分钟。可选，适合有经验的投资者。",
        estimated_minutes=8,
        questions=[
            Question(
                q_id="L3_Q1",
                text="您通常同时持有几只股票？",
                q_type="number_input",
                options=[],
                target_params=["hhi_concentration"],
            ),
            Question(
                q_id="L3_Q2",
                text="您亏损时通常？",
                q_type="single_choice",
                options=["A. 快速止损", "B. 等待反弹", "C. 躺平不管"],
                target_params=["disposition_effect"],
            ),
            Question(
                q_id="L3_Q3",
                text="您盈利时通常？",
                q_type="single_choice",
                options=["A. 尽快落袋为安", "B. 让利润奔跑", "C. 分批止盈"],
                target_params=["disposition_effect", "holding_period"],
            ),
            Question(
                q_id="L3_Q4",
                text="您是否会交易 ETF？",
                q_type="single_choice",
                options=["A. 经常", "B. 偶尔", "C. 从不"],
                target_params=["etf_ratio"],
            ),
            Question(
                q_id="L3_Q5",
                text="您偏好的股价区间？",
                q_type="single_choice",
                options=["A. 10元以下", "B. 10-30元", "C. 30-100元", "D. 100元以上"],
                target_params=["avg_price_preference"],
            ),
            Question(
                q_id="L3_Q6",
                text="您的交易时间分布？",
                q_type="single_choice",
                options=["A. 集中在少数几天", "B. 均匀分布", "C. 随机"],
                target_params=["position_uniformity", "turnover_rate"],
            ),
            Question(
                q_id="L3_Q7",
                text="您对哪些行业板块有偏好？（可多选）",
                q_type="multi_select",
                options=SW_INDUSTRIES,
                target_params=["industry_vector"],
            ),
            Question(
                q_id="L3_Q8",
                text="您是否使用过量化策略？",
                q_type="single_choice",
                options=["A. 是，自己编写", "B. 是，跟投他人", "C. 没有"],
                target_params=["beta"],
            ),
            Question(
                q_id="L3_Q9",
                text="您认为自己的投资风格更接近？",
                q_type="single_choice",
                options=["A. 价值型", "B. 成长型", "C. 技术型", "D. 事件驱动"],
                target_params=["beta", "holding_period"],
            ),
            Question(
                q_id="L3_Q10",
                text="您是否有未平仓的长期持仓？如有，大约持有了多久？",
                q_type="number_input",
                options=[],
                target_params=["holding_period"],
            ),
        ],
    )


# ============================================================
# 问卷评分引擎
# ============================================================

# L1 问题答案到特征的粗估映射
L1_FEATURE_MAP = {
    "L1_Q4": {
        "A": {"holding_period": 3, "turnover_rate": 4.0},
        "B": {"holding_period": 8, "turnover_rate": 1.5},
        "C": {"holding_period": 15, "turnover_rate": 0.5},
        "D": {"holding_period": 40, "turnover_rate": 0.15},
        "E": {"holding_period": 90, "turnover_rate": 0.05},
    },
    "L1_Q2": {
        "A": {"risk_tolerance": 1, "avg_loss_magnitude": 0.03},
        "B": {"risk_tolerance": 2, "avg_loss_magnitude": 0.05},
        "C": {"risk_tolerance": 4, "avg_loss_magnitude": 0.10},
        "D": {"risk_tolerance": 5, "avg_loss_magnitude": 0.15},
    },
    "L1_Q3": {
        "A": {"risk_tolerance": 1},
        "B": {"risk_tolerance": 2},
        "C": {"risk_tolerance": 3},
        "D": {"risk_tolerance": 5},
    },
}

# L2 问题答案到特征/参数的细化映射
L2_FEATURE_MAP = {
    "L2_Q1": {
        "A": {"etf_ratio": 0.6},
        "B": {"etf_ratio": 0.1},
        "C": {"etf_ratio": 0.05},
        "D": {"etf_ratio": 0.0},
    },
    "L2_Q4": {
        "A": {"trend_preference": -0.2},
        "B": {"trend_preference": 0.2},
        "C": {"trend_preference": 0.0},
    },
    "L2_Q5": {
        "A": {"buy_sell_ratio": 1.5, "position_uniformity": 0.3},
        "B": {"buy_sell_ratio": 1.0, "position_uniformity": 0.6},
        "C": {"buy_sell_ratio": 0.8, "position_uniformity": 0.7},
    },
    "L2_Q6": {
        "A": {"positive_trade_ratio": 0.35},
        "B": {"positive_trade_ratio": 0.45},
        "C": {"positive_trade_ratio": 0.55},
        "D": {"positive_trade_ratio": 0.65},
        "E": {"positive_trade_ratio": 0.80},
    },
    "L2_Q7": {
        "A": {"risk_tolerance": 1},
        "B": {"risk_tolerance": 2},
        "C": {"risk_tolerance": 3},
        "D": {"risk_tolerance": 5},
    },
    "L2_Q8": {
        "A": {"beta": 0.0},    # 经验少，更依赖产品推荐
        "B": {"beta": 0.05},
        "C": {"beta": 0.1},
        "D": {"beta": 0.15},
    },
}

# L3 问题答案到特征的细化映射
L3_FEATURE_MAP = {
    "L3_Q2": {
        "A": {"disposition_effect": 2.0},
        "B": {"disposition_effect": 0.8},
        "C": {"disposition_effect": 0.3},
    },
    "L3_Q3": {
        "A": {"disposition_effect": 1.5, "holding_period": -5},
        "B": {"disposition_effect": 0.5, "holding_period": 10},
        "C": {"disposition_effect": 1.0, "holding_period": 5},
    },
    "L3_Q4": {
        "A": {"etf_ratio": 0.5},
        "B": {"etf_ratio": 0.2},
        "C": {"etf_ratio": 0.0},
    },
    "L3_Q5": {
        "A": {"avg_price_preference": 7},
        "B": {"avg_price_preference": 20},
        "C": {"avg_price_preference": 60},
        "D": {"avg_price_preference": 150},
    },
    "L3_Q6": {
        "A": {"position_uniformity": 0.3, "turnover_rate": 0.5},
        "B": {"position_uniformity": 0.7, "turnover_rate": 0.0},
        "C": {"position_uniformity": 0.5, "turnover_rate": 0.2},
    },
    "L3_Q8": {
        "A": {"beta": 0.15},
        "B": {"beta": 0.1},
        "C": {"beta": -0.05},
    },
    "L3_Q9": {
        "A": {"holding_period": 20, "beta": 0.05},
        "B": {"holding_period": 10, "beta": 0.05},
        "C": {"holding_period": -5, "beta": 0.1},
        "D": {"holding_period": -3, "beta": 0.1},
    },
}


class QuestionnaireService:
    """问卷生成、评分、超参数推导"""

    def __init__(self, strategy_mean_features: dict[str, float] | None = None):
        """
        strategy_mean_features: 策略各特征的均值，用作问卷未覆盖特征的默认值。
        """
        self.strategy_mean = strategy_mean_features or {}
        self._questionnaires = {
            "L1": _build_L1(),
            "L2": _build_L2(),
            "L3": _build_L3(),
        }

    def get_questionnaire(self, level: str) -> Questionnaire:
        return self._questionnaires[level]

    def get_all_questionnaires(self) -> list[Questionnaire]:
        return [self._questionnaires["L1"], self._questionnaires["L2"], self._questionnaires["L3"]]

    def score_answers(self, level: str, answers: dict) -> dict:
        """
        评分问卷回答

        参数 answers: {"L1_Q1": "A", "L1_Q2": "B", ...}
            对于 slider 类型: {"L2_Q2": 55}（0-100 的值）
            对于 number_input: {"L3_Q1": 5}（数值）
            对于 multi_select: {"L2_Q1": ["A", "B"]}（选中的选项列表）

        返回: {
            "beta": float,
            "risk_tolerance": int,
            "initial_capital": float,
            "features": dict[str, float],
            "industry_vector": dict[str, float],
        }
        """
        result = {
            "beta": DEFAULT_BETA,
            "risk_tolerance": 3,
            "initial_capital": 0.0,
            "features": dict(self.strategy_mean),
            "industry_vector": {},
        }

        # 选择对应的特征映射表
        feature_maps = {
            "L1": L1_FEATURE_MAP,
            "L2": L2_FEATURE_MAP,
            "L3": L3_FEATURE_MAP,
        }
        fmap = feature_maps.get(level, {})

        for q_id, answer in answers.items():
            if q_id not in fmap:
                continue
            question_map = fmap[q_id]

            # 处理选项前缀（提取第一个字母）
            if isinstance(answer, str):
                choice = answer[0].upper()
            elif isinstance(answer, list):
                # multi_select: 对每个选中选项的权重取平均
                choice = None
                for key, val in question_map.items():
                    selected_vals = []
                    for opt in answer:
                        opt_key = opt[0].upper() if isinstance(opt, str) else opt
                        if opt_key in val:
                            selected_vals.append(val[opt_key])
                    if selected_vals:
                        avg_val = {}
                        for d in selected_vals:
                            for k, v in d.items():
                                avg_val[k] = avg_val.get(k, 0.0) + v
                        for k in avg_val:
                            avg_val[k] /= len(selected_vals)
                        self._merge_into_result(result, avg_val)
                continue
            else:
                # slider 或 number_input: 使用数值映射
                self._handle_numeric_answer(q_id, answer, result)
                continue

            if choice in question_map:
                self._merge_into_result(result, question_map[choice])

        # 处理 beta 粗估（L1 Q5 专用）
        if level == "L1" and "L1_Q5" in answers:
            choice = answers["L1_Q5"][0].upper() if isinstance(answers["L1_Q5"], str) else "B"
            result["beta"] = BETA_ROUGH_MAP.get(choice, DEFAULT_BETA)

        # 处理 initial_capital（L1 Q1 专用）
        if level == "L1" and "L1_Q1" in answers:
            choice = answers["L1_Q1"][0].upper() if isinstance(answers["L1_Q1"], str) else "B"
            result["initial_capital"] = {
                "A": 3, "B": 12, "C": 35, "D": 75, "E": 150,
            }.get(choice, 0.0)

        # 处理 risk_tolerance（取最高值）
        if level == "L1":
            rt_from_q2 = None
            if "L1_Q2" in answers:
                c = answers["L1_Q2"][0].upper()
                rt_from_q2 = {"A": 1, "B": 2, "C": 4, "D": 5}.get(c)
            rt_from_q3 = None
            if "L1_Q3" in answers:
                c = answers["L1_Q3"][0].upper()
                rt_from_q3 = {"A": 1, "B": 2, "C": 3, "D": 5}.get(c)
            if rt_from_q2 is not None and rt_from_q3 is not None:
                result["risk_tolerance"] = max(rt_from_q2, rt_from_q3)
            elif rt_from_q2 is not None:
                result["risk_tolerance"] = rt_from_q2
            elif rt_from_q3 is not None:
                result["risk_tolerance"] = rt_from_q3

        return result

    def _merge_into_result(self, result: dict, values: dict):
        """将映射值合并到结果中"""
        for key, val in values.items():
            if key == "beta":
                result["beta"] = max(0.0, min(1.0, result["beta"] + val))
            elif key == "risk_tolerance":
                result["risk_tolerance"] = max(1, min(5, val))
            elif key == "initial_capital":
                result["initial_capital"] = val
            else:
                result["features"][key] = val

    def _handle_numeric_answer(self, q_id: str, value: float, result: dict):
        """处理滑块/数字输入类型"""
        if q_id == "L2_Q2":
            # 持仓比例 -> position_uniformity
            result["features"]["position_uniformity"] = min(1.0, value / 100.0)
        elif q_id == "L2_Q3":
            # 亏损容忍度 -> avg_loss_magnitude
            result["features"]["avg_loss_magnitude"] = value / 100.0
        elif q_id == "L3_Q1":
            # 持股数量 -> hhi_concentration (越少越集中)
            n = max(1, int(value))
            result["features"]["hhi_concentration"] = 1.0 / n
        elif q_id == "L3_Q10":
            # 长期持仓天数
            result["features"]["holding_period"] = max(1, int(value))
