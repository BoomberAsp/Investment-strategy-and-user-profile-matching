"""
全局配置：路径、超参数、默认值
"""

from pathlib import Path

# === 路径 ===
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
PROFILES_DIR = DATA_DIR / "profiles"
TRADES_DIR = DATA_DIR / "trades"
QUESTIONNAIRES_DIR = DATA_DIR / "questionnaires"

# 策略数据路径（与 pipeline.py 一致）
STRATEGY_DATA_DIR = (
    PROJECT_ROOT / "净值_交易_资金及字段说明（相关性数据分析）" / "products_export_20260518_163122"
)

# === 超参数 ===
DEFAULT_BETA = 0.5
DEFAULT_LAMBDA = 1.0

# === EMA 更新参数 ===
EMA_DEFAULT_DECAY = 0.7    # 基础衰减因子 γ
EMA_DEFAULT_LR = 0.3       # 学习率（步幅）

# === 置信度阈值 ===
CONFIDENCE_MEDIUM_THRESHOLD = 3   # update_count >= 3 -> medium
CONFIDENCE_HIGH_THRESHOLD = 9     # update_count >= 9 -> high

# === 问卷 → 特征映射默认值 ===
# 未通过问卷覆盖的特征，使用策略总体均值作为默认值
# 这些值会在初始化时从策略数据中动态计算

# === β 粗估规则（L1 Q5）===
BETA_ROUGH_MAP = {
    "A": 0.7,   # 长期持有少动
    "B": 0.5,   # 均衡配置
    "C": 0.6,   # 积极调仓
    "D": 0.9,   # 高频短线
}

# === 滚动窗口选项 ===
ROLLING_WINDOW_OPTIONS = {
    "all": None,
    "120d": 120,
    "60d": 60,
    "30d": 30,
}

# === 行业列表（申万一级行业，28 个）===
SW_INDUSTRIES = [
    "煤炭", "石油石化", "基础化工", "钢铁", "有色金属",
    "电子", "家用电器", "食品饮料", "纺织服饰", "轻工制造",
    "美容护理", "医药生物", "公用事业", "交通运输", "房地产",
    "建筑材料", "建筑装饰", "商贸零售", "社会服务", "银行",
    "非银金融", "机械设备", "国防军工", "计算机", "传媒",
    "通信", "农林牧渔", "汽车",
]
