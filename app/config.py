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

# 源数据目录（统一放在 stats_data/ 下）
STATS_DATA_DIR = PROJECT_ROOT / "stats_data"

# 策略数据路径（与 pipeline.py 一致）
STRATEGY_DATA_DIR = (
    STATS_DATA_DIR / "净值_交易_资金及字段说明（相关性数据分析）" / "products_export_20260518_163122"
)

# 绩效 Excel 文件（DLMethod 用）
PERF_FILE_1 = STATS_DATA_DIR / "量化策略绩效-1.xlsx"
PERF_FILE_2 = STATS_DATA_DIR / "量化策略绩效-2.xlsx"

# 模拟账户记录
USER_FILES = {
    "A": STATS_DATA_DIR / "模拟账户A的记录.xlsx",
    "B": STATS_DATA_DIR / "模拟账户B的记录.xlsx",
    "C": STATS_DATA_DIR / "模拟账户C的记录.xlsx",
}

# 补充 CSV 策略
CSV_STRATEGY_FILES = [
    STATS_DATA_DIR / "带ETF的策略1.csv",
    STATS_DATA_DIR / "带ETF策略2.csv",
    STATS_DATA_DIR / "朝花夕拾策略.csv",
]

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

# === DLMethod 机器学习输出路径 ===
DLMETHOD_DIR = PROJECT_ROOT / "DLMethod"
LSTM_SIMILARITY_MATRIX = DLMETHOD_DIR / "matching_phase2_lstm.csv"
LSTM_RECOMMENDATIONS = DLMETHOD_DIR / "final_recommendations.csv"
LSTM_SHAP_ANALYSIS = DLMETHOD_DIR / "shap_analysis.json"
LSTM_EMBEDDING_META = DLMETHOD_DIR / "embedding_meta.json"

# === 融合权重 ===
FUSION_ALPHA = 0.7  # 统计方法权重，1-alpha = LSTM 权重
FUSION_ALPHA_OPTIONS = {
    "稳健展示版": 0.8,
    "平衡实验版": 0.7,
    "强调序列风格版": 0.6,
}
