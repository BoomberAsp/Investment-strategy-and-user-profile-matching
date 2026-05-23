# 策略推荐网页应用：设计文档

**版本**：V3.0
**日期**：2026-05-24
**框架**：Streamlit
**状态**：执行中

---

## 一、产品定位

本项目是理论 pipeline（`pipeline.py`）与终端用户之间的**产品中转账页**。核心价值：

1. 让用户通过注册登录进入系统
2. 通过问卷初始化用户画像并设定超参数 β
3. 基于交易数据动态更新用户画像（EMA 动量式衰减）
4. 展示个性化策略推荐与可解释归因
5. 支持多算法后端切换与融合（统计 PCA 路线 + LSTM 深度学习路线 + 融合加权路线）

---

## 二、目录结构

```
Investment-strategy-and-user-profile-matching/
├── pipeline.py                          # 理论流水线（不修改）
├── generate_simulated_data.py           # 模拟数据生成
├── app/
│   ├── __init__.py
│   ├── models/                          # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py                      # User, UserProfile 对象定义
│   │   └── questionnaire.py             # Questionnaire, Question, Answer 对象定义
│   ├── services/                        # 业务服务层
│   │   ├── __init__.py
│   │   ├── auth.py                      # 注册/登录/会话管理
│   │   ├── storage.py                   # JSON 文件存储引擎
│   │   ├── questionnaire.py             # 问卷生成、评分、β 推导
│   │   ├── profile.py                   # 画像初始化、EMA 更新、衰减
│   │   ├── feature_extractor.py         # 从 pipeline 中抽象出的特征提取方法
│   │   ├── matching_backend.py          # 统一匹配引擎接口（抽象基类）
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── statistical.py           # PCA + 径向惩罚余弦（当前实现）
│   │   │   ├── lstm.py                  # DLMethod: LSTM 128 维序列风格匹配
│   │   │   └── fusion.py                # 统计 + LSTM 加权融合（α=0.7/0.3）
│   │   ├── recommendation.py            # 推荐调度器（调用 MatchingBackend）
│   │   └── popup_generator.py           # 客户端弹窗话术生成器
│   ├── data/                            # 本地持久化数据
│   │   ├── users.json                   # 用户账号信息
│   │   ├── profiles/                    # 用户画像（每人一个文件）
│   │   │   └── {user_id}.json
│   │   ├── trades/                      # 用户上传的交易记录
│   │   │   └── {user_id}.csv
│   │   └── questionnaires/              # 问卷回答存档
│   │       └── {user_id}_{level}.json
│   └── config.py                        # 全局配置（路径、超参数等）
├── app.py                               # Streamlit 主入口
├── report.md                            # 数据分析报告
├── README.md                            # 项目说明
└── requirements.txt                     # 新增: streamlit 等依赖
```

---

## 三、数据模型设计

### 3.1 User（用户）

```python
@dataclass
class User:
    user_id: str                    # 唯一标识，如 "U_001"
    username: str                   # 登录用户名
    password_hash: str              # SHA-256 哈希（不存明文）
    created_at: str                 # ISO 8601 时间戳
    last_login: str                 # ISO 8601 时间戳
    onboarding_status: str          # "new" | "questionnaire_done" | "active"
```

### 3.2 UserProfile（用户画像）

```python
@dataclass
class UserProfile:
    user_id: str

    # === 超参数 ===
    beta: float                     # 行为特征 vs 非行为特征权重, [0, 1]
    risk_tolerance: int             # 1-5，来自问卷
    initial_capital: float          # 期初资金估计（问卷填写）

    # === 12 维特征向量（当前画像） ===
    features: dict[str, float]      # MATCH_FEATURES -> value

    # === 行业分布向量（资产偏好层的扩展） ===
    # 与 pipeline 中 etf_ratio 一脉相承：
    #   - 当前 pipeline 用 etf_ratio 代理行业偏好（ETF 占比）
    #   - 这里预留完整的 28 维申万行业分布向量，未来接入个股行业分类后启用
    #   - 当前默认值为均匀分布或基于 etf_ratio 的粗估
    industry_vector: dict[str, float]  # {"煤炭": 0.05, "电子": 0.12, ...}

    # === EMA 动量更新参数 ===
    feature_momentum: dict[str, float]   # 累计梯度（动量项）
    decay_factor: float             # 衰减系数 γ，默认 0.7
    update_count: int               # 已更新次数，用于自适应衰减

    # === 置信度（冷启动机制） ===
    # 类比项目计划书中"贝叶斯收缩"思想：
    #   - low: 仅问卷初始化，特征值可信度低
    #   - medium: 1-5 次交易数据更新，特征值逐步可靠
    #   - high: 6+ 次交易数据更新，特征值稳定
    confidence_level: str           # "low" | "medium" | "high"

    # === 元数据 ===
    source: str                     # "questionnaire" | "trade_data" | "hybrid"
    last_updated: str               # ISO 8601 时间戳
    questionnaire_scores: dict      # 各问卷原始得分（追溯用）
    matching_backend: str           # 当前使用的匹配后端: "statistical" | "lstm" | "fusion"

    def ema_update(self, new_features: dict[str, float], lr: float = 0.3):
        """
        EMA 动量式画像更新（类比 SGD with Momentum）

        类比公式:
            g = new_features - old_features          # "梯度"
            m = γ * m + (1-γ) * g                    # 动量累计
            features = features + lr * m             # 更新

        其中衰减因子 γ 随 update_count 自适应调整:
            γ = base_decay * (1 - 1 / (update_count + 2))
        早期更新 γ 较小（更信任新数据），后期 γ 趋近 base_decay（画像更稳定）
        """
        ...

    @property
    def is_confident(self) -> bool:
        """画像是否可信（medium 或 high）"""
        return self.confidence_level in ("medium", "high")
```

### 3.3 Questionnaire（问卷）

```python
@dataclass
class Question:
    q_id: str
    text: str                     # 问题文本
    q_type: str                   # "single_choice" | "slider" | "number_input" | "multi_select"
    options: list[str]            # 选项（单选/多选需要）
    target_params: list[str]      # 影响的超参数列表，如 ["beta", "risk_tolerance"]
    weights: dict                 # 各选项对 target 的映射权重

@dataclass
class Questionnaire:
    level: str                    # "L1" | "L2" | "L3"
    title: str
    description: str
    questions: list[Question]
    estimated_minutes: int
```

### 3.4 MatchingResult（匹配结果）

```python
@dataclass
class MatchingResult:
    user_id: str
    backend: str                  # "statistical" | "lstm" | "fusion"
    timestamp: str
    top_n: list[dict]             # [{"strategy": str, "similarity": float, "rank": int}, ...]
    explanation: dict             # 可解释归因
    popup_text: str               # 客户端弹窗话术
    confidence: str               # 与 UserProfile.confidence_level 一致
    # LSTM/Fusion 特有字段
    phase1_rank: dict[str, int] | None = None   # 特征基线排名
    phase2_rank: dict[str, int] | None = None   # LSTM 排名
    stat_score: dict[str, float] | None = None   # 统计归一化得分
    ml_score: dict[str, float] | None = None     # LSTM 归一化得分
```

---

## 四、匹配引擎抽象层（MatchingBackend）

### 4.1 抽象基类

所有匹配算法（PCA 统计、Word2Vec、未来 Transformer）统一实现此接口：

```python
from abc import ABC, abstractmethod

class MatchingBackend(ABC):
    """所有匹配算法的统一接口"""

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

        参数 industry_vector: 用户行业分布向量，后端可选择性地融合 JS 散度结果。
        """
        ...

    @abstractmethod
    def get_all_metrics(
        self, user_features: dict[str, float], beta: float = 0.5,
    ) -> dict:
        """返回所有可用度量下的推荐结果（用于可视化对比）"""
        ...
```

### 4.2 StatisticalBackend（当前实现）

```python
class StatisticalBackend(MatchingBackend):
    """
    PCA + 径向惩罚余弦相似度（当前主后端）

    直接从 pipeline.py 导入函数，不修改原代码：
      - apply_beta_weighting, build_feature_matrix, apply_pca
      - compute_radial_penalty_cosine, generate_explanation
    """

    def name(self) -> str:
        return "statistical"

    def fit(self, strategy_features, strategy_nav=None):
        """
        1. 从 pipeline 导入 PCA 相关函数
        2. 在策略特征上预计算 PCA 模型
        3. 缓存策略 PCA 坐标，避免重复计算
        """
        ...

    def predict(self, user_features, beta=0.5, top_n=3, industry_vector=None):
        """
        1. 对用户特征应用 beta 加权
        2. 投影到预计算的 PCA 空间
        3. 计算径向惩罚余弦相似度
        4. （可选）融合行业 JS 散度（若 industry_vector 不为空）
        5. 排序 + 生成可解释输出
        """
        ...

    def get_all_metrics(self, user_features, beta=0.5):
        """返回径向惩罚余弦、纯余弦、欧式距离三种度量结果"""
        ...
```

### 4.3 LSTMBackend（DLMethod 团队完成）

```python
class LSTMBackend(MatchingBackend):
    """
    LSTM 128 维序列风格匹配（DLMethod 团队输出）

    数据源：DLMethod/ 目录下预训练好的 LSTM 编码器输出：
      - matching_phase2_lstm.csv  → 账户 × 策略 LSTM 相似度矩阵
      - final_recommendations.csv → Top-N 推荐长表（含 Phase1/Phase2 排名）

    本质上是"查表式"匹配：用户上传交易数据后，用账户名映射
    直接从预计算的相似度矩阵中读取该账户的推荐结果。
    不依赖用户的 12 维特征向量，而是依赖交易序列的 token 风格匹配。
    """

    def name(self) -> str:
        return "lstm"

    def fit(self, strategy_features, strategy_nav=None):
        """
        1. 加载 matching_phase2_lstm.csv → 相似度矩阵 (Account × Strategy)
        2. 加载 final_recommendations.csv → 推荐长表
        3. 缓存供 predict() 查表使用
        """
        ...

    def predict(self, user_features, beta=0.5, top_n=3, industry_vector=None):
        """
        1. 将当前用户映射到 Account_A/B/C（或动态生成的账户名）
        2. 从相似度矩阵中取出该账户的向量
        3. 排序取 Top-N
        4. 返回结果（不含 PCA 归因，但含 Phase1/Phase2 排名对比）
        """
        ...
```

### 4.4 FusionBackend（统计 + LSTM 加权融合）

```python
class FusionBackend(MatchingBackend):
    """
    统计方法 + LSTM 辅助加权融合

    融合公式（来自 ML_INTEGRATION_GUIDE.md）：
        final_score = α * stat_norm + (1-α) * ml_norm
    默认 α = 0.7（统计方法为主，LSTM 为序列风格辅助项）

    步骤：
    1. 调用 StatisticalBackend.predict() 得到统计得分矩阵
    2. 调用 LSTMBackend.predict() 得到 LSTM 得分矩阵
    3. 对齐共同的账户和策略
    4. Min-Max 归一化到 0-1（避免量纲不同）
    5. 加权融合 → 输出最终排名
    """

    def name(self) -> str:
        return "fusion"

    def fit(self, strategy_features, strategy_nav=None):
        """同时 fit StatisticalBackend 和 LSTMBackend"""
        ...

    def predict(self, user_features, beta=0.5, top_n=3, industry_vector=None):
        """
        1. 分别获取统计得分和 LSTM 得分
        2. Min-Max 归一化
        3. 加权融合
        4. 输出 Top-N + 双源归因
        """
        ...
```

### 4.5 后端注册与切换

### 4.6 后端注册与切换

```python
class BackendRegistry:
    """匹配后端注册表，支持运行时切换"""

    def __init__(self):
        self._backends: dict[str, MatchingBackend] = {}

    def register(self, backend: MatchingBackend):
        self._backends[backend.name()] = backend

    def get(self, name: str) -> MatchingBackend:
        return self._backends.get(name)

    def list_available(self) -> list[str]:
        return list(self._backends.keys())

    def list_active(self) -> list[str]:
        """返回已 fit 过的后端"""
        return [name for name, b in self._backends.items() if getattr(b, "_is_fitted", False)]
```

### 4.7 三后端定位与使用场景

| 后端 | 定位 | 何时使用 | 推荐场景 |
|------|------|---------|---------|
| `statistical` | **主线**：12 维特征 + PCA + 径向惩罚余弦 | 用户画像完整、问卷填写充分 | 稳健展示、可解释性优先 |
| `lstm` | **辅线**：交易序列风格相似度 | 用户上传了交易数据、想看到"序列风格"视角的推荐 | 候选策略排序、与统计方法互相验证 |
| `fusion` | **推荐默认**：α=0.7 stat + 0.3 LSTM | 用户既有问卷画像又有交易数据 | 论文/汇报默认、综合推荐 |

**融合权重可配置**（`config.py` 中 `FUSION_ALPHA`）：

| 场景 | 统计方法权重 α | LSTM 权重 1-α |
|------|---:|---:|
| 稳健展示版 | 0.8 | 0.2 |
| 平衡实验版（默认） | 0.7 | 0.3 |
| 强调序列风格版 | 0.6 | 0.4 |

---

## 五、问卷体系设计

### 5.1 三级问卷架构

| 级别 | 名称 | 题目数 | 预估时间 | 触发条件 | 产出 |
|------|------|--------|---------|---------|------|
| **L1** | 基础风险评估 | 5 题 | 2 分钟 | 注册后必填 | `risk_tolerance`, `initial_capital`, β 粗估, 特征粗估 |
| **L2** | 投资偏好深化 | 8 题 | 5 分钟 | 可选，推荐填写 | β 精调, 资产偏好特征细化, 行业向量粗估 |
| **L3** | 交易行为进阶 | 10 题 | 8 分钟 | 可选，高级用户 | 完整 12 维特征初始化, β 精确设定 |

**用户可以选择仅填 L1 快速开始，后续随时补充更深问卷。**

### 5.2 Level 1：基础风险评估（必填）

| # | 问题 | 类型 | 选项/范围 | 映射目标 |
|---|------|------|---------|---------|
| Q1 | 您的可投资资金规模约为？ | 单选 | A. <5万 B. 5-20万 C. 20-50万 D. 50-100万 E. >100万 | `initial_capital` |
| Q2 | 如果投资组合短期下跌 10%，您会？ | 单选 | A. 立即止损 B. 观望 C. 加仓 D. 无感 | `risk_tolerance`, `avg_loss_magnitude` |
| Q3 | 您期望的年化收益率是？ | 单选 | A. 5-10% B. 10-20% C. 20-40% D. 40%+ | `risk_tolerance` |
| Q4 | 您平均多长时间进行一次交易？ | 单选 | A. 每天多次 B. 每天一次 C. 每周几次 D. 每月几次 E. 很少 | `holding_period` (粗估), `turnover_rate` (粗估) |
| Q5 | 您更倾向哪种投资方式？ | 单选 | A. 长期持有少动 B. 均衡配置 C. 积极调仓 D. 高频短线 | `beta` 粗估, `holding_period` |

**β 粗估规则**：

| Q5 答案 | β 值 | 理由 |
|---------|------|------|
| A. 长期持有 | 0.7 | 行为特征主导（持仓周期、换手率等） |
| B. 均衡配置 | 0.5 | 行为与偏好并重 |
| C. 积极调仓 | 0.6 | 行为特征略主导 |
| D. 高频短线 | 0.9 | 极度依赖行为特征 |

### 5.3 Level 2：投资偏好深化（可选）

| # | 问题 | 类型 | 映射目标 |
|---|------|------|---------|
| Q1 | 您偏好的投资品种？ | 多选 | `etf_ratio`, `avg_price_preference`, `industry_vector` (粗估) |
| Q2 | 您通常持仓占您总资金的比例？ | 滑块 (0-100%) | `position_uniformity` (间接) |
| Q3 | 您对单只股票的亏损容忍度？ | 滑块 (1-20%) | `avg_loss_magnitude` |
| Q4 | 您更倾向于？ | 单选 | `trend_preference` |
| Q5 | 您买入时通常？ | 单选 | `buy_sell_ratio`, `position_uniformity` |
| Q6 | 您过去交易的胜率大约？ | 单选 | `positive_trade_ratio` |
| Q7 | 您能接受的最大回撤？ | 单选 | `risk_tolerance` 更新, β 微调 |
| Q8 | 您投资经验年限？ | 单选 | β 微调 |

### 5.4 Level 3：交易行为进阶（可选）

| # | 问题 | 类型 | 映射目标 |
|---|------|------|---------|
| Q1 | 您通常同时持有几只股票？ | 数字输入 | `hhi_concentration` |
| Q2 | 您亏损时通常？ | 单选 | `disposition_effect` |
| Q3 | 您盈利时通常？ | 单选 | `disposition_effect`, `holding_period` |
| Q4 | 您是否会交易 ETF？ | 单选 | `etf_ratio` 精确化 |
| Q5 | 您偏好的股价区间？ | 单选 | `avg_price_preference` |
| Q6 | 您的交易时间分布？ | 单选 | `position_uniformity`, `turnover_rate` |
| Q7 | 您对行业板块有偏好吗？ | 多选（申万一级行业） | `industry_vector` (28 维) |
| Q8 | 您是否使用过量化策略？ | 单选 | β 微调 |
| Q9 | 您认为自己的投资风格更接近？ | 单选 | 综合微调多项 features |
| Q10 | 您是否有未平仓的长期持仓？ | 数字输入+日期 | 辅助持仓周期估算 |

### 5.5 问卷评分 → 特征映射引擎

问卷回答通过 `QuestionnaireService` 解析为 `dict[str, float]`。映射方式：**规则映射表**（基于金融常识 + 策略统计均值）。

未通过问卷覆盖的特征，使用**策略总体均值**作为默认值。

---

## 六、画像更新：EMA 动量算法

### 6.1 核心思想

类比 SGD with Momentum：问卷初始化是"初始权重"，每次上传交易数据产生"梯度"，用动量累计机制平滑多次更新。

### 6.2 更新公式

```
第 t 次更新时：

1. 计算"梯度"：
   g = new_features - old_features

2. 自适应衰减因子：
   γ_t = γ_base × (1 - 1/(t + 2))
   # t=0: γ = 0.7 × (1 - 1/2) = 0.35  (新数据权重高)
   # t=5: γ = 0.7 × (1 - 1/7) = 0.60  (逐渐稳定)
   # t→∞: γ → 0.7                      (稳定状态)

3. 动量累计：
   m = γ_t × m_old + (1 - γ_t) × g

4. 更新画像：
   features = features + lr × m
   # lr (学习率) 默认 0.3，控制每次更新的步幅

5. 更新置信度：
   if update_count <= 2:  confidence = "low"
   elif update_count <= 8: confidence = "medium"
   else: confidence = "high"
```

### 6.3 直观解释

| 阶段 | 更新次数 | 衰减因子 | 置信度 | 行为 |
|------|---------|---------|--------|------|
| 冷启动 | 0-2 次 | 0.35-0.47 | low | 画像快速向真实交易行为靠拢 |
| 收敛期 | 3-10 次 | 0.50-0.64 | medium | 画像逐步稳定，波动减小 |
| 稳定期 | 10+ 次 | 0.65-0.70 | high | 画像基本稳定，仅微调 |

---

## 七、服务层设计

### 7.1 StorageService（JSON 文件存储）

```python
class StorageService:
    """JSON 文件存储引擎，线程安全（通过 filelock）"""

    def __init__(self, data_dir: Path):
        ...

    # 用户 CRUD
    def save_user(self, user: User) -> bool
    def get_user(self, username: str) -> User | None
    def get_user_by_id(self, user_id: str) -> User | None

    # 画像 CRUD
    def save_profile(self, profile: UserProfile) -> bool
    def get_profile(self, user_id: str) -> UserProfile | None

    # 交易数据
    def save_trades(self, user_id: str, trades_df: pd.DataFrame) -> bool
    def load_trades(self, user_id: str) -> pd.DataFrame | None
    def list_trade_uploads(self, user_id: str) -> list[dict]  # 上传历史

    # 问卷存档
    def save_questionnaire_results(self, user_id: str, level: str, answers: dict) -> bool
    def load_questionnaire_results(self, user_id: str, level: str) -> dict | None
    def list_completed_levels(self, user_id: str) -> list[str]
```

### 7.2 AuthService（注册/登录）

```python
class AuthService:
    def register(self, username: str, password: str) -> tuple[bool, str, User | None]
    def login(self, username: str, password: str) -> tuple[bool, str, User | None]
    def update_user(self, user: User) -> bool
```

### 7.3 QuestionnaireService（问卷）

```python
class QuestionnaireService:
    """问卷生成、评分、超参数推导"""

    def __init__(self, strategy_mean_features: dict[str, float]):
        self.strategy_mean = strategy_mean_features

    def get_questionnaire(self, level: str) -> Questionnaire
    def score_answers(self, level: str, answers: dict) -> dict:
        """
        评分问卷回答
        返回: {
            "beta": float,
            "risk_tolerance": int,
            "initial_capital": float,
            "features": dict[str, float],       # 12 维特征估计
            "industry_vector": dict[str, float], # 28 维行业向量（L2/L3 可用）
        }
        """
```

### 7.4 ProfileService（画像管理 + EMA 更新）

```python
class ProfileService:
    """用户画像管理：初始化、EMA 更新、查询"""

    def __init__(self, storage: StorageService):
        self.default_decay = 0.7
        self.default_lr = 0.3

    def create_profile_from_questionnaire(
        self, user_id: str, score_result: dict
    ) -> UserProfile:
        """从问卷评分结果创建初始画像"""

    def update_profile_with_trades(
        self, user_id: str, trades_df: pd.DataFrame,
        strategy_features: dict, strategy_nav: dict,
        window_days: int | None = None,
    ) -> UserProfile:
        """
        用交易数据更新画像（EMA 动量方式）

        参数 window_days: 滚动窗口天数。若为 None 使用全量数据；
            否则仅用最近 window_days 天的交易（用于匹配稳定性分析）。
        """

    def get_profile(self, user_id: str) -> UserProfile | None
    def get_confidence(self, user_id: str) -> str:
        """获取画像置信度"""
```

### 7.5 FeatureExtractor（特征提取）

```python
class FeatureExtractor:
    """
    从 pipeline.py 中抽象出的特征提取方法。
    不修改 pipeline.py，而是导入后重新组合。
    """

    def extract_user_features(self, trades_df: pd.DataFrame) -> dict[str, float]:
        """从用户交易 DataFrame 提取 12 维特征"""
        ...

    def extract_strategy_features(
        self, nav_df: pd.DataFrame, trades_df: pd.DataFrame
    ) -> dict[str, float]:
        """从策略净值+交易数据提取 12 维特征"""
        ...

    def extract_features_with_window(
        self, trades_df: pd.DataFrame, window_days: int
    ) -> dict[str, float]:
        """滚动窗口特征提取：仅使用最近 N 天的交易"""
        ...
```

### 7.6 MatchingBackend（统一匹配接口）

见第四节。`RecommendationService` 通过此接口与具体算法解耦。

### 7.7 RecommendationService（推荐调度器）

```python
class RecommendationService:
    """
    推荐调度器。通过 MatchingBackend 抽象接口调用具体算法。
    支持多后端结果对比（Phase 2 功能）。
    """

    def __init__(self, registry: BackendRegistry, popup_gen: "PopupGenerator"):
        self.registry = registry
        self.popup_gen = popup_gen

    def recommend(
        self, user_id: str, profile: UserProfile,
        backend_name: str = "statistical", top_n: int = 3,
    ) -> MatchingResult:
        """
        主推荐入口：
        1. 获取指定后端
        2. 调用 backend.predict() 得到 Top-N
        3. 调用 popup_gen.generate() 生成弹窗话术
        4. 返回 MatchingResult
        """
        ...

    def compare_backends(
        self, user_id: str, profile: UserProfile, top_n: int = 3,
    ) -> dict[str, MatchingResult]:
        """
        对比所有已激活后端的推荐结果（用于三路线交叉验证可视化）。
        返回: {"statistical": result1, "lstm": result2, "fusion": result3, ...}
        """
        ...
```

### 7.8 PopupGenerator（弹窗话术生成器）

```python
class PopupGenerator:
    """
    客户端推荐弹窗话术生成器。

    对应客户需求：当用户登录 APP 交易时，后台自动匹配后弹出提示。
    话术模板基于项目计划书中的示例格式：

    "XXX 策略与您的交易风格匹配度达 70%，行业分布偏好匹配度也超过 70%。
     同时该策略过去半年的收益跑赢 XX%、回撤低 XX%，
     欢迎了解该策略，优化您的投资体验。"
    """

    def generate(
        self,
        strategy_id: str,
        similarity: float,
        top_similar_dims: list[dict],
        top_different_dims: list[dict],
        strategy_nav_info: dict,    # {"annual_return": float, "max_drawdown": float}
        confidence: str,            # "low" | "medium" | "high"
    ) -> str:
        """生成推荐弹窗话术"""
        ...
```

---

## 八、行业维度匹配（与三层特征体系衔接）

### 8.1 与现有设计的衔接

项目计划书要求使用 **28 维申万行业分布向量 + JS 散度**计算行业相似度。当前 pipeline 中的 **ETF 占比 (`etf_ratio`)** 已作为资产偏好特征代理了行业偏好。两者关系：

| 方案 | 维度 | 数据来源 | 当前状态 |
|------|------|---------|---------|
| `etf_ratio`（现有） | 1 维 | 代码前缀模式匹配 | 已实现，在三层特征体系的第二层 |
| 28 维行业向量（扩展） | 28 维 | 个股申万行业分类 | 预留，未来接入 |

### 8.2 实现策略

1. **当前阶段**：继续使用 `etf_ratio` 作为行业偏好代理，嵌入在三层特征体系的第 2 层。`industry_vector` 字段初始化为基于 `etf_ratio` 的粗估（ETF 偏好高的用户，宽基行业权重略高）。

2. **未来扩展**：当接入个股行业分类后，`FeatureExtractor` 可输出完整的 28 维行业分布向量。`StatisticalBackend.predict()` 在推荐时可选地融合行业 JS 散度：

```python
def _blend_industry_similarity(
    self, pca_similarity: float, industry_js: float,
    industry_weight: float = 0.3,
) -> float:
    """
    融合 PCA 相似度与行业 JS 散度相似度
    sim_total = (1 - w_ind) × sim_pca + w_ind × (1 - D_JS / ln2)
    """
    return (1 - industry_weight) * pca_similarity + industry_weight * (1 - industry_js / np.log(2))
```

---

## 九、Streamlit 应用页面设计

### 9.1 页面结构（st.sidebar.radio 路由）

```
┌─ 侧边栏 ─────────────────┐
│ 策略匹配推荐系统           │
│                           │
│ 登录状态: 张三 ✓           │
│ 画像置信度: low            │
│                           │
│ 📋 首页 (Dashboard)        │
│ 📝 完善问卷               │
│ 📤 上传交易数据            │
│ 📊 我的画像               │
│ 🎯 推荐策略               │
│ 📐 匹配稳定性              │
│ 🔧 设置                   │
│ 🚪 退出登录               │
└───────────────────────────┘
```

### 9.2 各页面详细设计

#### 页面 1：登录/注册（未登录时唯一可见页面）

```
┌──────────────────────────────────────┐
│        投资策略匹配推荐系统            │
│                                      │
│   [登录] | [注册]                     │
│                                      │
│   用户名: [____________]              │
│   密  码: [____________]              │
│                                      │
│        [ 登录 / 注册 ]                │
│                                      │
│   提示: 首次使用请先注册              │
└──────────────────────────────────────┘
```

#### 页面 2：首页 Dashboard（登录后默认页）

```
┌──────────────────────────────────────────────────────┐
│  欢迎回来，张三！                                      │
│                                                      │
│  ┌─ 当前状态 ───────────────────────────────┐        │
│  │  问卷完成度: L1 ✓ | L2 ✗ | L3 ✗           │        │
│  │  画像状态: 问卷初始化 (β=0.7, 置信度:低)  │        │
│  │  匹配后端: PCA 统计方法                   │        │
│  │  交易数据: 未上传                          │        │
│  │  画像更新次数: 0                           │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 快速开始 ───────────────────────────────┐        │
│  │  [完善问卷 →]  [上传交易数据 →]            │        │
│  │  [查看推荐 →]                              │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 快速推荐（基于当前画像）────────────────┐        │
│  │  提示: 当前画像置信度低，建议完善问卷或    │        │
│  │       上传交易数据以获得更准确的推荐。     │        │
│  │                                          │        │
│  │  #1 HX_朝花夕拾   匹配度 17.8%  [详情 →] │        │
│  │  #2 GZ2000_综合  匹配度 10.3%  [详情 →] │        │
│  │  #3 HS300_综合   匹配度 -7.8%  [详情 →] │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

#### 页面 3：完善问卷

```
┌──────────────────────────────────────────────────────┐
│  完善投资问卷                                          │
│                                                      │
│  L1: 基础风险评估  ✓ 已完成  (2分钟)                  │
│  L2: 投资偏好深化  [开始填写 →] (5分钟)               │
│  L3: 交易行为进阶  [开始填写 →] (8分钟)               │
│                                                      │
│  提示: 填写更深层级的问卷可以提升推荐准确度。          │
│        您随时可以回来补充填写。                        │
└──────────────────────────────────────────────────────┘
```

#### 页面 4：上传交易数据

```
┌──────────────────────────────────────────────────────┐
│  上传交易数据                                          │
│                                                      │
│  支持格式: Excel (.xlsx) 或 CSV                       │
│  要求列: 交易日期、操作类型、股票代码、价格、数量       │
│                                                      │
│  [选择文件] 未选择文件                                 │
│                                                      │
│  分析时间窗口: [全部数据 ▼]                            │
│              选项: 全部 / 最近 60 天 / 最近 120 天     │
│                                                      │
│  [ 上传并分析 ]                                       │
│                                                      │
│  ┌─ 上传历史 ─────────────────────────────┐          │
│  │  2026-05-20  |  127笔  |  已处理 ✓     │          │
│  │  2026-05-22  |  15笔   |  已处理 ✓     │          │
│  └────────────────────────────────────────┘          │
│                                                      │
│  提示: 每次上传后会基于动量算法更新您的画像。          │
│  当前画像更新次数: 2 次 (置信度: 低 → 中)             │
└──────────────────────────────────────────────────────┘
```

#### 页面 5：我的画像

```
┌──────────────────────────────────────────────────────┐
│  我的投资画像                                          │
│                                                      │
│  ┌─ 画像来源 ───────────────────────────────┐        │
│  │  来源: 问卷 + 2次交易数据更新              │        │
│  │  β = 0.65 (行为特征权重)                   │        │
│  │  风险承受能力: 3/5 (中等)                  │        │
│  │  置信度: medium (中等可靠)                 │        │
│  │  最近更新: 2026-05-23 14:30               │        │
│  │  匹配后端: PCA 统计方法                    │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 12维特征雷达图 ─────────────────────────┐        │
│  │  (可视化：用户 vs 策略平均)                │        │
│  │  ─●─ 我的画像  --- 策略平均               │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 画像变化轨迹 ───────────────────────────┐        │
│  │  更新#0 (问卷) → 更新#1 (交易) → 当前     │        │
│  │  PC1:  2.3 → 1.8 → 1.5                   │        │
│  │  PC2: -0.5 → 0.2 → 0.4                   │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 行业分布偏好 ───────────────────────────┐        │
│  │  (预留: 当接入个股行业分类后展示 28 维向量)│        │
│  │  当前: 基于 ETF 占比的粗估                │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

#### 页面 6：推荐策略

```
┌──────────────────────────────────────────────────────┐
│  策略推荐                                              │
│                                                      │
│  匹配后端: [融合推荐 (70%统计+30%LSTM) ▼]             │
│  画像置信度: medium                                   │
│                                                      │
│  ┌─ 推荐结果 ───────────────────────────────┐        │
│  │                                           │        │
│  │  🥇 HX_朝花夕拾                            │        │
│  │     融合匹配度: 22.5%                      │        │
│  │     统计排名: #2 (14.1%) | LSTM 排名: #1   │        │
│  │     与您最相似的维度: 持仓周期、买卖对称性  │        │
│  │     差异最大的维度: ETF偏好、趋势偏好       │        │
│  │     [查看详情]                            │        │
│  │                                           │        │
│  │  🥈 GZ2000_综合_v10                       │        │
│  │     融合匹配度: 18.7%                      │        │
│  │     统计排名: #1 (17.8%) | LSTM 排名: #4   │        │
│  │     [查看详情]                            │        │
│  │                                           │        │
│  │  🥉 HS300_综合_v10                        │        │
│  │     融合匹配度: 8.2%                       │        │
│  │     统计排名: #3 (-7.8%) | LSTM 排名: #3   │        │
│  │     [查看详情]                            │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─ 弹窗话术预览 ───────────────────────────┐        │
│  │  "策略 HX_朝花夕拾 与您的投资风格匹配度    │        │
│  │   22.5%（统计 14.1% + LSTM 序列风格辅助）。│        │
│  │   您在持仓周期、买卖对称性方面与该策略      │        │
│  │   风格最为接近，建议进一步了解该策略。"     │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘

[展开详情] → 点击某策略后（融合模式）：
┌──────────────────────────────────────────────────┐
│  HX_朝花夕拾  详细匹配报告                         │
│                                                  │
│  融合得分: 22.5% = 0.7×14.1%(统计) + 0.3×42.3%(LSTM)
│                                                  │
│  维度差异归因（统计侧）：                          │
│  ┌────────────┬────────┬────────┬───────┐        │
│  │ 特征       │ 您     │ 策略   │ 差异  │        │
│  ├────────────┼────────┼────────┼───────┤        │
│  │ 持仓周期   │ 2天    │ 2天    │ ★ 极小│        │
│  │ 换手率     │ 3.67   │ 3.20   │ ★ 小  │        │
│  │ ETF偏好    │ 0.01   │ 0.00   │       │        │
│  │ 趋势偏好   │ 0.15   │ 1.20   │ ⚠ 大  │        │
│  └────────────┴────────┴────────┴───────┘        │
│                                                  │
│  LSTM 序列风格:                                  │
│  相似度 = 0.423 (排名 #1/34)                     │
│  SHAP 驱动特征: 持仓周期 > 集中度 > 买卖对称性    │
└──────────────────────────────────────────────────┘
```

#### 页面 7：匹配稳定性（新增 — 滚动窗口分析）

```
┌──────────────────────────────────────────────────────┐
│  匹配稳定性分析                                        │
│                                                      │
│  提示: 使用不同时间窗口的交易数据计算推荐结果，         │
│        观察匹配是否稳定。                              │
│                                                      │
│  ┌─ 多窗口推荐对比 ─────────────────────────┐        │
│  │  窗口      │ Top-1 策略     │ 匹配度     │        │
│  ├────────────┼───────────────┼────────────┤        │
│  │ 全量       │ HX_朝花夕拾    │ 17.8%     │        │
│  │ 最近 120 天│ HX_朝花夕拾    │ 16.2%     │        │
│  │ 最近 60 天 │ HX_朝花夕拾    │ 15.1%     │        │
│  │ 最近 30 天 │ GZ2000_综合    │ 12.0%     │        │
│  └────────────┴───────────────┴────────────┘        │
│                                                      │
│  结论: 最近 30 天 Top-1 策略发生变化，                 │
│        建议关注近期交易风格是否有显著变化。            │
│                                                      │
│  ┌─ 匹配度趋势图 ───────────────────────────┐        │
│  │  (折线图: 不同窗口下的 Top-1 匹配度)       │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

#### 页面 8：设置

```
┌──────────────────────────────────────────────────────┐
│  设置                                                  │
│                                                      │
│  匹配算法: [融合推荐 (70%统计+30%LSTM) ▼]             │
│            选项:                                       │
│              - 融合推荐 (70%统计+30%LSTM) [默认]       │
│              - PCA 统计方法                            │
│              - LSTM 序列风格匹配                       │
│                                                      │
│  融合权重 α (仅融合模式):                              │
│            [━━━━━━━━●━━━━━━━] 0.7                      │
│            选项: 0.8(稳健) / 0.7(平衡) / 0.6(实验)    │
│                                                      │
│  β 超参数: 0.65 [━━━━━━━━●━━━━━━━]                    │
│            (当前: 行为特征权重 65%)                    │
│            [ 使用问卷值 ]  [ 手动调整 ]                │
│                                                      │
│  数据管理:                                             │
│  [ 导出画像数据 ]  [ 清除交易数据 ]                    │
│                                                      │
│  关于:                                                 │
│  统计方法: PCA + 径向惩罚余弦 (λ=1.0)                  │
│  LSTM 方法: BiLSTM 128 维序列风格匹配 (DLMethod)       │
│  融合公式: final = α * stat_norm + (1-α) * ml_norm    │
│  参考: report.md, DLMethod/ML_INTEGRATION_GUIDE.md    │
└──────────────────────────────────────────────────────┘
```

---

## 十、完整交互流程

```
                    ┌─────────────┐
                    │  用户访问    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ 登录/注册    │ ← 新用户注册
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
              ┌──── │ L1 问卷     │ ← 必填，5题，2分钟
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │ 创建初始画像  │ ← β 粗估 + 12维特征粗估
              │     │ 置信度: low  │
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │   Dashboard  │ ← 显示状态 + 快速推荐(低置信度提示)
              │     └──┬───────┬──┘
              │        ▼       ▼
              │  ┌──────┐ ┌──────────┐
              │  │ L2/L3│ │ 上传交易  │
              │  │ 问卷  │ │ 数据     │
              │  └──┬───┘ └────┬─────┘
              │     ▼          ▼
              │  ┌──────────────┐
              │  │ 更新画像      │ ← β 精调 或 EMA 动量更新
              │  │ 置信度提升    │
              │  └──────┬───────┘
              │         ▼
              │  ┌─────────────┐
              └─▶│ 推荐策略     │ ← MatchingBackend.predict()
              │  │ + 弹窗话术   │ ← PopupGenerator.generate()
              │  └──┬──────────┘
              │     ▼
              │  ┌─────────────┐
              └──│ 稳定性分析   │ ← 多窗口推荐对比
                 └─────────────┘
                           │ (循环)
                           ▼
              用户继续上传交易 → 画像更新 → 推荐刷新 → 置信度提升
```

---

## 十一、依赖与配置

### 11.1 新增依赖

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
filelock>=3.12.0        # JSON 文件读写锁
```

### 11.2 启动方式

```bash
streamlit run app.py
```

---

## 十二、安全与限制说明

1. **密码存储**：SHA-256 哈希，非明文。**非生产级安全**，仅用于演示。生产环境需换 bcrypt。
2. **并发**：JSON 文件存储通过 `filelock` 提供基础并发保护，但不支持高并发。生产环境应替换为 SQLite 或真实数据库。
3. **数据隔离**：每个用户的画像和交易数据以 `user_id` 为 key 隔离存储。
4. **文件权限**：`app/data/` 目录需要读写权限。

---

## 十三、实现优先级（分阶段）

### Phase 1（MVP — 已完成）

- 目录结构搭建
- 数据模型定义（User, UserProfile, Questionnaire, MatchingResult）
- 存储引擎（JSON 文件 + filelock）
- 注册/登录
- Level 1 问卷（5 题）+ β 粗估 + 画像初始化
- MatchingBackend 抽象基类 + StatisticalBackend 实现
- FeatureExtractor（抽象 pipeline.py 方法）
- RecommendationService + PopupGenerator
- Dashboard + 推荐页面 + 问卷页面 + 画像页面
- 上传交易数据（基础版，全量分析）

### Phase 2（本次新增 — DLMethod ML 接入）

- LSTMBackend 实现：加载 `matching_phase2_lstm.csv` + `final_recommendations.csv`
- FusionBackend 实现：α=0.7 统计 + 0.3 LSTM，Min-Max 归一化加权融合
- 账户名映射：webapp 用户 → `Account_A/B/C`（DLMethod 预计算矩阵的账户名）
- 推荐页面升级：展示双源排名（统计排名 + LSTM 排名 + 融合排名）
- 设置页面：后端切换（统计 / LSTM / 融合）、融合权重 α 调节
- 多后端交叉验证可视化（稳定性分析页面对比三路线结果）
- Level 2 问卷（8 题）+ β 精调
- EMA 画像更新 + 置信度体系
- 画像雷达图 + 变化轨迹可视化

### Phase 3（生产级探索）

- SQLite 替换 JSON 存储
- bcrypt 替换 SHA-256 密码哈希
- 真实用户数据接入（非模拟）
- 28 维申万行业分布 + JS 散度融合
- LSTM 模型增量训练：当积累足够真实用户交易数据后重新训练

---

## 十四、关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 存储方案 | JSON 文件 | 原型阶段，简单可调试 |
| 密码哈希 | SHA-256 | 演示级安全，生产级换 bcrypt |
| 问卷分级 | 3 级渐进 | 降低初始门槛，可逐步深化 |
| 画像更新 | EMA 动量 | 冷启动快、后期稳，类比 SGD Momentum |
| 与 pipeline 关系 | 导入不修改 | 理论与应用代码解耦 |
| 匹配引擎 | 抽象接口 + 三后端 | 统计路线 + LSTM 序列风格 + 融合加权，三线并行 |
| 行业匹配 | 先用 etf_ratio 代理，预留 28 维向量 | 与现有三层特征体系无缝衔接 |
| 置信度体系 | 三级 (low/medium/high) | 类比贝叶斯收缩思想，冷启动透明化 |
| 弹窗话术 | 独立组件 | 对应客户明确需求，可独立迭代文案 |
| 滚动窗口 | ProfileService 参数化 | 复用同一套特征提取，仅过滤时间范围 |
| Streamlit | 纯服务端 | 不引入前端框架，快速迭代 |

请审阅以上设计文档，确认无误后开始编码实现。

---

## 十五、DLMethod 机器学习接入详细设计

### 15.1 数据流概览

```
DLMethod/ (预训练输出)
├── matching_phase2_lstm.csv     ← 3×34 相似度矩阵 (Account_A/B/C × 34 策略)
├── final_recommendations.csv    ← Top-5 推荐长表 (含 Phase1/Phase2 排名)
├── shap_analysis.json           ← SHAP 特征归因结果
└── embedding_meta.json          ← 账户名/策略名映射

Webapp 用户
├── 注册 → L1 问卷 → 画像初始化 (统计侧冷启动)
├── 上传交易数据 → EMA 更新画像 → StatisticalBackend 推荐
└── 上传交易数据 → 账户名映射 → LSTMBackend 查表推荐
                                      ↓
                              FusionBackend 加权融合
                                      ↓
                              推荐页面展示三路线结果
```

### 15.2 账户名映射策略

DLMethod 预计算的相似度矩阵使用 `Account_A`, `Account_B`, `Account_C` 三个账户名。
Webapp 用户是动态注册的，无法一一对应。采用以下映射策略：

| Webapp 用户状态 | 映射方式 | LSTM 结果使用 |
|----------------|---------|-------------|
| 第 1 个上传交易数据的用户 | → `Account_A` | 直接使用预计算矩阵 |
| 第 2 个上传交易数据的用户 | → `Account_B` | 直接使用预计算矩阵 |
| 第 3 个上传交易数据的用户 | → `Account_C` | 直接使用预计算矩阵 |
| 第 4+ 个用户 | 无直接映射 | 仅展示统计推荐，LSTM 侧显示"数据不足" |

**实现方式**：`LSTMBackend` 在 `fit()` 时记录已分配的账户映射，
`assign_lstm_account(user_id)` 函数按上传顺序分配。

**未来扩展**：当积累足够真实用户交易数据后，可以重新训练 LSTM 模型，
生成更大规模的相似度矩阵（如 200 账户 × N 策略），此时映射不再是瓶颈。

### 15.3 FusionBackend 融合流程

```
输入: user_features (12D), beta, top_n
输出: 融合 Top-N 推荐

1. StatisticalBackend.predict(user_features, beta, top_n)
   → stat_scores: {strategy_name: raw_similarity}

2. LSTMBackend.predict(user_features, beta, top_n)
   → ml_scores: {strategy_name: lstm_cosine_similarity}

3. 对齐共同策略集合:
   common_strategies = set(stat_scores) ∩ set(ml_scores)

4. Min-Max 归一化（各自内部归一化到 0-1）:
   stat_norm[s] = (stat_scores[s] - min(stat)) / (max(stat) - min(stat))
   ml_norm[s] = (ml_scores[s] - min(ml)) / (max(ml) - min(ml))

5. 加权融合:
   final_score[s] = α * stat_norm[s] + (1-α) * ml_norm[s]

6. 排序取 Top-N，返回:
   {
       "top3": [{"strategy": s, "similarity": final_score[s], "rank": i}],
       "explanation": {统计归因 + LSTM SHAP 归因},
       "phase1_rank": {s: rank},    # 特征基线排名
       "phase2_rank": {s: rank},    # LSTM 排名
       "stat_score": {s: stat_norm[s]},
       "ml_score": {s: ml_norm[s]},
   }
```

### 15.4 三后端对比展示（稳定性分析页面扩展）

稳定性分析页面原有"多窗口推荐对比"功能，新增"多算法路线对比"：

```
┌─ 三路线推荐对比 ────────────────────────────────────┐
│  窗口: [全量数据 ▼]                                  │
│                                                    │
│  ┌──────────┬──────────────┬────────────┬─────────┐ │
│  │ 排名     │ 统计方法     │ LSTM 序列  │ 融合    │ │
│  ├──────────┼──────────────┼────────────┼─────────┤ │
│  │ #1       │ GZ2000_综合  │ HX_朝花夕拾│ HX_朝花 │ │
│  │ 得分     │ 17.8%        │ 42.3%      │ 22.5%   │ │
│  ├──────────┼──────────────┼────────────┼─────────┤ │
│  │ #2       │ HX_朝花夕拾  │ 行业etf增强│ GZ2000  │ │
│  │ 得分     │ 10.3%        │ 38.1%      │ 18.7%   │ │
│  ├──────────┼──────────────┼────────────┼─────────┤ │
│  │ #3       │ HS300_综合   │ 综合全     │ HS300   │ │
│  │ 得分     │ -7.8%        │ 25.6%      │ 8.2%    │ │
│  └──────────┴──────────────┴────────────┴─────────┘ │
│                                                    │
│  结论: 统计方法推荐 GZ2000（高频小盘风格匹配），       │
│        LSTM 推荐 HX_朝花夕拾（交易序列风格相似），     │
│        融合后 HX_朝花夕拾排名第一。                  │
└────────────────────────────────────────────────────┘
```

### 15.5 依赖与运行环境

LSTMBackend 需要以下依赖（仅在 `lstm` 或 `fusion` 后端激活时加载）：

```python
# 条件导入，避免不影响 statistical 后端的轻量运行
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    HAS_LSTM_DEPS = True
except ImportError:
    HAS_LSTM_DEPS = False
```

DLMethod 预训练文件路径配置（`config.py`）：

```python
# DLMethod 输出文件路径
DLMETHOD_DIR = Path(__file__).parent.parent / "DLMethod"
LSTM_SIMILARITY_MATRIX = DLMETHOD_DIR / "matching_phase2_lstm.csv"
LSTM_RECOMMENDATIONS = DLMETHOD_DIR / "final_recommendations.csv"
LSTM_SHAP_ANALYSIS = DLMETHOD_DIR / "shap_analysis.json"
LSTM_EMBEDDING_META = DLMETHOD_DIR / "embedding_meta.json"

# 融合权重
FUSION_ALPHA = 0.7  # 统计方法权重，1-alpha = LSTM 权重
```

### 15.6 方法边界声明（前端展示）

在推荐页面和设置页面中，需要明确标注：

> **LSTM 方法说明**：本系统的 LSTM 匹配模块基于小样本弱监督学习训练，
> 训练标签来自交易画像相似度生成的伪标签。它适合作为统计方法的辅助补充，
> 用于捕捉交易序列风格的相似性。最终推荐仍结合统计方法加权得出，
> 并建议结合风险承受能力、人工校验等综合判断。

在 SHAP 归因展示中，注明 Top-5 驱动特征：
**持仓周期 > 集中度 > 买卖对称性 > 换手率 > 波动偏好**（行业偏好贡献极低）。
