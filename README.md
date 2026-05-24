# 用户画像-投资策略匹配系统

基于行为特征与 PCA 降维的量化策略推荐引擎，通过 **径向惩罚余弦相似度** 为用户匹配风格一致的投资策略。

提供**三条并行路线**：
1. **传统统计学路线（主线）**：三层特征体系 + PCA + 径向惩罚余弦，可解释、可落地
2. **LSTM 深度学习路线（辅线）**：BiLSTM 128 维序列风格匹配（DLMethod 团队输出），用于交叉验证
3. **融合路线（推荐默认）**：α=0.7 统计 + 0.3 LSTM，Min-Max 归一化加权融合

并配有 **Streamlit 网页应用**，支持用户注册登录、问卷调查、交易数据上传、动态画像更新、策略推荐与可视化。

---

## 核心假设

整个匹配体系建立在四个基本假设之上：

1. **收益是过滤条件，不是匹配维度** — 没有客户会追求低收益。策略收益率只需满足"不低于用户历史收益"的门槛，不参与相似度计算。
2. **风格一致带来信任** — 用户更愿意接受"看起来像我自己会选的策略"。持仓周期、换手率、集中度等行为特征的对齐比收益率相似更重要。
3. **行为画像优于绝对收益** — 当缺乏期初资金数据时，从交易模式中反推用户"喜欢做什么"比强行估算绝对收益率更稳健。
4. **用户与策略可在同一特征空间中度量** — 无论是策略还是用户，"持仓周期""换手率""买卖对称性"等概念的定义是普适的。

---

## 方法概要

### 三层特征体系（12 维）

| 层次 | 维度 | 特征 | 刻画什么 |
|------|------|------|---------|
| **交易行为** | 6 | 持仓周期、换手率、买卖对称性、持仓集中度(HHI)、处置效应、胜率 | 用户"怎么做交易" |
| **资产偏好** | 3 | ETF占比、价格区间偏好、分仓均匀度 | 用户"喜欢买什么" |
| **风险代理** | 3 | 亏损幅度、波动偏好、趋势偏好 | 用户"能承受多大风险" |

所有 12 维特征**仅从交易流水计算**，无需行情数据或账户资金信息。

### 相似度度量：径向惩罚余弦

$$\text{sim}(u, s) = \frac{u \cdot s}{\|u\| \cdot \|s\|} \times \exp\left(-\lambda \cdot \left|\log\frac{\|u\|}{\|s\|}\right|\right)$$

同时捕捉**方向一致度**（余弦）与**模长差异**（径向惩罚），兼顾两种信息：
- 方向一致 + 模长接近 → 满分
- 方向一致 + 模长悬殊 → 降权
- 方向相反 → 负值（保留"风格相反"信号）

### 超参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| **β** (beta) | 0.5 | 行为特征 vs 非行为特征的权重比 |
| **λ** (lambda) | 1.0 | 径向惩罚的强度 |

### 画像动态更新（EMA 动量）

用户画像通过 EMA（指数移动平均）动量算法逐步更新：
- 问卷初始化 → 初始画像（置信度: low）
- 每次上传交易数据 → EMA 动量更新（类比 SGD with Momentum）
- 自适应衰减因子 γ，随更新次数从 0.35 趋近 0.7
- 置信度等级：low → medium（3 次更新）→ high（9 次更新）

---

## 快速开始

### 1. 安装依赖

推荐使用虚拟环境：

```bash
# 创建虚拟环境（如已有则跳过）
python -m venv investmentMatching

# 激活虚拟环境
# Windows:
investmentMatching\Scripts\activate
# Linux/macOS:
source investmentMatching/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行数据分析流水线

```bash
python pipeline.py
```

执行后将在 `output/` 目录下生成特征向量、PCA 结果、推荐结果与可视化图表。

### 3. 启动网页应用

```bash
# 使用虚拟环境中的 Python
investmentMatching/Scripts/python.exe -m streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 4. 生成模拟数据（可选）

用于大规模测试和 Word2Vec 训练：

```bash
python generate_simulated_data.py
```

生成 500 个模拟策略 + 200 个模拟用户的交易数据，输出至 `output/simulated_data/`。

---

## 项目结构

```
Investment-strategy-and-user-profile-matching/
│
├── pipeline.py                          # 主流水线代码（不修改）
│                                         # 三层特征提取 + PCA + 相似度计算
├── generate_simulated_data.py           # 模拟数据生成器（500策略+200用户）
│
├── app.py                               # Streamlit 网页应用主入口
│
├── stats_data/                          # 源数据（统一存放）
│   ├── 量化策略绩效-1.xlsx              # DLMethod: 12 策略交易记录
│   ├── 量化策略绩效-2.xlsx              # DLMethod: 22 策略交易记录
│   ├── 带ETF的策略1.csv                 # 补充策略 1
│   ├── 带ETF策略2.csv                   # 补充策略 2
│   ├── 朝花夕拾策略.csv                 # 补充策略 3
│   ├── 模拟账户A的记录.xlsx             # 模拟用户 A
│   ├── 模拟账户B的记录.xlsx             # 模拟用户 B
│   ├── 模拟账户C的记录.xlsx             # 模拟用户 C
│   └── 净值_交易_资金及字段说明（相关性数据分析）/
│       └── products_export_20260518_163122/  # 7 策略（目录格式）
│
├── app/
│   ├── config.py                        # 全局配置（路径、超参数、行业列表）
│   ├── requirements.txt                 # 应用依赖
│   │
│   ├── models/                          # 数据模型
│   │   ├── user.py                      # User, UserProfile, MatchingResult
│   │   └── questionnaire.py             # Question, Questionnaire
│   │
│   ├── services/                        # 业务服务层
│   │   ├── auth.py                      # 注册/登录/会话管理
│   │   ├── storage.py                   # JSON 文件存储引擎（含 filelock）
│   │   ├── questionnaire.py             # 三级问卷定义 + 评分引擎
│   │   ├── feature_extractor.py         # 从 pipeline.py 抽象出的特征提取
│   │   ├── profile.py                   # 画像管理 + EMA 动量更新
│   │   ├── matching_backend.py          # 统一匹配引擎接口（抽象基类）
│   │   ├── recommendation.py            # 推荐调度器
│   │   ├── popup_generator.py           # 客户端弹窗话术生成器
│   │   ├── excel_strategy_loader.py     # 从 Excel 文件加载37种策略数据
│   │   │
│   │   └── backends/                    # 匹配算法后端
│   │       ├── statistical.py           # PCA + 径向惩罚余弦（已实现）
│   │       ├── lstm.py                  # DLMethod: LSTM 128 维序列风格匹配
│   │       └── fusion.py                # 统计 + LSTM 加权融合（α=0.7）
│   │
│   └── data/                            # 本地持久化数据（运行时自动创建）
│       ├── users.json                   # 用户账号信息
│       ├── profiles/                    # 用户画像（每人一个文件）
│       ├── trades/                      # 用户上传的交易记录
│       └── questionnaires/              # 问卷回答存档
│
├── output/                              # 流水线输出
│   ├── strategy_features_v2.csv         # 策略 12 维特征
│   ├── user_features_v2.csv             # 用户 12 维特征
│   ├── pca_results_v2.csv               # PCA 投影坐标
│   ├── pca_info_v2.json                 # PCA 模型详情
│   ├── recommendations_v2.json          # 推荐结果 + 可解释归因
│   └── *.png                            # 可视化图表
│
├── DLMethod/                            # 深度学习团队流水线
│   ├── step1_data_loader.py             # 数据加载与清洗
│   ├── step2_industry_mapping.py        # 行业映射
│   ├── step3_feature_extraction.py      # 特征提取
│   ├── step4_word2vec_pretrain.py       # Word2Vec 预训练
│   ├── step5_simulate_data.py           # 模拟数据
│   ├── step6_lstm_contrastive.py        # LSTM 对比学习
│   ├── step7_evaluation.py              # 评估归因
│   ├── matching_phase2_lstm.csv         # LSTM 相似度矩阵（3 账户 × 37 策略）
│   ├── final_recommendations.csv        # Top-N 推荐长表
│   └── shap_analysis.json               # SHAP 特征归因
│
├── docs/                                # 文档
│   ├── APP_DESIGN.md                    # 网页应用详细设计文档（V3.0）
│   ├── report.md                        # 数据分析完整报告
│   └── DATA_ANALYSIS_WORKFLOW.md        # 数据分析工作流
│
├── README.md                            # 本文件
└── requirements.txt                     # 项目依赖
```

---

## 数据

### 策略数据

**总计 44 种策略**，来自两个数据源：

| 来源 | 数量 | 格式 | 用途 |
|------|------|------|------|
| `stats_data/...products_export_20260518_163122/` | 7 | 每策略一个目录（daily_value.csv + trades.csv） | 统计方法主线数据，含完整净值曲线 |
| `stats_data/量化策略绩效-1.xlsx` | 12 | Excel 多 sheet | DLMethod 策略数据（交易记录） |
| `stats_data/量化策略绩效-2.xlsx` | 22 | Excel 多 sheet | DLMethod 策略数据（交易记录） |
| `stats_data/` 补充 CSV × 3 | 3 | CSV 交易记录 | 补充 ETF 策略和择时策略 |

其中 Excel/CSV 来源的 **37 种策略** 同时用于统计方法和 LSTM 方法的匹配与融合，确保两个后端有完全相同的策略池。

### 用户数据（模拟）

3 个模拟用户的交易流水：
- 用户 A：127 笔交易，长线低频风格（持仓 52 天）
- 用户 B：1727 笔交易，超短线高频风格（持仓 2 天）
- 用户 C：285 笔交易，短线中频风格（胜率 67%）

---

## 网页应用功能

### 用户流程

```
注册/登录 → Level 1 问卷（必填） → 创建初始画像 → Dashboard
    ↓                              ↓
  完善 L2/L3 问卷              上传交易数据
    ↓                              ↓
  β 精调 + 特征细化          EMA 动量更新画像
                                    ↓
                            策略推荐 + 可解释归因（三后端）
                                    ↓
                            匹配稳定性分析（多窗口）
```

### 核心页面

| 页面 | 功能 |
|------|------|
| **首页 Dashboard** | 当前状态概览、快速导航、实时推荐 |
| **完善问卷** | 三级渐进问卷（L1 必填 5 题、L2 可选 8 题、L3 可选 10 题） |
| **上传交易数据** | 支持 Excel/CSV，可选择分析时间窗口（全量/30/60/120 天） |
| **我的画像** | 12 维特征雷达图、画像变化轨迹、行业分布预览 |
| **推荐策略** | Top-3 推荐 + 维度级归因 + 弹窗话术预览 + 双源排名 |
| **匹配稳定性** | 多窗口推荐对比 + 多后端对比 + 趋势图 + 一致性结论 |
| **设置** | β 手动调整、数据导出/清除、后端切换、融合权重 α 调节 |

### 三后端对比

| 后端 | 定位 | 推荐场景 |
|------|------|---------|
| `statistical` | **主线**：12 维特征 + PCA + 径向惩罚余弦 | 可解释性优先 |
| `lstm` | **辅线**：交易序列风格相似度 | 候选策略排序、互相验证 |
| `fusion` | **推荐默认**：α=0.7 stat + 0.3 LSTM | 综合推荐 |

---

## 匹配结果示例

| 用户 | 画像 | 推荐策略（统计） | 匹配度 |
|------|------|-----------------|--------|
| A | 长线低频，持仓 52 天 | 宽指配置_V251216 | 13.4% |
| B | 超短线高频，持仓 2 天 | HX_朝花夕拾 | 17.8% |
| C | 短线中频，胜率 67% | HX_朝花夕拾 | 22.3% |

每个推荐都附带**维度级归因**，说明哪些特征最相似、哪些最不同。

---

## 架构设计

### 代码分层

```
┌─────────────────────────────────────────────────────┐
│  app.py (Streamlit UI)                              │
│  - 页面路由、交互组件、可视化                        │
├─────────────────────────────────────────────────────┤
│  app/services/ (业务服务层)                          │
│  - RecommendationService: 调度匹配后端              │
│  - ProfileService: 画像管理 + EMA 更新              │
│  - QuestionnaireService: 问卷评分 → 特征映射         │
│  - PopupGenerator: 客户端弹窗话术                    │
│  - ExcelStrategyLoader: Excel 策略数据适配器        │
├─────────────────────────────────────────────────────┤
│  app/services/backends/ (匹配算法层)                 │
│  - StatisticalBackend: PCA + 径向惩罚余弦            │
│  - LSTMBackend: DLMethod 预计算结果查表              │
│  - FusionBackend: α 加权融合（Min-Max 归一化）       │
│  统一接口: name() / fit() / predict() / get_all_metrics()
├─────────────────────────────────────────────────────┤
│  app/services/feature_extractor.py                   │
│  - 导入 pipeline.py 函数，包装为用户/策略特征提取    │
├─────────────────────────────────────────────────────┤
│  pipeline.py (理论流水线，不修改)                    │
│  - 特征提取、PCA、相似度计算、可解释输出             │
└─────────────────────────────────────────────────────┘
```

### 关键实现细节

- **FIFO 配对**：买卖逐笔匹配计算已实现收益和持仓周期
- **PCA 保留模长**：标准化拟合方向，去均值但不缩放投影，保留"极端程度"信息
- **ETF 识别**：代码前缀模式匹配（15/51/56/58/12 开头）
- **匹配引擎抽象**：通过 `MatchingBackend` 接口实现算法解耦，支持三后端切换
- **画像置信度**：三级体系（low/medium/high），类比贝叶斯收缩思想
- **双源策略融合**：统计方法与 LSTM 方法共享 37 种策略池，确保 FusionBackend 有共同策略进行加权融合
- **Excel 数据适配器**：`excel_strategy_loader.py` 将 DLMethod Excel 格式自动转换为统计方法兼容格式，含 btype→BUY/SELL 映射和自动净值序列构建

---

## 局限性

1. 用户数据量有限（仅 3 个模拟用户）
2. β 超参数未通过真实问卷校准
3. 行业维度使用 ETF 占比代理，缺少申万行业精确映射
4. 无法精确计算用户持仓市值（无个股行情数据）
5. 密码存储使用 SHA-256（非生产级安全方案）
6. LSTM 方法基于小样本弱监督学习，训练标签来自伪标签，适合作为统计方法的辅助

详见 `docs/report.md` 第十节。

---

## 路线图

### Phase 1（已完成 ✅）

- [x] 三层特征体系（12 维）
- [x] PCA + 径向惩罚余弦
- [x] 数据分析流水线（pipeline.py）
- [x] 网页应用（注册/问卷/上传/推荐/画像/设置）
- [x] 匹配引擎抽象接口
- [x] 问卷评分引擎

### Phase 2（已完成 ✅）

- [x] LSTMBackend 实现：加载 LSTM 相似度矩阵 + 推荐长表
- [x] FusionBackend 实现：α=0.7 统计 + 0.3 LSTM，Min-Max 归一化
- [x] 统计方法接入 37 种 DLMethod 策略（Excel 数据适配器）
- [x] 三后端策略池对齐（44 种策略：7 原始 + 37 DLMethod）
- [x] 推荐页面双源排名展示
- [x] 设置页面后端切换 + 融合权重调节
- [x] 源数据统一存放至 stats_data/ 目录

### Phase 3（后续迭代）

- [ ] 完整实现 L2/L3 问卷特征映射细化
- [ ] 多后端交叉验证可视化
- [ ] 28 维申万行业分布 + JS 散度融合

### Phase 4（生产级探索）

- [ ] SQLite 替换 JSON 存储
- [ ] bcrypt 替换 SHA-256 密码哈希
- [ ] 真实用户数据接入
- [ ] A/B 测试验证匹配质量
- [ ] LSTM 模型增量训练

---

## 相关文档

- [`docs/report.md`](docs/report.md) — 数据分析完整报告，含假设体系、公式推导、结果解读
- [`docs/APP_DESIGN.md`](docs/APP_DESIGN.md) — 网页应用详细设计文档（V3.0）
- [`docs/DATA_ANALYSIS_WORKFLOW.md`](docs/DATA_ANALYSIS_WORKFLOW.md) — 数据分析完整工作流
- [DLMethod/README.md](DLMethod/README.md) — 深度学习团队流水线说明
- [DLMethod/ML_INTEGRATION_GUIDE.md](DLMethod/ML_INTEGRATION_GUIDE.md) — ML 接入指南

---

## License

内部项目，仅供内部使用。
