# 用户画像-投资策略匹配：完整数据分析工作流

> 基于 PCA 特征提取 + 欧氏距离/余弦相似度的策略推荐方案
> 日期：2026-05-23 | 版本：V1.0

---

## 一、项目总览

### 1.1 核心思路

将**用户交易画像**和**投资策略画像**映射到同一个高维特征空间，在该空间中用距离度量（欧氏距离或余弦相似度）衡量匹配度。

**关键技术路线：**
1. 从策略净值和交易记录中提取多维度特征（收益、风险、换手率、行业集中度等）
2. 从用户交易记录中提取相同维度的特征
3. 对全部特征做 PCA 降维，取前 k 个主成分作为特征空间的基
4. 在 PCA 特征空间中计算用户与策略之间的距离/相似度
5. 为每个用户推荐 Top-N 最匹配的策略

### 1.2 数据源清单

| 数据源 | 内容 | 规模 |
|--------|------|------|
| `净值_交易_资金及字段说明/products_export_20260518_163122/` | 7个策略的 daily_value.csv, trades.csv, funds.csv | 每策略 ~812 天净值 + 数百笔交易 |
| `量化策略绩效-1.xlsx` | 12个策略的绩效汇总表（累计收益率、年化收益、最大回撤） | 12行 × 5列 |
| `量化策略绩效-2.xlsx` | 29个策略的绩效汇总表 | 29行 × 5列 |
| `模拟账户A/B/C的记录.xlsx` | 3个模拟客户的交易明细 | A:127笔, B:1727笔, C:285笔 |
| `更新策略/更新策略/` | 新增策略的交易记录（CSV格式） | 3个文件 |

### 1.3 7个核心策略概览

| 策略ID | 净值区间 | 期末NAV | 最大回撤 | 日收益波动 |
|--------|----------|---------|----------|------------|
| CYB300_综合_v10 | 2023-01 ~ 2026-05 | 2.31 | -12.8% | 1.32% |
| GZ2000_综合_v10 | 2023-01 ~ 2026-05 | 2.47 | -11.2% | 1.07% |
| HS300_综合_v10 | 2023-01 ~ 2026-05 | 1.69 | -15.5% | 0.96% |
| HX_朝花夕拾 | 2025-01 ~ 2026-05 | 3.26 | -19.1% | 3.04% |
| 宽指配置_V251216 | 2023-01 ~ 2026-05 | 1.99 | -14.3% | 1.04% |
| 集中_红利增强_V20 | 2023-01 ~ 2026-05 | 1.75 | -12.8% | 0.99% |
| 集中_行业轮动_V2511 | 2023-01 ~ 2026-05 | 1.91 | -16.3% | 0.87% |

### 1.4 3个模拟用户概览

| 用户 | 交易笔数 | 持股数 | 均价 | 特征 |
|------|----------|--------|------|------|
| A | 127 | 14 | ~8.68 | 低频、少股、低价股偏好 |
| B | 1727 | 89 | ~22.83 | 高频、多股、分散化 |
| C | 285 | 53 | ~26.91 | 中频、创业板/科技股偏好 |

---

## 二、完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 0: 环境准备                              │
│   pip install scikit-learn scipy pandas numpy matplotlib        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: 数据加载与预处理                             │
│  1.1 加载7个策略的 daily_value.csv → 净值时间序列                  │
│  1.2 加载7个策略的 trades.csv → 交易记录                          │
│  1.3 加载绩效汇总 Excel → 策略级指标                              │
│  1.4 加载3个模拟账户 Excel → 用户交易记录                          │
│  1.5 数据清洗：时间对齐、缺失值处理、异常值剔除                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: 策略特征工程                                 │
│  从每个策略的净值+交易数据中提取以下特征向量：                       │
│  ┌──────────────────────┬─────────────────────────────────────┐  │
│  │ 特征类别              │ 具体特征                              │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 收益特征 (4维)        │ 累计收益率, 年化收益率,              │  │
│  │                      │ 平均月收益率, 正收益月占比             │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 风险特征 (4维)        │ 最大回撤, 日收益标准差(年化),         │  │
│  │                      │ VaR(5%), 下行波动率                   │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 风险调整收益 (2维)    │ 夏普比率, Calmar比率                 │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 交易行为 (4维)        │ 年化换手率, 平均持仓周期,             │  │
│  │                      │ 买卖对称性(B/S比), 交易频率           │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 持仓特征 (3维)        │ 持股集中度(HHI), 平均仓位比例,        │  │
│  │                      │ 现金占比波动                          │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 动量/反转特征 (2维)   │ 趋势强度(线性回归斜率),              │  │
│  │                      │ 回撤恢复天数                          │  │
│  └──────────────────────┴─────────────────────────────────────┤  │
│  合计: 19维原始特征向量 / 策略                                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: 用户特征工程                                 │
│  从每个用户的交易记录中提取相同维度的特征向量：                      │
│  ┌──────────────────────┬─────────────────────────────────────┐  │
│  │ 特征类别              │ 计算方式                              │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 收益特征 (4维)        │ 基于用户持仓的模拟收益率               │  │
│  │                      │ (以买入价/卖出价或最新价估算)           │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 风险特征 (4维)        │ 基于持仓股票的波动率加权               │  │
│  │                      │ (用策略净值波动代理市场波动)            │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 风险调整收益 (2维)    │ 同上计算                              │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 交易行为 (4维)        │ 直接从交易记录计算                     │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 持仓特征 (3维)        │ 从当前持仓和交易历史推断               │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 动量/反转特征 (2维)   │ 从交易时机推断                         │  │
│  └──────────────────────┴─────────────────────────────────────┤  │
│  合计: 19维原始特征向量 / 用户                                   │
│                                                                 │
│  注意：冷启动用户（交易少）使用收缩估计向市场均值收缩               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: 特征标准化 + PCA降维                         │
│  4.1 将策略和用户特征合并为一个矩阵 (n_strategies + n_users) × 19│
│  4.2 标准化：StandardScaler (零均值、单位方差)                     │
│  4.3 PCA：计算主成分，保留累积方差 > 85% 的前 k 个主成分            │
│  4.4 将策略和用户分别投影到 PCA 空间                              │
│  4.5 可视化：PCA前2-3主成分的散点图                               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: 匹配度计算                                   │
│  对每个用户 u，计算与所有策略 s 的相似度：                          │
│  ┌──────────────────────┬─────────────────────────────────────┐  │
│  │ 度量方法              │ 公式                                  │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 欧氏距离              │ d(u,s) = ||u - s||_2               │  │
│  │                      │ sim = 1/(1+d)                        │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 余弦相似度            │ sim = (u·s)/(||u||·||s||)           │  │
│  ├──────────────────────┼─────────────────────────────────────┤  │
│  │ 加权欧氏距离          │ d_w = sqrt(Σ w_i·(u_i-s_i)^2)       │  │
│  │                      │ 权重可按特征重要性设定                 │  │
│  └──────────────────────┴─────────────────────────────────────┘  │
│  输出: 每个用户的策略相似度排名                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 6: 结果输出与可解释性                            │
│  6.1 为每个用户输出 Top-3 推荐策略                                 │
│  6.2 维度级归因分析：哪个特征维度贡献最大                          │
│  6.3 生成弹窗话术示例                                              │
│  6.4 敏感性分析：不同PCA成分数、不同距离度量下的稳定性               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 7: 交叉验证（与绩效数据对比）                      │
│  7.1 将匹配结果与量化策略绩效.xlsx中的指标对比                      │
│  7.2 检验：推荐策略的收益率是否优于用户自选                         │
│  7.3 检验：推荐策略的回撤是否在用户承受范围内                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、关键实现细节

### 3.1 特征计算的具体公式

#### 收益特征
```python
# 累计收益率
cumulative_return = nav[-1] / nav[0] - 1

# 年化收益率
n_years = (date[-1] - date[0]).days / 365.25
annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1

# 平均月收益率
monthly_returns = nav.resample('M').last().pct_change().dropna()
mean_monthly_return = monthly_returns.mean()

# 正收益月占比
positive_month_ratio = (monthly_returns > 0).mean()
```

#### 风险特征
```python
# 最大回撤
running_max = nav.cummax()
max_drawdown = ((nav - running_max) / running_max).min()

# 日收益标准差 (年化)
daily_returns = nav.pct_change().dropna()
daily_vol_annual = daily_returns.std() * np.sqrt(252)

# VaR (5%)
var_5 = np.percentile(daily_returns, 5)

# 下行波动率
downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
```

#### 风险调整收益
```python
# 夏普比率 (假设无风险利率 2%)
sharpe = (annualized_return - 0.02) / daily_vol_annual

# Calmar比率
calmar = annualized_return / abs(max_drawdown)
```

#### 交易行为特征
```python
# 年化换手率
total_turnover = trades[trades['side'] == 'BUY']['amount'].sum()
avg_market_value = daily_value['market_value'].mean()
annual_turnover = total_turnover / avg_market_value / n_years

# 平均持仓周期 (配对买入-卖出)
holding_periods = sell_dates - buy_dates  # 按股票配对
avg_holding_period = holding_periods.mean()

# 买卖对称性
buy_amount = trades[trades['side'] == 'BUY']['amount'].sum()
sell_amount = abs(trades[trades['side'] == 'SELL']['amount'].sum())
buy_sell_ratio = buy_amount / sell_amount

# 交易频率
n_trades = len(trades)
trades_per_year = n_trades / n_years
```

#### 持仓特征
```python
# 持股集中度 (HHI)
# 按 trades 计算各股票的交易金额占比
stock_amount = trades.groupby('symbol')['amount'].sum().abs()
weights = stock_amount / stock_amount.sum()
hhi = (weights ** 2).sum()  # HHI 指数, 越大越集中

# 平均仓位比例
position_ratio = daily_value['market_value'] / daily_value['total_value']
avg_position = position_ratio.mean()

# 现金占比波动
cash_ratio = daily_value['cash'] / daily_value['total_value']
cash_ratio_std = cash_ratio.std()
```

#### 动量/反转特征
```python
# 趋势强度 (对数NAV做线性回归的斜率)
from scipy.stats import linregress
x = np.arange(len(nav))
y = np.log(nav)
slope, _, _, _, _ = linregress(x, y)
trend_strength = slope * 252  # 年化斜率

# 回撤恢复天数
# 计算每次回撤从谷底回到前高所需天数的中位数
recovery_days = []
peak = nav.iloc[0]
peak_date = dates[0]
in_drawdown = False
drawdown_start = None
for i, (val, dt) in enumerate(zip(nav, dates)):
    if val > peak:
        if in_drawdown:
            recovery_days.append((dt - drawdown_start).days)
            in_drawdown = False
        peak = val
        peak_date = dt
    elif val < peak and not in_drawdown:
        in_drawdown = True
        drawdown_start = peak_date
avg_recovery_days = np.median(recovery_days) if recovery_days else len(nav)
```

### 3.2 用户特征计算的特殊处理

用户没有净值曲线，需要用交易记录间接推断：

```python
# 用户的"收益特征"：基于交易盈亏推断
# 对每笔买入，找到对应的卖出，计算收益率
for each stock in user_trades:
    buy_records = trades[side == 'BUY']
    sell_records = trades[side == 'SELL']
    # FIFO配对
    for buy, sell in pair(buy_records, sell_records):
        trade_return = (sell.price - buy.price) / buy.price
# 平均交易收益率作为用户的收益特征

# 用户的"风险特征"：基于持仓股票的波动率
# 由于没有个股日频数据，用策略净值波动率作为市场波动的代理
# 或者用用户交易金额分布的离散度作为风险偏好代理

# 用户没有 market_value / total_value 数据
# 用 买入金额 / (买入金额 + 现金余额) 估算仓位比例
# 现金余额用初始资金 - 净买入金额估算
```

### 3.3 PCA 实施细节

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 合并特征矩阵: 7个策略 + 3个用户 = 10个样本, 19个特征
X = np.vstack([strategy_features, user_features])  # shape: (10, 19)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.85)  # 保留85%方差
X_pca = pca.fit_transform(X_scaled)

# 查看各主成分的解释方差
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.2%}")
print(f"Cumulative: {pca.explained_variance_ratio_.sum():.2%}")

# 策略和用户分别投影
strategy_pca = X_pca[:7]  # 前7行
user_pca = X_pca[7:]      # 后3行
```

### 3.4 匹配度计算

```python
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

# 欧氏距离
dist_matrix = cdist(user_pca, strategy_pca, metric='euclidean')
sim_euclidean = 1 / (1 + dist_matrix)

# 余弦相似度
sim_cosine = cosine_similarity(user_pca, strategy_pca)

# 加权欧氏距离 (可自定义权重)
# 权重基于PCA解释方差: 每个主成分的权重 = 其解释方差占比
weights = pca.explained_variance_ratio_
dist_weighted = cdist(
    user_pca * np.sqrt(weights),
    strategy_pca * np.sqrt(weights),
    metric='euclidean'
)
sim_weighted = 1 / (1 + dist_weighted)
```

### 3.5 可解释输出

```python
# 对每个用户的Top-1推荐，给出维度级归因
def explain_match(user_idx, strategy_idx, user_features, strategy_features, feature_names):
    """计算每个特征维度对匹配度的贡献"""
    diff = np.abs(user_features[strategy_idx] - strategy_features[strategy_idx])
    # 归一化差异
    max_vals = np.maximum(np.abs(user_features[strategy_idx]), np.abs(strategy_features[strategy_idx]))
    normalized_diff = diff / (max_vals + 1e-8)
    # 排序贡献
    contributions = sorted(zip(feature_names, normalized_diff), key=lambda x: x[1])
    return contributions
```

---

## 四、代码文件结构

```
pipeline.py              # 完整可执行的数据分析流水线
├── load_data()          # 数据加载
├── extract_strategy_features()  # 策略特征提取
├── extract_user_features()      # 用户特征提取
├── build_feature_matrix()       # 构建特征矩阵
├── pca_transform()              # PCA降维
├── compute_similarity()         # 匹配度计算
├── generate_recommendations()   # 生成推荐结果
├── plot_results()               # 可视化
└── main()               # 主流程

output/
├── strategy_features.csv        # 策略特征表
├── user_features.csv            # 用户特征表
├── pca_results.csv              # PCA投影结果
├── recommendations.json         # 推荐结果
├── pca_scatter.png              # PCA散点图
└── similarity_heatmap.png       # 相似度热力图
```

---

## 五、扩展方向

### 5.1 行业维度匹配
如果有股票的申万行业分类数据，可以：
1. 统计每个策略在各行业的持仓市值占比 → 28维行业分布向量
2. 统计每个用户在各行业的交易金额占比 → 28维行业分布向量
3. 用 JS 散度计算行业相似度

### 5.2 滚动窗口匹配
- 不使用全量历史数据，而是用最近 N 天（60/120/252天）的数据
- 观察匹配结果随时间的变化，评估匹配稳定性

### 5.3 冷启动处理
- 对于没有交易数据的新用户，使用问卷结果映射到特征空间
- 或推荐"默认策略"（如宽指配置策略，风险收益居中）

### 5.4 权重优化
- 初始权重基于金融常识设定
- 如果有用户反馈数据（如订阅转化率），可以用贝叶斯优化调整权重

---

## 六、验证方案

| 验证方法 | 具体内容 |
|----------|----------|
| 内部一致性 | 改变PCA成分数（3/5/8），推荐结果是否稳定 |
| 度量对比 | 欧氏距离 vs 余弦相似度 vs 加权距离的排名相关性 |
| 绩效对照 | 推荐策略的收益是否优于用户自选收益 |
| 回测验证 | 用前2年数据匹配，后1年验证推荐策略的表现 |
