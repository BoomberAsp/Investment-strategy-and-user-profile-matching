"""
用户画像-投资策略匹配：完整数据分析流水线 (V2)
================================================
基于 三层特征体系 + PCA + 欧氏距离/余弦相似度的策略推荐方案

特征架构:
  第一层: 交易行为特征 (6维) — 从交易流水直接计算，零噪声
  第二层: 资产偏好特征 (3维) — 刻画"用户喜欢买什么"
  第三层: 风险代理特征 (3维) — 从交易模式反推风险偏好

匹配:
  beta 超参数控制行为特征 vs 非行为特征的权重
  PCA 降维后在特征空间中计算欧氏距离/余弦相似度
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import linregress
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

STRATEGY_DATA_DIR = (
    BASE_DIR / '净值_交易_资金及字段说明（相关性数据分析）' / 'products_export_20260518_163122'
)
PERF_FILE_1 = BASE_DIR / '量化策略绩效-1.xlsx'
PERF_FILE_2 = BASE_DIR / '量化策略绩效-2.xlsx'
USER_FILES = {
    'A': BASE_DIR / '模拟账户A的记录.xlsx',
    'B': BASE_DIR / '模拟账户B的记录.xlsx',
    'C': BASE_DIR / '模拟账户C的记录.xlsx',
}

# beta 超参数: 控制行为特征 vs 非行为特征的权重
# beta=0.0: 仅关注资产偏好+风险代理（用户信任产品推荐）
# beta=1.0: 仅关注行为特征（用户偏好与自己一致的策略）
# beta=0.5: 均衡
BETA = 0.5

# 径向惩罚余弦的 lambda 超参数
# lambda=0: 退化为纯余弦相似度
# lambda→∞: 退化为纯模长匹配
LAMBDA = 1.0

# 特征分类定义
# 第一层: 交易行为特征 (6维)
BEHAVIOR_FEATURES = [
    'holding_period',       # 平均持仓周期(天)
    'turnover_rate',        # 交易换手率(笔/天)
    'buy_sell_ratio',       # 买卖对称性(买入金额/卖出金额)
    'hhi_concentration',    # 持仓集中度(HHI指数)
    'disposition_effect',   # 处置效应系数
    'positive_trade_ratio', # 正收益交易占比(胜率)
]

# 第二层: 资产偏好特征 (3维)
ASSET_PREF_FEATURES = [
    'etf_ratio',            # ETF交易占比
    'avg_price_preference', # 价格区间偏好(买入均价)
    'position_uniformity',  # 分仓均匀度(单笔金额CV的倒数)
]

# 第三层: 风险代理特征 (3维)
RISK_PROXY_FEATURES = [
    'avg_loss_magnitude',   # 最大回撤代理(平均单笔亏损幅度)
    'vol_preference',       # 波动偏好(交易标的价格波动代理)
    'trend_preference',     # 趋势偏好(追涨杀跌倾向)
]

# 全部匹配特征 (12维)
MATCH_FEATURES = BEHAVIOR_FEATURES + ASSET_PREF_FEATURES + RISK_PROXY_FEATURES

# 特征分组标签（用于加权）
FEATURE_GROUPS = {}
for f in BEHAVIOR_FEATURES:
    FEATURE_GROUPS[f] = 'behavior'
for f in ASSET_PREF_FEATURES:
    FEATURE_GROUPS[f] = 'asset_pref'
for f in RISK_PROXY_FEATURES:
    FEATURE_GROUPS[f] = 'risk_proxy'

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02


# ============================================================
# STEP 1: 数据加载
# ============================================================
def load_strategy_nav_data():
    """加载所有策略的净值数据"""
    strategy_nav = {}
    for dir_path in sorted(STRATEGY_DATA_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dv_file = dir_path / 'daily_value.csv'
        if not dv_file.exists():
            continue
        strategy_id = dir_path.name
        df = pd.read_csv(dv_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        strategy_nav[strategy_id] = df
    print(f"[STEP 1] Loaded NAV data for {len(strategy_nav)} strategies")
    return strategy_nav


def load_strategy_trades():
    """加载所有策略的交易记录"""
    strategy_trades = {}
    for dir_path in sorted(STRATEGY_DATA_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        trades_file = dir_path / 'trades.csv'
        if not trades_file.exists():
            continue
        strategy_id = dir_path.name
        df = pd.read_csv(trades_file)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        strategy_trades[strategy_id] = df
    print(f"[STEP 1] Loaded trades data for {len(strategy_trades)} strategies")
    return strategy_trades


def load_user_data():
    """加载用户交易记录"""
    user_data = {}
    for user_id, file_path in USER_FILES.items():
        if not file_path.exists():
            continue
        df = pd.read_excel(file_path, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=[df.columns[0]])
        user_data[user_id] = df
        print(f"[STEP 1] User {user_id}: {len(df)} trades, {df.iloc[:, 2].nunique()} stocks")
    return user_data


# ============================================================
# 工具函数
# ============================================================
def parse_user_trades(user_df):
    """
    解析用户交易数据，返回标准化结构。
    由于Excel列名是中文且编码不确定，用列位置解析。

    Returns:
        DataFrame with columns: trade_date, action_type, symbol, price, quantity, amount
    """
    date_col = user_df.columns[0]
    action_col = user_df.columns[1]
    sym_col = user_df.columns[2]
    price_col = user_df.columns[4]
    qty_col = user_df.columns[5]

    trades = pd.DataFrame({
        'trade_date': user_df[date_col],
        'action_type': user_df[action_col],
        'symbol': user_df[sym_col].astype(str),
        'price': user_df[price_col],
        'quantity': user_df[qty_col],
        'amount': user_df[price_col] * user_df[qty_col],
    })
    trades['is_buy'] = trades['action_type'].str.contains('买入', na=False)
    trades['is_sell'] = trades['action_type'].str.contains('卖出', na=False)
    return trades.sort_values('trade_date').reset_index(drop=True)


def fifo_pair_trades(trades):
    """
    FIFO 配对：为每笔买入找到对应的卖出，计算单笔收益率。

    Returns:
        closed_trades: list of dicts with {symbol, buy_price, sell_price, return, hold_days}
        open_positions: dict of symbol -> remaining qty from buys not yet sold
    """
    buy_queue = {}  # symbol -> list of {price, date, qty}
    closed_trades = []

    for _, t in trades.iterrows():
        sym = t['symbol']
        if t['is_buy']:
            if sym not in buy_queue:
                buy_queue[sym] = []
            buy_queue[sym].append({
                'price': t['price'],
                'date': t['trade_date'],
                'qty': t['quantity'],
            })
        elif t['is_sell']:
            remaining_sell = t['quantity']
            if sym not in buy_queue or len(buy_queue[sym]) == 0:
                # 卖出没有对应买入记录的 → 期初底仓，跳过
                continue
            for buy_rec in buy_queue[sym]:
                if remaining_sell <= 0:
                    break
                fill_qty = min(remaining_sell, buy_rec['qty'])
                if fill_qty <= 0:
                    continue
                ret = (t['price'] - buy_rec['price']) / buy_rec['price'] if buy_rec['price'] > 0 else 0
                hold_days = (t['trade_date'] - buy_rec['date']).days
                closed_trades.append({
                    'symbol': sym,
                    'buy_price': buy_rec['price'],
                    'sell_price': t['price'],
                    'return': ret,
                    'hold_days': hold_days,
                    'qty': fill_qty,
                    'amount': t['price'] * fill_qty,
                })
                buy_rec['qty'] -= fill_qty
                remaining_sell -= fill_qty
            # 清理已用完的买入记录
            buy_queue[sym] = [r for r in buy_queue[sym] if r['qty'] > 0]

    # 剩余未平仓头寸
    open_positions = {}
    for sym, recs in buy_queue.items():
        total_qty = sum(r['qty'] for r in recs)
        if total_qty > 0:
            avg_price = sum(r['price'] * r['qty'] for r in recs) / total_qty
            open_positions[sym] = {'qty': total_qty, 'avg_price': avg_price}

    return closed_trades, open_positions


# ============================================================
# STEP 2: 策略特征提取（三层特征体系）
# ============================================================
def extract_strategy_behavior_features(trades_df, nav_df):
    """策略的交易行为特征 (6维)"""
    n_days = max((trades_df['trade_date'].max() - trades_df['trade_date'].min()).days, 1)
    n_trades = len(trades_df)

    # 标准化列名
    t = trades_df.copy()
    if '_side' in t.columns and 'side' not in t.columns:
        t['side'] = t['_side']
    if 'quantity' not in t.columns and 'qty' in t.columns:
        t['quantity'] = t['qty']

    t['is_buy'] = t['side'] == 'BUY'
    t['is_sell'] = t['side'] == 'SELL'

    # 1. 持仓周期 + 正收益占比 (FIFO配对)
    closed, _ = fifo_pair_trades(t)
    if closed:
        holding_periods = [ct['hold_days'] for ct in closed if ct['hold_days'] > 0]
        avg_holding_period = np.median(holding_periods) if holding_periods else 30
        positive_ratio = np.mean([ct['return'] > 0 for ct in closed])
    else:
        avg_holding_period = 30
        positive_ratio = 0.5

    # 2. 换手率
    turnover_rate = n_trades / n_days

    # 3. 买卖对称性
    buy_amt = t[t['is_buy']]['amount'].sum()
    sell_amt = abs(t[t['is_sell']]['amount'].sum())
    bs_ratio = buy_amt / sell_amt if sell_amt > 0 else 2.0

    # 4. HHI集中度
    stock_amount = t.groupby('symbol')['amount'].apply(lambda x: x.abs().sum())
    weights = stock_amount / stock_amount.sum()
    hhi = (weights ** 2).sum()

    # 5. 处置效应系数
    if closed:
        n_profit = sum(1 for ct in closed if ct['return'] > 0)
        n_loss = sum(1 for ct in closed if ct['return'] < 0)
        disposition = (n_profit / max(len(closed), 1)) / max((n_loss / max(len(closed), 1)), 0.01)
    else:
        disposition = 1.0

    return {
        'holding_period': avg_holding_period,
        'turnover_rate': turnover_rate,
        'buy_sell_ratio': bs_ratio,
        'hhi_concentration': hhi,
        'disposition_effect': disposition,
        'positive_trade_ratio': positive_ratio,
    }


def extract_strategy_asset_pref_features(trades_df):
    """策略的资产偏好特征 (3维)"""
    # 1. ETF占比
    all_names = trades_df.get('name', pd.Series([]))
    if len(all_names) > 0:
        etf_count = all_names.str.contains('ETF', na=False).sum()
        etf_ratio = etf_count / len(trades_df)
    else:
        etf_ratio = 0.0

    # 2. 价格区间偏好
    avg_price = trades_df['price'].mean() if len(trades_df) > 0 else 0

    # 3. 分仓均匀度 (单笔金额CV的倒数)
    amounts = trades_df['amount'].abs()
    if len(amounts) > 1 and amounts.mean() > 0:
        cv = amounts.std() / amounts.mean()
        position_uniformity = 1 / (1 + cv)  # [0, 1]，越大越均匀
    else:
        position_uniformity = 0.5

    return {
        'etf_ratio': etf_ratio,
        'avg_price_preference': avg_price,
        'position_uniformity': position_uniformity,
    }


def extract_strategy_risk_proxy_features(trades_df, nav_df):
    """策略的风险代理特征 (3维)"""
    # 标准化列名
    t = trades_df.copy()
    if '_side' in t.columns and 'side' not in t.columns:
        t['side'] = t['_side']
    if 'quantity' not in t.columns and 'qty' in t.columns:
        t['quantity'] = t['qty']
    t['is_buy'] = t['side'] == 'BUY'
    t['is_sell'] = t['side'] == 'SELL'

    # 1. 最大回撤代理: 平均单笔亏损幅度
    closed, _ = fifo_pair_trades(t)
    loss_trades = [t for t in closed if t['return'] < 0]
    if loss_trades:
        avg_loss_magnitude = np.mean([abs(t['return']) for t in loss_trades])
    else:
        avg_loss_magnitude = 0.02  # 默认2%

    # 2. 波动偏好: 从NAV计算
    if 'nav' in nav_df.columns and len(nav_df) > 10:
        daily_rets = nav_df['nav'].pct_change().dropna()
        vol_preference = daily_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        vol_preference = 0.15

    # 3. 趋势偏好: 净值趋势强度
    if 'nav' in nav_df.columns and len(nav_df) > 10:
        nav = nav_df['nav'].values
        x = np.arange(len(nav))
        y = np.log(np.clip(nav, 1e-8, None))
        slope, _, _, _, _ = linregress(x, y)
        trend_preference = slope * TRADING_DAYS_PER_YEAR
    else:
        trend_preference = 0

    return {
        'avg_loss_magnitude': avg_loss_magnitude,
        'vol_preference': vol_preference,
        'trend_preference': trend_preference,
    }


def extract_all_strategy_features(strategy_nav, strategy_trades):
    """提取所有策略的三层特征向量"""
    features = {}
    for strategy_id in sorted(strategy_nav.keys()):
        nav_df = strategy_nav[strategy_id]
        trades_df = strategy_trades.get(strategy_id, pd.DataFrame())
        if trades_df.empty:
            continue

        behavior = extract_strategy_behavior_features(trades_df, nav_df)
        asset_pref = extract_strategy_asset_pref_features(trades_df)
        risk = extract_strategy_risk_proxy_features(trades_df, nav_df)

        features[strategy_id] = {**behavior, **asset_pref, **risk}

        nav_end = nav_df['nav'].iloc[-1]
        print(f"  Strategy {strategy_id}: "
              f"NAV={nav_end:.2f}, "
              f"holding={behavior['holding_period']:.0f}d, "
              f"turnover={behavior['turnover_rate']:.2f}/day, "
              f"etf_ratio={asset_pref['etf_ratio']:.2f}")

    return features


# ============================================================
# STEP 3: 用户特征提取（三层特征体系）
# ============================================================
def extract_user_behavior_features(trades):
    """用户的交易行为特征 (6维)"""
    n_trades = len(trades)
    if n_trades == 0:
        return {f: 0.0 for f in BEHAVIOR_FEATURES}

    n_days = max((trades['trade_date'].max() - trades['trade_date'].min()).days, 1)

    # FIFO配对
    closed, open_pos = fifo_pair_trades(trades)

    # 1. 持仓周期
    if closed:
        holding_periods = [t['hold_days'] for t in closed if t['hold_days'] > 0]
        avg_holding_period = np.median(holding_periods) if holding_periods else 30
        positive_ratio = np.mean([t['return'] > 0 for t in closed])
    else:
        avg_holding_period = 30
        positive_ratio = 0.5

    # 2. 换手率 (笔/天)
    turnover_rate = n_trades / n_days

    # 3. 买卖对称性
    buy_amt = trades[trades['is_buy']]['amount'].sum()
    sell_amt = abs(trades[trades['is_sell']]['amount'].sum())
    bs_ratio = buy_amt / sell_amt if sell_amt > 0 else 2.0

    # 4. HHI集中度
    stock_amount = trades.groupby('symbol')['amount'].apply(lambda x: x.abs().sum())
    if stock_amount.sum() > 0:
        weights = stock_amount / stock_amount.sum()
        hhi = (weights ** 2).sum()
    else:
        hhi = 1.0

    # 5. 处置效应系数
    if closed:
        n_profit = sum(1 for t in closed if t['return'] > 0)
        n_loss = sum(1 for t in closed if t['return'] < 0)
        # 简化: 已实现盈亏比
        disposition = (n_profit / max(len(closed), 1)) / max((n_loss / max(len(closed), 1)), 0.01)
    else:
        disposition = 1.0

    return {
        'holding_period': avg_holding_period,
        'turnover_rate': turnover_rate,
        'buy_sell_ratio': bs_ratio,
        'hhi_concentration': hhi,
        'disposition_effect': disposition,
        'positive_trade_ratio': positive_ratio,
    }


def extract_user_asset_pref_features(trades):
    """用户的资产偏好特征 (3维)"""
    if trades.empty:
        return {f: 0.0 for f in ASSET_PREF_FEATURES}

    # 1. ETF占比 (从symbol格式判断: ETF通常是6位代码中的特定范围)
    # 简化: 用股票代码长度和格式判断
    symbols = trades['symbol'].astype(str)
    # ETF代码通常以 15/51 开头(6位) 或 12 开头
    etf_mask = symbols.str.match(r'^(15|51|56|58|12)')
    etf_ratio = etf_mask.sum() / len(trades) if len(trades) > 0 else 0.0

    # 2. 价格区间偏好
    avg_price = trades['price'].mean()

    # 3. 分仓均匀度
    amounts = trades['amount'].abs()
    if len(amounts) > 1 and amounts.mean() > 0:
        cv = amounts.std() / amounts.mean()
        position_uniformity = 1 / (1 + cv)
    else:
        position_uniformity = 0.5

    return {
        'etf_ratio': etf_ratio,
        'avg_price_preference': avg_price,
        'position_uniformity': position_uniformity,
    }


def extract_user_risk_proxy_features(trades):
    """用户的风险代理特征 (3维)"""
    if trades.empty:
        return {f: 0.0 for f in RISK_PROXY_FEATURES}

    closed, _ = fifo_pair_trades(trades)

    # 1. 最大回撤代理: 平均单笔亏损幅度
    loss_trades = [t for t in closed if t['return'] < 0]
    if loss_trades:
        avg_loss_magnitude = np.mean([abs(t['return']) for t in loss_trades])
    else:
        # 没有亏损交易 → 低回撤偏好
        avg_loss_magnitude = 0.02

    # 2. 波动偏好: 从交易标的价格分布推断
    # 高价股通常波动更大，用买入价格的标准差代理
    buy_prices = trades[trades['is_buy']]['price']
    if len(buy_prices) > 1:
        vol_preference = buy_prices.std() / buy_prices.mean()  # 价格变异系数
    else:
        vol_preference = 0.3

    # 3. 趋势偏好: 追涨倾向
    # 简化: 如果用户的买入均价高于该股票首次出现时的价格，说明追涨
    trend_signals = []
    for sym in trades['symbol'].unique():
        sym_trades = trades[trades['symbol'] == sym].sort_values('trade_date')
        first_price = sym_trades.iloc[0]['price']
        buy_prices_sym = sym_trades[sym_trades['is_buy']]['price']
        if len(buy_prices_sym) > 0 and first_price > 0:
            avg_buy = buy_prices_sym.mean()
            trend_signals.append((avg_buy - first_price) / first_price)
    trend_preference = np.mean(trend_signals) if trend_signals else 0

    return {
        'avg_loss_magnitude': avg_loss_magnitude,
        'vol_preference': vol_preference,
        'trend_preference': trend_preference,
    }


def extract_all_user_features(user_data):
    """提取所有用户的三层特征向量"""
    user_features = {}
    for user_id in sorted(user_data.keys()):
        user_df = user_data[user_id]
        trades = parse_user_trades(user_df)

        behavior = extract_user_behavior_features(trades)
        asset_pref = extract_user_asset_pref_features(trades)
        risk = extract_user_risk_proxy_features(trades)

        features = {**behavior, **asset_pref, **risk}
        user_features[user_id] = features

        print(f"  User {user_id}: "
              f"trades={len(trades)}, "
              f"holding={behavior['holding_period']:.0f}d, "
              f"turnover={behavior['turnover_rate']:.3f}/day, "
              f"etf_ratio={asset_pref['etf_ratio']:.2f}, "
              f"win_rate={behavior['positive_trade_ratio']:.2%}")

    return user_features


# ============================================================
# STEP 4: 特征加权 + PCA
# ============================================================
def apply_beta_weighting(feature_dict, beta):
    """
    对特征应用 beta 加权:
    - 行为特征 × beta
    - 非行为特征 × (1 - beta)
    """
    weighted = {}
    for fname, fval in feature_dict.items():
        if FEATURE_GROUPS.get(fname) == 'behavior':
            weighted[fname] = fval * beta
        else:
            weighted[fname] = fval * (1 - beta)
    return weighted


def build_feature_matrix(strategy_features, user_features, beta=BETA):
    """
    构建统一特征矩阵，应用 beta 加权。

    流程:
    1. 对每个实体（策略/用户）的特征应用 beta 加权
    2. 合并为矩阵
    3. 标准化
    4. PCA
    """
    strategy_ids = sorted(strategy_features.keys())
    user_ids = sorted(user_features.keys())

    # 应用 beta 加权
    strategy_weighted = {sid: apply_beta_weighting(strategy_features[sid], beta)
                         for sid in strategy_ids}
    user_weighted = {uid: apply_beta_weighting(user_features[uid], beta)
                     for uid in user_ids}

    # 构建特征矩阵
    S = np.array([[strategy_weighted[sid][f] for f in MATCH_FEATURES] for sid in strategy_ids])
    U = np.array([[user_weighted[uid][f] for f in MATCH_FEATURES] for uid in user_ids])
    X = np.vstack([S, U])

    labels = strategy_ids + user_ids
    types = ['strategy'] * len(strategy_ids) + ['user'] * len(user_ids)

    # 处理异常值
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"[STEP 4] Feature matrix: {X.shape}, beta={beta}")
    print(f"  Strategies: {len(strategy_ids)}, Users: {len(user_ids)}")

    return X, labels, types, strategy_ids, user_ids


def apply_pca(X, labels, types, n_components=None):
    """
    PCA降维（保留模长信息）

    1. 标准化数据确定主成分方向
    2. 去均值但不缩放的数据做投影，保留模长
    """
    scaler_std = StandardScaler()
    X_scaled = scaler_std.fit_transform(X)

    if n_components is None:
        pca = PCA(n_components=0.90)
    else:
        pca = PCA(n_components=n_components)

    pca.fit(X_scaled)

    # 投影: 去均值但不缩放
    scaler_center = StandardScaler(with_std=False)
    X_centered = scaler_center.fit_transform(X)
    X_centered = np.nan_to_num(X_centered, nan=0.0)

    X_pca = X_centered @ pca.components_.T

    print(f"[STEP 4] PCA: {pca.n_components_} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    n_strategies = sum(1 for t in types if t == 'strategy')
    strategy_norms = np.linalg.norm(X_pca[:n_strategies], axis=1)
    user_norms = np.linalg.norm(X_pca[n_strategies:], axis=1)
    print(f"  Strategy norms: [{strategy_norms.min():.1f}, {strategy_norms.max():.1f}]")
    user_labels = [l for l, t in zip(labels, types) if t == 'user']
    for ul, un in zip(user_labels, user_norms):
        print(f"  User {ul} norm: {un:.1f}")

    return X_pca, pca, scaler_std


# ============================================================
# STEP 5: 匹配度计算
# ============================================================
def compute_radial_penalty_cosine(user_pca, strategy_pca, lam=LAMBDA):
    """
    径向惩罚余弦相似度 (Radial-Penalty Cosine)

    sim(u, s) = cos(u, s) × exp(-λ × |log(‖u‖/‖s‖)|)

    物理含义:
    - 方向一致 + 模长接近 → 满分
    - 方向一致 + 模长悬殊 → 降权
    - 方向相反 → 负值
    """
    # 余弦部分
    cos_sim = cosine_similarity(user_pca, strategy_pca)

    # 模长部分
    user_norms = np.linalg.norm(user_pca, axis=1, keepdims=True)  # (n_users, 1)
    strategy_norms = np.linalg.norm(strategy_pca, axis=1, keepdims=True).T  # (1, n_strategies)

    # 防止除零
    user_norms = np.clip(user_norms, 1e-10, None)
    strategy_norms = np.clip(strategy_norms, 1e-10, None)

    # log 比: |log(‖u‖/‖s‖)|
    log_ratio = np.abs(np.log(user_norms / strategy_norms))  # (n_users, n_strategies)

    # 径向惩罚因子: exp(-λ × |log_ratio|)
    radial_penalty = np.exp(-lam * log_ratio)

    return cos_sim * radial_penalty


def compute_similarity(X_pca, n_strategies):
    """计算用户-策略相似度"""
    strategy_pca = X_pca[:n_strategies]
    user_pca = X_pca[n_strategies:]

    dist_euclidean = cdist(user_pca, strategy_pca, metric='euclidean')
    sim_euclidean = 1 / (1 + dist_euclidean)

    sim_cosine = cosine_similarity(user_pca, strategy_pca)

    sim_rp = compute_radial_penalty_cosine(user_pca, strategy_pca, lam=LAMBDA)

    return {
        'radial_penalty': {'similarity': sim_rp, 'distance': 1 - sim_rp,
                            'metric_name': f'Radial-Penalty Cosine (λ={LAMBDA})'},
        'cosine': {'similarity': sim_cosine, 'distance': 1 - sim_cosine,
                    'metric_name': 'Cosine Similarity'},
        'euclidean': {'similarity': sim_euclidean, 'distance': dist_euclidean,
                       'metric_name': 'Euclidean Distance'},
    }


def rank_strategies(similarity_matrix, strategy_ids, user_ids, eligible_map=None):
    """对每个用户排序策略，可选地应用收益过滤"""
    rankings = {}
    for i, user_id in enumerate(user_ids):
        sims = similarity_matrix[i]
        ranked_indices = np.argsort(-sims)

        # 应用过滤
        if eligible_map:
            eligible = eligible_map.get(user_id, set(strategy_ids))
            filtered = [idx for idx in ranked_indices if strategy_ids[idx] in eligible]
        else:
            filtered = ranked_indices

        rankings[user_id] = []
        for rank_pos, idx in enumerate(filtered):
            rankings[user_id].append({
                'strategy': strategy_ids[idx],
                'similarity': float(sims[idx]),
                'rank': rank_pos + 1,
            })

    return rankings


# ============================================================
# STEP 6: 可解释输出
# ============================================================
def generate_explanation(user_id, recommendation, user_feats, strategy_feats):
    """生成可解释的推荐说明"""
    strategy_id = recommendation['strategy']
    sim = recommendation['similarity']

    diffs = {}
    for fname in MATCH_FEATURES:
        u_val = user_feats.get(fname, 0)
        s_val = strategy_feats.get(fname, 0)
        max_val = max(abs(u_val), abs(s_val), 1e-8)
        diff = abs(u_val - s_val) / max_val
        diffs[fname] = diff

    top_similar = sorted(diffs.items(), key=lambda x: x[1])[:3]
    top_different = sorted(diffs.items(), key=lambda x: -x[1])[:2]

    # 特征名中文映射
    name_map = {
        'holding_period': '持仓周期', 'turnover_rate': '换手率',
        'buy_sell_ratio': '买卖对称性', 'hhi_concentration': '持仓集中度',
        'disposition_effect': '处置效应', 'positive_trade_ratio': '胜率',
        'etf_ratio': 'ETF偏好', 'avg_price_preference': '价格偏好',
        'position_uniformity': '分仓均匀度', 'avg_loss_magnitude': '亏损幅度',
        'vol_preference': '波动偏好', 'trend_preference': '趋势偏好',
    }

    explanation = {
        'customer_id': user_id,
        'matched_strategy': strategy_id,
        'overall_similarity': round(sim, 4),
        'most_similar_dimensions': [
            {'feature': name_map.get(f, f), 'diff': round(d, 4)} for f, d in top_similar
        ],
        'most_different_dimensions': [
            {'feature': name_map.get(f, f), 'diff': round(d, 4)} for f, d in top_different
        ],
        'beta': BETA,
        'popup_text': (
            f"策略 {strategy_id} 与您的投资风格匹配度 {sim:.1%}。"
            f"您在 {'、'.join([name_map.get(f, f) for f, _ in top_similar])} "
            f"方面与该策略风格最为接近，建议进一步了解该策略。"
        ),
    }
    return explanation


# ============================================================
# STEP 7: 可视化
# ============================================================
def plot_pca_scatter(X_pca, labels, types, strategy_ids, user_ids, pca, save_path=None):
    """PCA散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_strategy = plt.cm.Set1(np.linspace(0, 1, len(strategy_ids)))

    ax = axes[0]
    for i, sid in enumerate(strategy_ids):
        idx = labels.index(sid)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[colors_strategy[i]],
                   s=150, marker='s', label=sid[:20], edgecolors='black', linewidths=0.5, zorder=5)
    for uid in user_ids:
        idx = labels.index(uid)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c='red',
                   s=200, marker='*', label=f'User {uid}', edgecolors='darkred', linewidths=1, zorder=10)
        ax.annotate(f'User {uid}', (X_pca[idx, 0], X_pca[idx, 1]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA: PC1 vs PC2 (beta={BETA})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    if X_pca.shape[1] >= 3:
        ax = axes[1]
        for i, sid in enumerate(strategy_ids):
            idx = labels.index(sid)
            ax.scatter(X_pca[idx, 0], X_pca[idx, 2], c=[colors_strategy[i]],
                       s=150, marker='s', label=sid[:20], edgecolors='black', linewidths=0.5, zorder=5)
        for uid in user_ids:
            idx = labels.index(uid)
            ax.scatter(X_pca[idx, 0], X_pca[idx, 2], c='red',
                       s=200, marker='*', label=f'User {uid}', edgecolors='darkred', linewidths=1, zorder=10)
            ax.annotate(f'User {uid}', (X_pca[idx, 0], X_pca[idx, 2]),
                        textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('PCA: PC1 vs PC3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {save_path}")
    plt.close()


def plot_similarity_heatmap(results, strategy_ids, user_ids, save_path=None):
    """相似度热力图（支持任意数量的度量方法）"""
    n_metrics = len(results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    for idx, (metric_name, data) in enumerate(results.items()):
        ax = axes[idx]
        sim = data['similarity']
        sns.heatmap(sim, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[s[:15] for s in strategy_ids],
                    yticklabels=[f'User {u}' for u in user_ids],
                    ax=ax, cbar_kws={'label': 'Similarity'})
        ax.set_title(data['metric_name'])
        ax.set_xlabel('Strategy')
        ax.set_ylabel('User')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {save_path}")
    plt.close()


def plot_feature_importance(pca, feature_names, save_path=None):
    """PCA特征载荷图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    loadings = pca.components_[:3]
    n_features = loadings.shape[1]
    x = np.arange(n_features)
    width = 0.25

    ax.bar(x - width, loadings[0], width, label=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.bar(x, loadings[1], width, label=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    if loadings.shape[0] > 2:
        ax.bar(x + width, loadings[2], width, label=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Loading')
    ax.set_title('PCA Feature Loadings (Top 3 Components)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {save_path}")
    plt.close()


def plot_strategy_nav(strategy_nav, save_path=None):
    """策略净值曲线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy_id, df in strategy_nav.items():
        ax.plot(df['date'], df['nav'], label=strategy_id[:20], linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.set_title('Strategy NAV Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {save_path}")
    plt.close()


def plot_feature_comparison(strategy_features, user_features, save_path=None):
    """特征雷达图: 对比策略和用户的特征分布"""
    n_features = len(MATCH_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))

    for ax_idx, user_id in enumerate(sorted(user_features.keys())):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]

        user_vals = [user_features[user_id][f] for f in MATCH_FEATURES]
        # 归一化到 [0, 1] 用于雷达图
        all_vals = [strategy_features[s][f] for s in strategy_features for f in MATCH_FEATURES]
        all_vals += [user_features[u][f] for u in user_features for f in MATCH_FEATURES]
        min_val = min(all_vals)
        max_val = max(all_vals)
        range_val = max_val - min_val if max_val != min_val else 1

        user_normed = [(v - min_val) / range_val for v in user_vals]
        user_normed += user_normed[:1]

        ax.plot(angles, user_normed, 'o-', linewidth=2, label=f'User {user_id}')
        ax.fill(angles, user_normed, alpha=0.1)

        # 画一个策略的平均作为参考
        avg_strategy = [np.mean([strategy_features[s][f] for s in strategy_features]) for f in MATCH_FEATURES]
        avg_normed = [(v - min_val) / range_val for v in avg_strategy]
        avg_normed += avg_normed[:1]
        ax.plot(angles, avg_normed, '--', linewidth=1, label='Avg Strategy', alpha=0.5)

        short_names = [f[:10] for f in MATCH_FEATURES]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_names, fontsize=6)
        ax.set_title(f'User {user_id} Profile', pad=15)
        ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {save_path}")
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("  User-Investment Strategy Matching Pipeline (V2)")
    print("  Three-layer Feature Architecture + PCA + beta Weighting")
    print("=" * 70)

    # STEP 1: 数据加载
    print("\n>>> STEP 1: Loading data...")
    strategy_nav = load_strategy_nav_data()
    strategy_trades = load_strategy_trades()
    user_data = load_user_data()

    # STEP 2: 策略特征提取
    print("\n>>> STEP 2: Extracting strategy features (3-layer)...")
    strategy_features = extract_all_strategy_features(strategy_nav, strategy_trades)

    sf_df = pd.DataFrame(strategy_features).T
    sf_df.index.name = 'strategy_id'
    sf_df.to_csv(OUTPUT_DIR / 'strategy_features_v2.csv')
    print(f"[SAVE] Strategy features saved to {OUTPUT_DIR / 'strategy_features_v2.csv'}")

    # STEP 3: 用户特征提取
    print("\n>>> STEP 3: Extracting user features (3-layer)...")
    user_features = extract_all_user_features(user_data)

    uf_df = pd.DataFrame(user_features).T
    uf_df.index.name = 'user_id'
    uf_df.to_csv(OUTPUT_DIR / 'user_features_v2.csv')
    print(f"[SAVE] User features saved to {OUTPUT_DIR / 'user_features_v2.csv'}")

    # STEP 4: 特征加权 + PCA
    print(f"\n>>> STEP 4: Beta weighting (beta={BETA}) + PCA...")
    print(f"  Behavior features weight: {BETA}")
    print(f"  Non-behavior features weight: {1 - BETA}")
    X, labels, types, strategy_ids, user_ids = build_feature_matrix(
        strategy_features, user_features, beta=BETA
    )
    X_pca, pca, scaler_std = apply_pca(X, labels, types)

    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df['label'] = labels
    pca_df['type'] = types
    pca_df.to_csv(OUTPUT_DIR / 'pca_results_v2.csv', index=False)

    pca_info = {
        'n_components': int(pca.n_components_),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'feature_names': MATCH_FEATURES,
        'feature_groups': FEATURE_GROUPS,
        'beta': BETA,
        'loadings': pca.components_.tolist(),
    }
    with open(OUTPUT_DIR / 'pca_info_v2.json', 'w') as f:
        json.dump(pca_info, f, indent=2)

    # STEP 5: 匹配度计算
    print("\n>>> STEP 5: Computing similarity...")
    sim_results = compute_similarity(X_pca, len(strategy_ids))

    for metric_name, data in sim_results.items():
        print(f"\n  --- {data['metric_name']} ---")
        rankings = rank_strategies(data['similarity'], strategy_ids, user_ids)
        for user_id, recs in rankings.items():
            print(f"  User {user_id}:")
            for rec in recs[:3]:
                print(f"    #{rec['rank']} {rec['strategy']}: sim={rec['similarity']:.4f}")

    # STEP 6: 可解释输出
    print("\n>>> STEP 6: Generating explanations...")
    recommendations = {}
    for metric_name, data in sim_results.items():
        rankings = rank_strategies(data['similarity'], strategy_ids, user_ids)
        recommendations[metric_name] = {}
        for user_id, recs in rankings.items():
            top_rec = recs[0]
            explanation = generate_explanation(
                user_id, top_rec,
                user_features[user_id],
                strategy_features[top_rec['strategy']]
            )
            recommendations[metric_name][user_id] = {
                'top3': recs[:3],
                'explanation': explanation,
            }
            print(f"\n  [{metric_name}] User {user_id}:")
            print(f"    Top: {top_rec['strategy']} (sim={top_rec['similarity']:.4f})")
            print(f"    {explanation['popup_text']}")

    with open(OUTPUT_DIR / 'recommendations_v2.json', 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] Recommendations saved to {OUTPUT_DIR / 'recommendations_v2.json'}")

    # STEP 7: 可视化
    print("\n>>> STEP 7: Generating plots...")
    plot_strategy_nav(strategy_nav, OUTPUT_DIR / 'strategy_nav.png')
    plot_pca_scatter(X_pca, labels, types, strategy_ids, user_ids, pca, OUTPUT_DIR / 'pca_scatter_v2.png')
    plot_similarity_heatmap(sim_results, strategy_ids, user_ids, OUTPUT_DIR / 'similarity_heatmap_v2.png')
    plot_feature_importance(pca, MATCH_FEATURES, OUTPUT_DIR / 'pca_loadings_v2.png')
    plot_feature_comparison(strategy_features, user_features, OUTPUT_DIR / 'feature_radar_v2.png')

    # 总结
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print(f"\n  Feature architecture: {len(MATCH_FEATURES)} dimensions")
    print(f"    Behavior (6): {BEHAVIOR_FEATURES}")
    print(f"    Asset Pref (3): {ASSET_PREF_FEATURES}")
    print(f"    Risk Proxy (3): {RISK_PROXY_FEATURES}")
    print(f"  Beta: {BETA}, Lambda: {LAMBDA}")
    print(f"  Primary metric: Radial-Penalty Cosine")
    print(f"\n  Output files in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
