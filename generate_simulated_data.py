"""
模拟数据生成器（阶段二：深度学习方向的技术储备）
===================================================
基于真实12+个策略的统计特性，生成模拟策略和模拟客户的交易数据。

用于:
  1. 在大规模模拟数据上与统计方法交叉验证
  2. 为Word2Vec嵌入训练提供数据
  3. 未来真实数据量扩充后的备选方案

生成规则:
  - 500个模拟策略 + 200个模拟客户
  - 每个实体50~200笔交易
  - 时序逻辑合理（买入->卖出配对）
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent / 'output' / 'simulated_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
N_SIMULATED_STRATEGIES = 500
N_SIMULATED_USERS = 200
MIN_TRADES = 50
MAX_TRADES = 200

# 基于真实策略统计特性设定的参数分布
# 从真实7个策略的统计结果推断
STRATEGY_PARAM_DIST = {
    # 年化收益率范围
    'annualized_return': {'mean': 0.25, 'std': 0.10},
    # 日波动率范围
    'daily_vol': {'mean': 0.17, 'std': 0.08},
    # 最大回撤范围
    'max_drawdown': {'mean': -0.14, 'std': 0.03},
    # 年化换手率范围
    'annual_turnover': {'mean': 25, 'std': 15},
    # 平均持仓周期(天)
    'avg_holding_period': {'mean': 15, 'std': 10},
    # 买卖对称性
    'buy_sell_ratio': {'mean': 1.2, 'std': 0.15},
    # 交易频率(每年)
    'trades_per_year': {'mean': 250, 'std': 80},
    # 持股集中度(HHI)
    'hhi': {'mean': 0.05, 'std': 0.04},
    # 平均仓位比例
    'avg_position': {'mean': 0.75, 'std': 0.12},
}

# 行业列表 (申万一级行业, 简化为20个)
INDUSTRIES = [
    '煤炭', '石油石化', '基础化工', '钢铁', '有色金属',
    '电子', '家用电器', '食品饮料', '纺织服饰', '轻工制造',
    '医药生物', '公用事业', '交通运输', '房地产', '商贸零售',
    '银行', '非银金融', '建筑材料', '通信', '计算机',
]

# 模拟股票池 (每个行业5只股票)
STOCK_POOL = {}
stock_id = 1000
for ind in INDUSTRIES:
    STOCK_POOL[ind] = [f'{stock_id + i:06d}' for i in range(5)]
    stock_id += 100

ALL_STOCKS = [s for stocks in STOCK_POOL.values() for s in stocks]
STOCK_TO_INDUSTRY = {}
for ind, stocks in STOCK_POOL.items():
    for s in stocks:
        STOCK_TO_INDUSTRY[s] = ind


# ============================================================
# 数据生成
# ============================================================
def generate_industry_distribution(hhi_target):
    """生成行业分布向量，使得HHI接近目标值"""
    n = len(INDUSTRIES)
    # 使用Dirichlet分布，通过调节alpha控制集中度
    # HHI近似 = (1 + alpha) / (n * alpha + 1)
    # 反推alpha
    alpha = max(0.5, (1 - hhi_target * n) / (hhi_target * n - 1 / n))
    alpha = min(alpha, 50)
    weights = np.random.dirichlet(np.ones(n) * alpha)
    return weights


def generate_trades(params, n_trades, entity_id, entity_type='strategy'):
    """
    为单个实体生成交易记录

    Args:
        params: 策略/用户参数
        n_trades: 交易笔数
        entity_id: 实体ID
        entity_type: 'strategy' or 'user'
    """
    trades = []
    date_start = pd.Timestamp('2023-01-01')
    date_end = pd.Timestamp('2026-05-01')
    n_days = (date_end - date_start).days

    # 行业偏好 (决定选股倾向)
    ind_weights = generate_industry_distribution(params['hhi'])

    # 生成时间序列
    trade_dates = sorted(
        date_start + pd.Timedelta(days=int(np.random.randint(0, n_days)))
        for _ in range(n_trades)
    )

    # 跟踪持仓
    positions = {}  # symbol -> {'qty': int, 'price': float, 'date': Timestamp}

    for i, trade_date in enumerate(trade_dates):
        # 决定买入还是卖出
        if positions and np.random.random() < 0.4:  # 40%概率卖出
            # 卖出已有持仓
            symbol = np.random.choice(list(positions.keys()))
            pos = positions[symbol]
            # 卖出价格: 买入价格 +/- 波动
            vol = max(params['daily_vol'], 0.05)
            daily_ret = np.random.normal(0, vol / np.sqrt(252))
            holding_days = max((trade_date - pos['date']).days, 1)
            price_change = daily_ret * np.sqrt(holding_days)
            sell_price = pos['price'] * (1 + price_change)
            sell_price = round(max(sell_price, 0.5), 2)

            trades.append({
                'entity_id': entity_id,
                'entity_type': entity_type,
                'trade_date': trade_date,
                'symbol': symbol,
                'industry': STOCK_TO_INDUSTRY.get(symbol, '未知'),
                'side': 'SELL',
                'price': sell_price,
                'quantity': pos['qty'],
                'amount': -round(sell_price * pos['qty'], 2),
                'return': round((sell_price - pos['price']) / pos['price'], 4),
            })
            del positions[symbol]
        else:
            # 买入
            # 按行业偏好选股
            industry = np.random.choice(INDUSTRIES, p=ind_weights)
            symbol = np.random.choice(STOCK_POOL[industry])
            qty = np.random.randint(100, 5000) // 100 * 100  # 100的整数倍
            price = round(np.random.uniform(5, 100), 2)

            trades.append({
                'entity_id': entity_id,
                'entity_type': entity_type,
                'trade_date': trade_date,
                'symbol': symbol,
                'industry': industry,
                'side': 'BUY',
                'price': price,
                'quantity': qty,
                'amount': round(price * qty, 2),
                'return': 0.0,
            })
            positions[symbol] = {
                'qty': qty,
                'price': price,
                'date': trade_date,
            }

    return pd.DataFrame(trades)


def generate_simulated_strategy(idx):
    """生成单个模拟策略的参数"""
    params = {
        'annualized_return': np.random.normal(
            STRATEGY_PARAM_DIST['annualized_return']['mean'],
            STRATEGY_PARAM_DIST['annualized_return']['std']
        ),
        'daily_vol': np.random.normal(
            STRATEGY_PARAM_DIST['daily_vol']['mean'],
            STRATEGY_PARAM_DIST['daily_vol']['std']
        ),
        'max_drawdown': -abs(np.random.normal(
            STRATEGY_PARAM_DIST['max_drawdown']['mean'],
            STRATEGY_PARAM_DIST['max_drawdown']['std']
        )),
        'annual_turnover': max(1, np.random.normal(
            STRATEGY_PARAM_DIST['annual_turnover']['mean'],
            STRATEGY_PARAM_DIST['annual_turnover']['std']
        )),
        'avg_holding_period': max(1, np.random.normal(
            STRATEGY_PARAM_DIST['avg_holding_period']['mean'],
            STRATEGY_PARAM_DIST['avg_holding_period']['std']
        )),
        'buy_sell_ratio': max(0.5, np.random.normal(
            STRATEGY_PARAM_DIST['buy_sell_ratio']['mean'],
            STRATEGY_PARAM_DIST['buy_sell_ratio']['std']
        )),
        'trades_per_year': max(50, np.random.normal(
            STRATEGY_PARAM_DIST['trades_per_year']['mean'],
            STRATEGY_PARAM_DIST['trades_per_year']['std']
        )),
        'hhi': np.clip(np.random.normal(
            STRATEGY_PARAM_DIST['hhi']['mean'],
            STRATEGY_PARAM_DIST['hhi']['std']
        ), 1 / len(INDUSTRIES), 0.5),
        'avg_position': np.clip(np.random.normal(
            STRATEGY_PARAM_DIST['avg_position']['mean'],
            STRATEGY_PARAM_DIST['avg_position']['std']
        ), 0.3, 0.95),
    }
    return params


def generate_simulated_user(idx):
    """生成单个模拟用户的参数"""
    # 用户参数变化更大 (多样性)
    params = {
        'annualized_return': np.random.normal(0.10, 0.20),  # 用户收益更低、波动更大
        'daily_vol': np.random.normal(0.25, 0.10),
        'max_drawdown': -abs(np.random.normal(0.20, 0.08)),
        'annual_turnover': max(1, np.random.normal(15, 10)),
        'avg_holding_period': max(1, np.random.normal(30, 20)),  # 用户持仓更长
        'buy_sell_ratio': max(0.3, np.random.normal(1.5, 0.4)),
        'trades_per_year': max(20, np.random.normal(120, 80)),  # 用户交易更少
        'hhi': np.clip(np.random.normal(0.10, 0.06), 1 / len(INDUSTRIES), 0.7),
        'avg_position': np.clip(np.random.normal(0.65, 0.15), 0.2, 0.95),
    }
    return params


def main():
    print("=" * 60)
    print("  Simulated Data Generator")
    print("=" * 60)

    all_trades = []

    # 生成模拟策略
    print(f"\nGenerating {N_SIMULATED_STRATEGIES} simulated strategies...")
    strategy_params_list = []
    for i in range(N_SIMULATED_STRATEGIES):
        params = generate_simulated_strategy(i)
        strategy_params_list.append(params)
        n_trades = np.random.randint(MIN_TRADES, MAX_TRADES + 1)
        trades = generate_trades(params, n_trades, f'SIM_S_{i:04d}', 'strategy')
        all_trades.append(trades)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} strategies...")

    # 生成模拟用户
    print(f"\nGenerating {N_SIMULATED_USERS} simulated users...")
    user_params_list = []
    for i in range(N_SIMULATED_USERS):
        params = generate_simulated_user(i)
        user_params_list.append(params)
        n_trades = np.random.randint(MIN_TRADES // 2, MAX_TRADES + 1)
        trades = generate_trades(params, n_trades, f'SIM_U_{i:04d}', 'user')
        all_trades.append(trades)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1} users...")

    # 合并所有交易
    print("\nMerging all trades...")
    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # 保存
    all_trades_df.to_csv(OUTPUT_DIR / 'simulated_trades.csv', index=False)
    print(f"Saved {len(all_trades_df)} trades to {OUTPUT_DIR / 'simulated_trades.csv'}")

    # 保存参数
    strategy_meta = []
    for i, params in enumerate(strategy_params_list):
        row = {'entity_id': f'SIM_S_{i:04d}', 'entity_type': 'strategy'}
        row.update(params)
        strategy_meta.append(row)
    pd.DataFrame(strategy_meta).to_csv(OUTPUT_DIR / 'strategy_params.csv', index=False)

    user_meta = []
    for i, params in enumerate(user_params_list):
        row = {'entity_id': f'SIM_U_{i:04d}', 'entity_type': 'user'}
        row.update(params)
        user_meta.append(row)
    pd.DataFrame(user_meta).to_csv(OUTPUT_DIR / 'user_params.csv', index=False)

    # 统计摘要
    print(f"\nSummary:")
    print(f"  Total strategies: {N_SIMULATED_STRATEGIES}")
    print(f"  Total users: {N_SIMULATED_USERS}")
    print(f"  Total trades: {len(all_trades_df):,}")
    print(f"  Avg trades/entity: {len(all_trades_df) / (N_SIMULATED_STRATEGIES + N_SIMULATED_USERS):.1f}")

    # 生成Word2Vec训练用的token序列
    print("\nGenerating token sequences for Word2Vec...")
    tokens_file = OUTPUT_DIR / 'token_sequences.txt'
    with open(tokens_file, 'w') as f:
        for entity_id in all_trades_df['entity_id'].unique():
            entity_trades = all_trades_df[all_trades_df['entity_id'] == entity_id].sort_values('trade_date')
            tokens = []
            for _, row in entity_trades.iterrows():
                ind = row['industry'][:2]  # 行业前2字
                side = 'B' if row['side'] == 'BUY' else 'S'
                # 数量等级
                if row['quantity'] < 500:
                    size = 'S'
                elif row['quantity'] < 2000:
                    size = 'M'
                else:
                    size = 'L'
                tokens.append(f'{ind}_{side}_{size}')
            f.write(' '.join(tokens) + '\n')
    print(f"Saved token sequences to {tokens_file}")
    print(f"  Vocabulary size: {len(set(' '.join(open(tokens_file).read().split())) )}")

    print("\n" + "=" * 60)
    print("  Simulated Data Generation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
