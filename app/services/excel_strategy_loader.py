"""
从 DLMethod Excel 文件加载37种策略数据，转换为统计方法兼容格式。

数据来源:
  - 量化策略绩效-1.xlsx (12策略)
  - 量化策略绩效-2.xlsx (22策略)
  - 带ETF的策略1.csv, 带ETF策略2.csv, 朝花夕拾策略.csv (3策略)

输出:
  strategy_trades: {strategy_name: DataFrame(trades.csv 兼容格式)}
  strategy_nav: {strategy_name: DataFrame(daily_value.csv 兼容格式)}
"""

import pandas as pd
import numpy as np
from pathlib import Path


# sheet名 -> 策略名映射
SHEET_MAP_1 = {
    '交易记录煤炭': '煤炭周期优选动态轮动策略',
    '交易记录300': '沪深300增强策略',
    '交易记录半导体': '半导体优选策略',
    '交易记录红利': '成长红利量化选股',
    '交易记录双创': '双创50增强',
    '交易记录计算机': '计算机ETF优选策略',
    '交易记录科创': '科创参数加强板',
    '交易记录机器人': '机器人ETF优选策略',
    '交易记录化工': '化工ETF优选策略',
    '交易记录中证2000': '中证2000增强',
    '交易记录形态': '形态识别',
    '交易记录800': '中证800增强',
}

SHEET_MAP_2 = {
    '交易记录军工': '军工etf增强',
    '交易记录1000': '中证1000增强',
    '交易记录旅游': '旅游etf增强',
    '交易记录国企': '国企etf增强',
    '交易记录游戏': '游戏etf增强',
    '交易记录酒': '酒etf增强',
    '交易记录动量趋势': '动量趋势策略',
    '交易记录锤子策略': '锤子策略',
    '交易记录综合全': '综合全',
    '医疗etf增强': '医疗etf增强',
    '交易记录通信': '通信etf增强',
    '交易记录房地产': '房地产etf增强',
    '交易记录食品': '食品etf增强',
    '交易记录创业板': '创业板增强',
    '交易记录etf动量改': 'etf动量改',
    '交易记录行业': '行业etf增强',
    '交易记录养殖': '养殖etf增强',
    '交易记录综合拆分1': '综合拆分1',
    '交易记录综合拆分2': '综合拆分2',
    '交易记录申万动量': '申万动量',
    '交易记录均衡持仓': '均衡持仓',
    '交易记录杠铃': '杠铃',
}

CSV_STRATEGIES = [
    ('带ETF的策略1.csv', '国证2000ETF增强'),
    ('带ETF策略2.csv', '创业板300ETF增强'),
    ('朝花夕拾策略.csv', '朝花夕拾策略'),
]

INITIAL_CAPITAL = 1_000_000


def _standardize_raw_excel(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    标准化 Excel 原始交易数据为 trades.csv 兼容格式。

    Excel 列: trade_time, symbol, side, volume, vwap, amount, btype, ...
    btype: '买入开仓' -> BUY, '卖出平仓' -> SELL, '银证转入' -> 过滤

    trades.csv 期望列: product_id, symbol, side, price, qty, amount, trade_time, trade_date
    """
    out = pd.DataFrame()
    out['product_id'] = strategy_name
    out['product_name'] = ''
    out['symbol'] = df['symbol'].astype(str)
    out['name'] = ''

    # 用 btype 判断买卖方向
    if 'btype' in df.columns:
        btype = df['btype'].astype(str)
        mapped_side = btype.apply(
            lambda x: 'BUY' if '买入' in x else ('SELL' if '卖出' in x else None)
        )
    else:
        # 回退: 用 side 列
        raw_side = df['side'].astype(str)
        mapped_side = raw_side.apply(
            lambda x: 'BUY' if '买' in x else ('SELL' if '卖' in x else None)
        )

    out['side'] = mapped_side
    out['_side'] = mapped_side
    out['price'] = pd.to_numeric(df['vwap'], errors='coerce')
    raw_qty = pd.to_numeric(df['volume'], errors='coerce')
    out['qty'] = raw_qty.abs()
    out['quantity'] = raw_qty.abs()  # alias for fifo_pair_trades
    out['volume'] = raw_qty.abs()
    out['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
    out['trade_time'] = pd.to_datetime(df['trade_time'])
    out['trade_date'] = pd.to_datetime(df['trade_time']).dt.date

    # 过滤：去除 symbol 为空或 trade_time 为 NaT 的记录
    out = out[out['symbol'].notna() & out['symbol'] != 'nan'].copy()
    out = out[out['trade_time'].notna()].copy()

    # 只保留实际买卖
    out = out[out['side'].isin(['BUY', 'SELL'])].copy()
    return out


def _standardize_dlmethod(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    标准化 DLMethod 清洗后格式（用于 CSV 补充策略）。

    DLMethod 列: datetime, stock_code, action, volume, price, amount
    """
    out = pd.DataFrame()
    out['product_id'] = strategy_name
    out['product_name'] = ''
    out['symbol'] = df['stock_code'].astype(str)
    out['name'] = ''
    out['side'] = df['action']
    out['_side'] = df['action']
    out['price'] = pd.to_numeric(df['price'], errors='coerce')
    out['qty'] = pd.to_numeric(df['volume'], errors='coerce')
    out['quantity'] = pd.to_numeric(df['volume'], errors='coerce')  # alias for fifo_pair_trades
    out['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    out['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    out['trade_time'] = pd.to_datetime(df['datetime'])
    out['trade_date'] = pd.to_datetime(df['datetime']).dt.date
    out = out[out['side'].isin(['BUY', 'SELL'])].copy()
    return out


def _build_nav_from_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    从交易记录构建净值序列（单次遍历）。

    跟踪现金 + 持仓市值。买入消耗现金，卖出增加现金。
    市值用最后成交价计算。
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['date', 'nav'])

    df = trades_df.copy()
    df['date'] = pd.to_datetime(df['trade_time']).dt.normalize()
    df = df.sort_values('date').reset_index(drop=True)

    cash = float(INITIAL_CAPITAL)
    position = {}       # symbol -> net qty
    last_px = {}        # symbol -> last traded price
    daily_nav = {}

    for _, t in df.iterrows():
        sym = t['symbol']
        price = float(t['price'])
        qty = float(t['qty'])

        if t['side'] == 'BUY':
            cash -= qty * price
            position[sym] = position.get(sym, 0) + qty
        else:
            cash += qty * price
            position[sym] = position.get(sym, 0) - qty

        last_px[sym] = price

        # 总市值 = 现金 + 未平仓市值
        mv = cash
        for s, q in position.items():
            if q > 0 and s in last_px:
                mv += q * last_px[s]

        daily_nav[t['date']] = mv / INITIAL_CAPITAL

    nav_df = pd.DataFrame(sorted(daily_nav.items()), columns=['date', 'nav'])
    nav_df['nav'] = nav_df['nav'].clip(lower=0.01)
    return nav_df


def load_excel_strategies(stats_dir: Path | None = None) -> tuple[dict, dict]:
    """
    从 stats_data/ 下的 Excel 文件加载37种策略。

    优先使用 app.config 中定义的路径，回退到 stats_data/ 子目录。

    Returns:
        (strategy_trades, strategy_nav)
        strategy_trades: {strategy_name: DataFrame}
        strategy_nav: {strategy_name: DataFrame}
    """
    if stats_dir is None:
        try:
            from app.config import STATS_DATA_DIR
            stats_dir = STATS_DATA_DIR
        except ImportError:
            project_root = Path(__file__).parent.parent.parent
            stats_dir = project_root / "stats_data"

    strategy_trades = {}
    strategy_nav = {}

    # --- 绩效1.xlsx ---
    file1 = stats_dir / '量化策略绩效-1.xlsx'
    if file1.exists():
        for sheet_name, strategy_name in SHEET_MAP_1.items():
            try:
                df = pd.read_excel(file1, sheet_name=sheet_name)
                trades = _standardize_raw_excel(df, strategy_name)
                if not trades.empty:
                    strategy_trades[strategy_name] = trades
                    strategy_nav[strategy_name] = _build_nav_from_trades(trades)
            except Exception as e:
                print(f"[excel_loader] Warning: Failed to load {sheet_name}: {e}")

    # --- 绩效2.xlsx ---
    file2 = stats_dir / '量化策略绩效-2.xlsx'
    if file2.exists():
        for sheet_name, strategy_name in SHEET_MAP_2.items():
            try:
                df = pd.read_excel(file2, sheet_name=sheet_name)
                trades = _standardize_raw_excel(df, strategy_name)
                if not trades.empty:
                    strategy_trades[strategy_name] = trades
                    strategy_nav[strategy_name] = _build_nav_from_trades(trades)
            except Exception as e:
                print(f"[excel_loader] Warning: Failed to load {sheet_name}: {e}")

    # --- CSV 补充策略 ---
    csv_dir = stats_dir
    if not csv_dir.exists():
        csv_dir = stats_dir.parent
    for csv_file, strategy_name in CSV_STRATEGIES:
        csv_path = csv_dir / csv_file
        if not csv_path.exists():
            csv_path = csv_dir.parent / csv_file
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                df['datetime'] = pd.to_datetime(df['trade_time'])
                df['stock_code'] = (df['symbol'].astype(str)
                                    .str.replace('.XSHE', '', regex=False)
                                    .str.replace('.XSHG', '', regex=False)
                                    .str.replace('.SZSE', '', regex=False)
                                    .str.replace('.SHSE', '', regex=False))
                df['action'] = df['side'].str.upper().str.strip()
                df['volume'] = pd.to_numeric(df['qty'], errors='coerce').abs()
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
                trades = _standardize_dlmethod(df, strategy_name)
                if not trades.empty:
                    strategy_trades[strategy_name] = trades
                    strategy_nav[strategy_name] = _build_nav_from_trades(trades)
            except Exception as e:
                print(f"[excel_loader] Warning: Failed to load CSV {csv_file}: {e}")

    print(f"[excel_loader] Loaded {len(strategy_trades)} strategies from Excel/CSV files")
    return strategy_trades, strategy_nav
