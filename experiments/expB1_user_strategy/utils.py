"""
Shared utilities for Experiment B.1 — data loading, format conversion, feature extraction.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (                                         # noqa: E402
    MATCH_FEATURES,
    extract_user_behavior_features,
    extract_user_asset_pref_features,
    extract_user_risk_proxy_features,
)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

STRATEGIES_CSV = PROJECT_ROOT / "DLMethod" / "clean_strategies.csv"
ACCOUNTS_CSV = PROJECT_ROOT / "DLMethod" / "clean_accounts.csv"


def load_and_prepare_trades(csv_path: Path) -> pd.DataFrame:
    """Load raw trading records from CSV and prepare columns expected by pipeline.py."""
    df = pd.read_csv(csv_path, dtype={"stock_code": str})
    return _prepare(df)


def prepare_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare an already-loaded DataFrame (adds derived columns, sorts by date)."""
    return _prepare(df)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Shared preparation: add columns expected by pipeline.py feature extractors."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["datetime"])
    out["symbol"] = out["stock_code"].astype(str)
    out["action"] = out["action"].astype(str).str.strip().str.upper()
    out["is_buy"] = out["action"] == "BUY"
    out["is_sell"] = out["action"] == "SELL"
    out["quantity"] = out["volume"].astype(float)
    out["amount"] = out["amount"].astype(float)
    return out.sort_values("trade_date").reset_index(drop=True)


def extract_features(trades: pd.DataFrame) -> dict[str, float]:
    """Extract 12-dim feature vector from prepared trading records."""
    if trades.empty:
        return {f: 0.0 for f in MATCH_FEATURES}
    return {
        **extract_user_behavior_features(trades),
        **extract_user_asset_pref_features(trades),
        **extract_user_risk_proxy_features(trades),
    }


def features_to_array(feat_dicts: list[dict]) -> np.ndarray:
    """Convert list of feature dicts to (n, 12) array in MATCH_FEATURES order."""
    return np.array([[d.get(f, 0.0) for f in MATCH_FEATURES] for d in feat_dicts])
