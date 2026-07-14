from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_strategy_data(extractor):
    """Load strategy features, NAV, and trades without importing Streamlit."""
    from app.config import STRATEGY_DATA_DIR

    strategy_features: dict[str, dict] = {}
    strategy_nav: dict[str, pd.DataFrame] = {}
    strategy_trades: dict[str, pd.DataFrame] = {}

    if STRATEGY_DATA_DIR.exists():
        for dir_path in sorted(STRATEGY_DATA_DIR.iterdir()):
            if not dir_path.is_dir():
                continue
            strategy_id = dir_path.name

            dv_file = dir_path / "daily_value.csv"
            if dv_file.exists():
                nav_df = pd.read_csv(dv_file)
                nav_df["date"] = pd.to_datetime(nav_df["date"])
                nav_df = nav_df.sort_values("date").reset_index(drop=True)
                strategy_nav[strategy_id] = nav_df

            trades_file = dir_path / "trades.csv"
            if trades_file.exists() and strategy_id in strategy_nav:
                trades_df = pd.read_csv(trades_file)
                trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"])
                strategy_trades[strategy_id] = trades_df
                try:
                    strategy_features[strategy_id] = extractor.extract_strategy_features(
                        strategy_nav[strategy_id], trades_df
                    )
                except Exception:
                    pass

    try:
        from app.config import STATS_DATA_DIR
        from app.services.excel_strategy_loader import load_excel_strategies

        excel_trades, excel_nav = load_excel_strategies(STATS_DATA_DIR)
        strategy_trades.update(excel_trades)
        for sid, trades_df in excel_trades.items():
            if sid in strategy_nav or sid in excel_nav:
                if sid not in strategy_nav:
                    strategy_nav[sid] = excel_nav.get(sid, pd.DataFrame())
                try:
                    strategy_features[sid] = extractor.extract_strategy_features(
                        excel_nav.get(sid, pd.DataFrame()), trades_df
                    )
                except Exception as exc:
                    print(f"[api excel_loader] Failed to extract features for {sid}: {exc}")
    except Exception as exc:
        print(f"[api excel_loader] Failed to load Excel strategies: {exc}")

    return strategy_features, strategy_nav, strategy_trades


def _compute_strategy_nav_info(strategy_nav: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    nav_info: dict[str, dict[str, float]] = {}
    for sid, nav_df in strategy_nav.items():
        if "nav" not in nav_df.columns or len(nav_df) < 10:
            nav_info[sid] = {"annual_return": 0.0, "max_drawdown": 0.0}
            continue

        nav = nav_df["nav"].values
        n_days = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days
        total_ret = (nav[-1] - nav[0]) / nav[0]
        ann_ret = ((1 + total_ret) ** (365 / max(n_days, 1)) - 1) * 100

        peak = nav[0]
        max_dd = 0.0
        for value in nav:
            peak = max(peak, value)
            max_dd = min(max_dd, (value - peak) / peak)

        nav_info[sid] = {"annual_return": float(ann_ret), "max_drawdown": float(max_dd * 100)}
    return nav_info


@lru_cache(maxsize=1)
def init_services() -> dict:
    """Initialize shared project services for API routes."""
    from app.services.auth import AuthService
    from app.services.backends.fusion import FusionBackend
    from app.services.backends.lstm import LSTMBackend
    from app.services.backends.statistical import StatisticalBackend
    from app.services.feature_extractor import FeatureExtractor
    from app.services.matching_backend import BackendRegistry
    from app.services.popup_generator import PopupGenerator
    from app.services.profile import ProfileService
    from app.services.questionnaire import QuestionnaireService
    from app.services.recommendation import RecommendationService
    from app.services.storage import StorageService

    storage = StorageService()
    auth = AuthService(storage)
    extractor = FeatureExtractor()

    strategy_features, strategy_nav, strategy_trades = _load_strategy_data(extractor)
    feature_means = extractor.get_feature_means(strategy_features)
    questionnaire_svc = QuestionnaireService(strategy_mean_features=feature_means)
    profile_svc = ProfileService(storage, extractor)

    registry = BackendRegistry()
    stat_backend = StatisticalBackend()
    stat_backend.fit(strategy_features, strategy_nav)
    registry.register(stat_backend)

    lstm_backend = LSTMBackend(storage)
    lstm_available = True
    try:
        lstm_backend.fit(strategy_features, strategy_nav)
    except FileNotFoundError:
        lstm_available = False
    registry.register(lstm_backend)

    fusion_backend = FusionBackend(stat_backend, lstm_backend)
    if lstm_available:
        fusion_backend.fit(strategy_features, strategy_nav)
    registry.register(fusion_backend)

    popup_gen = PopupGenerator()
    recommendation_svc = RecommendationService(registry, popup_gen)
    nav_info = _compute_strategy_nav_info(strategy_nav)
    recommendation_svc.set_strategy_nav_info(nav_info)

    return {
        "storage": storage,
        "auth": auth,
        "extractor": extractor,
        "questionnaire_svc": questionnaire_svc,
        "profile_svc": profile_svc,
        "registry": registry,
        "stat_backend": stat_backend,
        "lstm_backend": lstm_backend,
        "lstm_available": lstm_available,
        "fusion_backend": fusion_backend,
        "recommendation_svc": recommendation_svc,
        "popup_gen": popup_gen,
        "strategy_features": strategy_features,
        "strategy_nav": strategy_nav,
        "strategy_trades": strategy_trades,
        "nav_info": nav_info,
    }
