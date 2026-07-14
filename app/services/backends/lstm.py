"""
LSTM 128 维序列风格匹配后端。

运行时使用客户上传交易流水生成 token 序列，再通过 DLMethod 训练好的
BiLSTM encoder 生成客户向量，与预计算策略向量做 cosine similarity。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.config import (
    LSTM_CACHE_DIR,
    LSTM_EMBEDDING_META,
    LSTM_INDUSTRY_MAPPING,
    LSTM_INDUSTRY_MAPPING_REVIEW,
    LSTM_MODEL_FILE,
    LSTM_RECOMMENDATIONS,
    LSTM_SHAP_ANALYSIS,
    LSTM_SIMILARITY_MATRIX,
    LSTM_STRATEGY_EMBEDDINGS,
    LSTM_TOKEN_VOCAB,
)
from app.services.matching_backend import MatchingBackend
from app.services.price_data import normalize_symbol
from app.services.storage import StorageService
from app.services.trend_service import standardize_trades


MIN_VALID_TOKENS = 10


class _RuntimeLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        mask = (x != self.embedding.padding_idx).unsqueeze(-1).float()
        pooled = (unpacked * mask).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
        out = self.projection(pooled)
        return nn.functional.normalize(out, p=2, dim=1)


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"missing")
            continue
        stat = path.stat()
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _amount_bucket(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "S"
    if value <= q66:
        return "M"
    return "L"


def _holding_bucket(days: Any) -> str:
    if days is None or pd.isna(days):
        return "UNK"
    if days <= 3:
        return "D0_3"
    if days <= 10:
        return "D4_10"
    if days <= 30:
        return "D11_30"
    if days <= 90:
        return "D31_90"
    return "D90P"


def _turnover_bucket(turnover: float) -> str:
    if turnover <= 5:
        return "LOW"
    if turnover <= 30:
        return "MID"
    if turnover <= 80:
        return "HIGH"
    return "ULTRA"


def _return_bucket(ret: Any) -> str:
    if ret is None or pd.isna(ret):
        return "NA"
    if ret <= -0.05:
        return "LOSS_L"
    if ret < 0:
        return "LOSS_S"
    if ret < 0.05:
        return "GAIN_S"
    return "GAIN_L"


def _drawdown_bucket(dd: float) -> str:
    if dd <= 0.03:
        return "LOW"
    if dd <= 0.10:
        return "MID"
    if dd <= 0.20:
        return "HIGH"
    return "EXTREME"


def _market_bucket(state: float) -> str:
    if state <= -0.005:
        return "BEAR"
    if state >= 0.005:
        return "BULL"
    return "FLAT"


class LSTMBackend(MatchingBackend):
    def __init__(self, storage: StorageService | None = None):
        self.storage = storage or StorageService()
        self._is_fitted = False
        self._recommendations: pd.DataFrame | None = None
        self._shap_report: dict | None = None
        self._strategy_ids: list[str] = []
        self._strategy_embeddings: np.ndarray | None = None
        self._token2id: dict[str, int] = {}
        self._code2industry: dict[str, str] = {}
        self._model: _RuntimeLSTMEncoder | None = None
        self._pad_idx = 0
        self._max_seq_len = 512
        self._model_fingerprint = ""
        self._strategy_universe_version = ""
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def name(self) -> str:
        return "lstm"

    def fit(self, strategy_features: dict, strategy_nav: dict | None = None):
        required = [
            LSTM_TOKEN_VOCAB,
            LSTM_STRATEGY_EMBEDDINGS,
            LSTM_EMBEDDING_META,
            LSTM_MODEL_FILE,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("LSTM 推理产物缺失: " + ", ".join(missing))

        with open(LSTM_TOKEN_VOCAB, "r", encoding="utf-8") as f:
            self._token2id = {str(k): int(v) for k, v in json.load(f)["token2id"].items()}
        with open(LSTM_EMBEDDING_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._strategy_ids = [str(item) for item in meta["strategy_names"]]
        self._strategy_embeddings = np.load(LSTM_STRATEGY_EMBEDDINGS).astype(np.float32)

        if len(self._strategy_ids) != len(self._strategy_embeddings):
            raise ValueError("LSTM 策略名称数量与策略向量数量不一致。")

        self._load_model()
        self._load_industry_mapping()

        if LSTM_RECOMMENDATIONS.exists():
            self._recommendations = pd.read_csv(LSTM_RECOMMENDATIONS)
        if LSTM_SHAP_ANALYSIS.exists():
            with open(LSTM_SHAP_ANALYSIS, "r", encoding="utf-8") as f:
                self._shap_report = json.load(f)

        self._model_fingerprint = _file_digest([LSTM_MODEL_FILE, LSTM_TOKEN_VOCAB])
        self._strategy_universe_version = _file_digest(
            [LSTM_STRATEGY_EMBEDDINGS, LSTM_EMBEDDING_META, LSTM_SIMILARITY_MATRIX]
        )
        LSTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._is_fitted = True

    def assign_account(self, user_id: str) -> None:
        """兼容旧上传接口；实时推理不再分配 Account_A/B/C 槽位。"""
        return None

    def get_assigned_accounts(self) -> dict[str, str]:
        """兼容旧 AppState 字段；实时推理没有账户槽位。"""
        return {}

    def predict(
        self,
        user_features: dict[str, float],
        beta: float = 0.5,
        top_n: int = 3,
        industry_vector: dict[str, float] | None = None,
    ) -> dict:
        if not self._is_fitted:
            raise RuntimeError("LSTMBackend not fitted. Call fit() first.")

        user_id = str(user_features.get("_user_id") or "")
        if not user_id:
            return self._empty_result("missing_user_id", "缺少客户 ID，无法读取客户交易流水。")

        data_status = self.storage.trade_data_status(user_id)
        if data_status["tradeFingerprint"] == "no-trades":
            return self._empty_result(
                "no_trades",
                "客户尚未上传交易流水，LSTM 序列风格匹配暂不可用。",
                data_status=data_status,
            )

        cache_key = self._cache_key(user_id, data_status["tradeFingerprint"])
        cached = self._load_cached(cache_key)
        if cached is not None:
            return self._rank_from_sims(cached["all_sims"], top_n, cached["metadata"] | {"cacheHit": True})

        trades_df = self.storage.load_trades(user_id)
        if trades_df is None or trades_df.empty:
            return self._empty_result("no_trades", "客户交易流水为空。", data_status=data_status)

        tokens = self._build_tokens(trades_df)
        token_ids = [self._token2id[token] for token in tokens if token in self._token2id]
        unknown_count = len(tokens) - len(token_ids)
        if len(token_ids) < MIN_VALID_TOKENS:
            return self._empty_result(
                "insufficient_tokens",
                f"LSTM 有效交易 token 少于 {MIN_VALID_TOKENS} 个，无法形成可靠序列风格匹配。",
                data_status=data_status,
                extra={
                    "tokenCount": len(tokens),
                    "validTokenCount": len(token_ids),
                    "unknownTokenCount": unknown_count,
                },
            )

        vector = self._encode_token_ids(token_ids)
        assert self._strategy_embeddings is not None
        sims = self._strategy_embeddings @ vector
        all_sims = {sid: float(score) for sid, score in zip(self._strategy_ids, sims)}
        metadata = self._metadata(
            "ok",
            "LSTM 已基于当前客户交易流水实时生成序列风格向量。",
            data_status,
            cache_hit=False,
            tokenCount=len(tokens),
            validTokenCount=len(token_ids),
            unknownTokenCount=unknown_count,
        )
        self._save_cached(cache_key, {"all_sims": all_sims, "metadata": metadata})
        return self._rank_from_sims(all_sims, top_n, metadata)

    def predict_for_account(self, account_name: str, top_n: int = 3) -> dict:
        """兼容旧内部调用。实时推理不再支持按 Account_A/B/C 查表。"""
        return self._empty_result(
            "legacy_account_disabled",
            "LSTM 已切换为客户交易流水实时推理，不再使用预计算账户槽位。",
        )

    def get_all_metrics(self, user_features: dict[str, float], beta: float = 0.5) -> dict:
        if not self._is_fitted:
            return {"lstm": {"similarity": None, "metric_name": "LSTM (not fitted)"}}
        return {"lstm": {"similarity": None, "metric_name": "LSTM realtime cosine similarity"}}

    def get_strategy_ids(self) -> list[str]:
        return list(self._strategy_ids)

    def get_data_status(self, user_id: str | None = None) -> dict[str, Any]:
        status = {
            "modelFingerprint": self._model_fingerprint,
            "strategyUniverseVersion": self._strategy_universe_version,
        }
        if user_id:
            status.update(self.storage.trade_data_status(user_id))
        return status

    def _load_model(self) -> None:
        checkpoint = torch.load(LSTM_MODEL_FILE, map_location="cpu")
        config = checkpoint["config"]
        self._pad_idx = int(config.get("pad_idx", len(self._token2id)))
        self._max_seq_len = int(config.get("max_seq_len", 512))
        model = _RuntimeLSTMEncoder(
            vocab_size=int(config["vocab_size"]),
            embed_dim=int(config["embed_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            output_dim=int(config["output_dim"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            pad_idx=self._pad_idx,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model

    def _load_industry_mapping(self) -> None:
        path = LSTM_INDUSTRY_MAPPING_REVIEW if LSTM_INDUSTRY_MAPPING_REVIEW.exists() else LSTM_INDUSTRY_MAPPING
        if not path.exists():
            self._code2industry = {}
            return
        df = pd.read_csv(path, dtype={"stock_code": str}, encoding="utf-8-sig")
        df["stock_code"] = df["stock_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str[-6:].str.zfill(6)
        if "review_industry" in df.columns:
            reviewed = df["review_industry"].fillna("").astype(str).str.strip()
            df["industry_final"] = np.where(reviewed.ne(""), reviewed, df.get("industry", "综合"))
        else:
            df["industry_final"] = df.get("industry", "综合")
        self._code2industry = dict(zip(df["stock_code"], df["industry_final"]))

    def _build_tokens(self, raw_trades: pd.DataFrame) -> list[str]:
        trades = standardize_trades(raw_trades, source="customer")
        if trades.empty:
            return []

        df = pd.DataFrame({
            "date": trades["date"],
            "stock_code": trades["symbol"].apply(normalize_symbol),
            "action": trades["side"].map({"buy": "BUY", "sell": "SELL"}),
            "volume": pd.to_numeric(trades["quantity"], errors="coerce").abs(),
            "price": pd.to_numeric(trades["price"], errors="coerce"),
            "amount": pd.to_numeric(trades["amount"], errors="coerce").abs(),
        })
        df = df[df["action"].isin(["BUY", "SELL"])]
        df = df.dropna(subset=["date", "price", "volume", "amount"])
        df = df[(df["stock_code"] != "") & (df["price"] > 0) & (df["volume"] > 0) & (df["amount"] > 0)]
        if df.empty:
            return []
        df["industry"] = df["stock_code"].map(self._code2industry).fillna("综合")
        return self._build_tokens_for_entity(df)

    def _build_tokens_for_entity(self, df: pd.DataFrame) -> list[str]:
        df = self._add_trade_style_columns(df)
        q33, q66 = np.percentile(df["amount"].values, [33.33, 66.67])
        turnover_style = _turnover_bucket(self._compute_entity_turnover(df))
        tokens = []
        for _, row in df.sort_values("date").iterrows():
            tokens.append(
                f"{row['industry']}_{row['action']}"
                f"_A{_amount_bucket(float(row['amount']), q33, q66)}"
                f"_H{_holding_bucket(row['matched_holding_days'])}"
                f"_T{turnover_style}"
                f"_R{_return_bucket(row['realized_return'])}"
                f"_D{_drawdown_bucket(float(row['running_drawdown']))}"
                f"_M{_market_bucket(float(row['market_state_value']))}"
            )
        return tokens

    def _compute_entity_turnover(self, df: pd.DataFrame) -> float:
        total_buy = df[df["action"] == "BUY"]["amount"].sum()
        if total_buy <= 0:
            return 0.0
        days_span = (df["date"].max() - df["date"].min()).days
        years = max(days_span / 365.25, 1 / 252)
        positions: dict[str, float] = defaultdict(float)
        snapshots = []
        for _, row in df.sort_values("date").iterrows():
            if row["action"] == "BUY":
                positions[row["stock_code"]] += float(row["amount"])
            else:
                positions[row["stock_code"]] = max(0.0, positions[row["stock_code"]] - float(row["amount"]))
            snapshots.append(sum(positions.values()))
        avg_position = np.mean(snapshots) if snapshots else total_buy
        return float(total_buy / max(avg_position, 1.0) / years)

    def _add_trade_style_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        df["matched_holding_days"] = np.nan
        df["realized_return"] = np.nan
        df["running_drawdown"] = 0.0

        for code, idxs in df.groupby("stock_code").groups.items():
            queue = deque()
            for idx in df.loc[idxs].sort_values("date").index:
                row = df.loc[idx]
                price = float(row["price"])
                volume = float(row["volume"])
                if price <= 0 or volume <= 0:
                    continue
                if row["action"] == "BUY":
                    queue.append((row["date"], price, volume))
                    continue
                remaining = volume
                hold_parts = []
                ret_parts = []
                while remaining > 0 and queue:
                    buy_date, buy_price, buy_vol = queue[0]
                    matched = min(buy_vol, remaining)
                    hold_parts.append(((row["date"] - buy_date).days, matched))
                    ret_parts.append(((price - buy_price) / buy_price, matched * price))
                    if buy_vol <= remaining:
                        queue.popleft()
                    else:
                        queue[0] = (buy_date, buy_price, buy_vol - matched)
                    remaining -= matched
                if hold_parts:
                    total_v = sum(v for _, v in hold_parts)
                    df.loc[idx, "matched_holding_days"] = sum(d * v for d, v in hold_parts) / max(total_v, 1e-9)
                if ret_parts:
                    total_w = sum(w for _, w in ret_parts)
                    df.loc[idx, "realized_return"] = sum(r * w for r, w in ret_parts) / max(total_w, 1e-9)

        positions = defaultdict(lambda: {"volume": 0.0, "price": 0.0})
        peak_equity = 0.0
        for idx, row in df.iterrows():
            code = row["stock_code"]
            price = float(row["price"])
            volume = float(row["volume"])
            if price > 0 and volume > 0:
                pos = positions[code]
                if row["action"] == "BUY":
                    new_vol = pos["volume"] + volume
                    pos["price"] = (pos["price"] * pos["volume"] + price * volume) / max(new_vol, 1e-9)
                    pos["volume"] = new_vol
                else:
                    pos["volume"] = max(0.0, pos["volume"] - volume)
                    pos["price"] = price
            equity = sum(v["volume"] * v["price"] for v in positions.values())
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                df.loc[idx, "running_drawdown"] = np.clip(1.0 - equity / peak_equity, 0.0, 1.0)

        daily_price = df.pivot_table(index="date", columns="stock_code", values="price", aggfunc="last")
        if len(daily_price) >= 2:
            daily_ret = daily_price.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            market_ret = daily_ret.mean(axis=1).rolling(5, min_periods=1).mean()
            df["market_state_value"] = [market_ret.reindex([d], method="ffill").iloc[0] for d in df["date"]]
            df["market_state_value"] = df["market_state_value"].fillna(0.0)
        else:
            df["market_state_value"] = 0.0
        return df

    def _encode_token_ids(self, token_ids: list[int]) -> np.ndarray:
        assert self._model is not None
        trimmed = token_ids[: self._max_seq_len]
        x = torch.full((1, len(trimmed)), self._pad_idx, dtype=torch.long)
        x[0, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)
        lengths = torch.tensor([len(trimmed)], dtype=torch.long)
        with torch.no_grad():
            vector = self._model(x, lengths).cpu().numpy()[0].astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def _cache_key(self, user_id: str, trade_fingerprint: str) -> str:
        raw = "|".join([user_id, trade_fingerprint, self._model_fingerprint, self._strategy_universe_version])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return LSTM_CACHE_DIR / f"{cache_key}.json"

    def _load_cached(self, cache_key: str) -> dict[str, Any] | None:
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._memory_cache[cache_key] = data
            return data
        except Exception:
            return None

    def _save_cached(self, cache_key: str, data: dict[str, Any]) -> None:
        self._memory_cache[cache_key] = data
        try:
            with open(self._cache_path(cache_key), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _rank_from_sims(self, all_sims: dict[str, float], top_n: int, metadata: dict[str, Any]) -> dict:
        metadata = self._normalize_metadata(metadata)
        ranked = sorted(all_sims.items(), key=lambda item: -item[1])
        top3 = [
            {"strategy": strategy, "similarity": float(score), "rank": index + 1}
            for index, (strategy, score) in enumerate(ranked[:top_n])
        ]
        return {
            "top3": top3,
            "explanation": self._build_explanation(top3, metadata),
            "metric_used": "LSTM realtime cosine similarity (128-dim BiLSTM embedding)",
            "all_sims": all_sims,
            "phase1_rank": {},
            "phase2_rank": {},
            "metadata": metadata,
        }

    def _empty_result(
        self,
        status: str,
        message: str,
        data_status: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        metadata = self._metadata(status, message, data_status or {}, cache_hit=False, **(extra or {}))
        return {
            "top3": [],
            "explanation": {"lstm_status": status, "lstm_message": message},
            "metric_used": f"LSTM realtime cosine similarity ({status})",
            "note": message,
            "all_sims": {},
            "phase1_rank": {},
            "phase2_rank": {},
            "metadata": metadata,
        }

    def _metadata(
        self,
        status: str,
        message: str,
        data_status: dict[str, Any],
        cache_hit: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "lstmStatus": status,
            "lstmMessage": message,
            "cacheHit": cache_hit,
            "modelFingerprint": self._model_fingerprint,
            "strategyUniverseVersion": self._strategy_universe_version,
            "tradeFingerprint": data_status.get("tradeFingerprint"),
            "tradeLastUpdated": data_status.get("tradeLastUpdated"),
            "tradeFileCount": data_status.get("tradeFileCount", 0),
            "tradeCount": data_status.get("tradeCount", 0),
            **extra,
        }

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata)
        aliases = {
            "token_count": "tokenCount",
            "valid_token_count": "validTokenCount",
            "unknown_token_count": "unknownTokenCount",
        }
        for old, new in aliases.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized[old]
        return normalized

    def _build_explanation(self, top3: list[dict], metadata: dict[str, Any]) -> dict:
        explanation = {
            "most_similar_dimensions": [],
            "most_different_dimensions": [],
            "lstm_shap_top_features": [],
            "lstm_status": metadata.get("lstmStatus"),
            "lstm_message": metadata.get("lstmMessage"),
        }
        if self._shap_report and "top_features_by_shap" in self._shap_report:
            for feat_name, shap_val in self._shap_report["top_features_by_shap"][:5]:
                explanation["lstm_shap_top_features"].append({
                    "feature": feat_name,
                    "mean_abs_shap": round(float(shap_val), 4),
                })
        return explanation
