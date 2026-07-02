from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from api.services import init_services


SESSION_COOKIE = "investment_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7
SECRET_KEY = os.getenv("INVESTMENT_API_SECRET", "dev-investment-api-secret")

serializer = URLSafeTimedSerializer(SECRET_KEY, salt="investment-session")

app = FastAPI(title="Investment Strategy Matching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FEATURE_LABELS = {
    "holding_period": "持仓周期",
    "turnover_rate": "换手率",
    "buy_sell_ratio": "买卖对称",
    "hhi_concentration": "持仓集中",
    "disposition_effect": "处置效应",
    "positive_trade_ratio": "胜率",
    "etf_ratio": "ETF占比",
    "avg_price_preference": "价格偏好",
    "position_uniformity": "分仓均匀",
    "avg_loss_magnitude": "亏损幅度",
    "vol_preference": "波动偏好",
    "trend_preference": "趋势偏好",
}

BACKEND_DISPLAY = {
    "statistical": "PCA 统计方法",
    "lstm": "LSTM 序列风格匹配",
    "fusion": "融合推荐 (统计 + LSTM)",
}


class AuthPayload(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class QuestionnaireSubmit(BaseModel):
    answers: dict[str, Any]


class CustomerPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=1000)


class CustomerUpdatePayload(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=1000)
    status: str | None = Field(default=None, max_length=80)


class RecommendPayload(BaseModel):
    backend: str = "fusion"
    top_n: int = Field(default=5, ge=1, le=20)


class TrendPayload(BaseModel):
    strategy_ids: list[str] = Field(default_factory=list)


class SettingsPayload(BaseModel):
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    backend: str | None = None
    fusion_alpha: float | None = Field(default=None, ge=0.0, le=1.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _services() -> dict:
    return init_services()


def _set_session(response: Response, user_id: str) -> None:
    token = serializer.dumps({"user_id": user_id})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def _clear_session(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=401, detail="Invalid session") from None
    user_id = payload.get("user_id")
    user = _services()["storage"].get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _profile_to_dict(profile) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "userId": profile.user_id,
        "beta": profile.beta,
        "riskTolerance": profile.risk_tolerance,
        "initialCapital": profile.initial_capital,
        "features": profile.features,
        "industryVector": profile.industry_vector,
        "updateCount": profile.update_count,
        "confidenceLevel": profile.confidence_level,
        "source": profile.source,
        "lastUpdated": profile.last_updated,
        "matchingBackend": profile.matching_backend,
        "history": profile.history,
    }


def _user_to_dict(user) -> dict[str, Any]:
    return {
        "userId": user.user_id,
        "username": user.username,
        "createdAt": user.created_at,
        "lastLogin": user.last_login,
        "onboardingStatus": user.onboarding_status,
    }


def _customer_entity_id(user, customer_id: str | None = None) -> str:
    services = _services()
    if not customer_id:
        return services["storage"].ensure_default_customer(user).customer_id
    customer = services["storage"].get_customer(user.user_id, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="客户不存在或无权访问。")
    return customer.customer_id


def _customer_status(entity_id: str, services: dict) -> tuple[str, str, str]:
    completed = services["storage"].list_completed_levels(entity_id)
    uploads = services["storage"].list_trade_uploads(entity_id)
    profile = services["profile_svc"].get_profile(entity_id)
    if len(completed) < 3:
        return "needs_questionnaire", "待补资料", f"继续完成问卷：{len(completed)}/3"
    if not uploads:
        return "needs_trades", "待上传交易", "上传交易流水以更新客户画像"
    if profile is None or not profile.features:
        return "needs_profile", "待生成画像", "补齐画像后再生成推荐"
    return "ready_to_recommend", "可生成推荐", "可生成推荐方案并与客户沟通"


def _customer_to_dict(customer, services: dict) -> dict[str, Any]:
    completed = services["storage"].list_completed_levels(customer.customer_id)
    uploads = services["storage"].list_trade_uploads(customer.customer_id)
    profile = services["profile_svc"].get_profile(customer.customer_id)
    status, status_label, next_action = _customer_status(customer.customer_id, services)
    return {
        "customerId": customer.customer_id,
        "ownerUserId": customer.owner_user_id,
        "name": customer.name,
        "status": status,
        "statusLabel": status_label,
        "nextAction": next_action,
        "note": customer.note,
        "createdAt": customer.created_at,
        "lastUpdated": customer.last_updated,
        "completedLevels": completed,
        "uploadCount": len(uploads),
        "tradeCount": sum(upload["trade_count"] for upload in uploads),
        "hasProfile": profile is not None and bool(profile.features),
        "confidenceLevel": profile.confidence_level if profile else None,
    }


def _questionnaire_to_dict(qn, completed: bool = False) -> dict[str, Any]:
    return {
        "level": qn.level,
        "title": qn.title,
        "description": qn.description,
        "estimatedMinutes": qn.estimated_minutes,
        "completed": completed,
        "questions": [
            {
                "id": q.q_id,
                "text": q.text,
                "type": q.q_type,
                "options": q.options,
            }
            for q in qn.questions
        ],
    }


def _profile_feature_chart(profile, strategy_features: dict[str, dict]) -> list[dict[str, Any]]:
    if profile is None or not profile.features:
        return []
    from pipeline import MATCH_FEATURES

    values = [v for row in strategy_features.values() for v in row.values()]
    values.extend(profile.features.values())
    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 1.0
    span = max(max_val - min_val, 1e-9)
    rows = []
    for feature in MATCH_FEATURES:
        value = _safe_float(profile.features.get(feature))
        avg = sum(_safe_float(sf.get(feature)) for sf in strategy_features.values()) / max(len(strategy_features), 1)
        rows.append({
            "key": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": round((value - min_val) / span * 100, 2),
            "rawValue": value,
            "strategyAverage": round((avg - min_val) / span * 100, 2),
        })
    return rows


def _strategy_symbols(strategy_id: str, limit: int = 8) -> list[str]:
    trades = _services()["strategy_trades"].get(strategy_id)
    if trades is None or trades.empty:
        return []
    from app.services.price_data import normalize_symbol

    candidates: list[str] = []
    for col in ["symbol", "stock_code", "证券代码"]:
        if col in trades.columns:
            candidates = [normalize_symbol(v) for v in trades[col].dropna().head(200).tolist()]
            break
    seen: list[str] = []
    for symbol in candidates:
        if symbol and symbol not in seen:
            seen.append(symbol)
        if len(seen) >= limit:
            break
    return seen


@lru_cache(maxsize=128)
def _cached_strategy_trend(strategy_id: str) -> dict[str, Any]:
    from app.services.trend_service import compute_strategy_trend

    trades_df = _services()["strategy_trades"].get(strategy_id)
    if trades_df is None or trades_df.empty:
        return {
            "label": strategy_id,
            "trend": [],
            "meta": {
                "dataQuality": "insufficient_trade_data",
                "warnings": ["该策略缺少可用于行情重算的交易记录。"],
                "missingSymbols": [],
                "fallbackSymbols": [],
                "coverageRate": 0.0,
                "finalReturn": None,
                "startDate": None,
                "endDate": None,
            },
        }
    return compute_strategy_trend(strategy_id, trades_df)


def _recommendation_to_dict(item: dict[str, Any], result, index: int) -> dict[str, Any]:
    strategy_id = item.get("strategy", "")
    similarity = _safe_float(item.get("similarity"))
    score = _normalize_score(similarity * 100)
    nav = _services()["nav_info"].get(strategy_id, {})
    annual_return = _safe_float(nav.get("annual_return"))
    max_drawdown = _safe_float(nav.get("max_drawdown"))
    explanation_parts = []
    if result.metric_used:
        explanation_parts.append(result.metric_used)
    if result.explanation.get("fusion_note"):
        explanation_parts.append(result.explanation["fusion_note"])
    if result.explanation.get("most_similar_dimensions"):
        dims = "、".join(d.get("feature", "") for d in result.explanation["most_similar_dimensions"][:3])
        explanation_parts.append(f"相近维度：{dims}")
    if not explanation_parts:
        explanation_parts.append(f"{strategy_id} 与当前画像的匹配度为 {score:.1f}%。")

    stat_score = result.stat_score.get(strategy_id) if result.stat_score else None
    ml_score = result.ml_score.get(strategy_id) if result.ml_score else None
    if stat_score is not None and ml_score is not None:
        factor_score = _normalize_score(_safe_float(stat_score) * 100)
        cca_score = _normalize_score(_safe_float(ml_score) * 100)
    else:
        factor_score = score
        cca_score = 50.0

    risk_score = _normalize_score(100.0 + max_drawdown)
    performance_score = _normalize_score(50.0 + annual_return)

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_id,
        "score": round(score, 2),
        "rank_sum": float(index + 1),
        "rank_score": score,
        "final_rank_score": score,
        "dimension_ranks": {
            "style": index + 1,
            "performance": index + 1,
            "risk": index + 1,
            "preference": index + 1,
            "factor": index + 1,
            "cluster": index + 1,
            "cca": index + 1,
        },
        "style_similarity": round(score, 2),
        "mahalanobis_style_score": round(score, 2),
        "factor_score": round(factor_score, 2),
        "cluster_score": 50.0,
        "cca_score": round(cca_score, 2),
        "cluster_label": "目标项目策略池",
        "performance_score": round(performance_score, 2),
        "risk_score": round(risk_score, 2),
        "preference_score": round(score, 2),
        "symbol_overlap": 0.0,
        "theme_overlap": 0.0,
        "themes": [BACKEND_DISPLAY.get(result.backend, result.backend)],
        "top_symbols": _strategy_symbols(strategy_id),
        "factor_label": BACKEND_DISPLAY.get(result.backend, result.backend),
        "return_2025": round(annual_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown_2025": round(max_drawdown, 2),
        "explanation": "；".join(part for part in explanation_parts if part),
    }


def _read_upload(file: UploadFile, content: bytes) -> pd.DataFrame:
    suffix = (file.filename or "").lower()
    buffer = io.BytesIO(content)
    if suffix.endswith(".csv"):
        try:
            return pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="gbk")
    return pd.read_excel(buffer)


def _app_state(user, customer_id: str | None = None) -> dict[str, Any]:
    services = _services()
    default_customer = services["storage"].ensure_default_customer(user)
    entity_id = _customer_entity_id(user, customer_id or default_customer.customer_id)
    current_customer = services["storage"].get_customer(user.user_id, entity_id) or default_customer
    customers = services["storage"].list_customers(user.user_id)
    profile = services["profile_svc"].get_profile(entity_id)
    completed = services["storage"].list_completed_levels(entity_id)
    uploads = services["storage"].list_trade_uploads(entity_id)
    active_backends = services["registry"].list_active()
    return {
        "user": _user_to_dict(user),
        "currentCustomer": _customer_to_dict(current_customer, services),
        "customers": [_customer_to_dict(customer, services) for customer in customers],
        "profile": _profile_to_dict(profile),
        "completedLevels": completed,
        "uploads": uploads,
        "backends": [
            {"name": name, "label": BACKEND_DISPLAY.get(name, name)}
            for name in active_backends
        ],
        "lstmAvailable": bool(services["lstm_available"]),
        "lstmAssignedAccounts": services["lstm_backend"].get_assigned_accounts(),
        "fusionAlpha": _safe_float(getattr(services["fusion_backend"], "_alpha", 0.7), 0.7),
        "featureChart": _profile_feature_chart(profile, services["strategy_features"]),
    }


@app.get("/api/health")
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register")
@app.post("/auth/register")
def register(payload: AuthPayload, response: Response) -> dict[str, Any]:
    ok, message, user = _services()["auth"].register(payload.username.strip(), payload.password)
    if not ok or user is None:
        raise HTTPException(status_code=400, detail=message)
    _set_session(response, user.user_id)
    return {"message": message, **_app_state(user)}


@app.post("/api/auth/login")
@app.post("/auth/login")
def login(payload: AuthPayload, response: Response) -> dict[str, Any]:
    ok, message, user = _services()["auth"].login(payload.username.strip(), payload.password)
    if not ok or user is None:
        raise HTTPException(status_code=401, detail=message)
    _set_session(response, user.user_id)
    return {"message": message, **_app_state(user)}


@app.post("/api/auth/logout")
@app.post("/auth/logout")
def logout(response: Response) -> dict[str, str]:
    _clear_session(response)
    return {"message": "已退出登录"}


@app.get("/api/auth/me")
@app.get("/auth/me")
def me(user=Depends(current_user)) -> dict[str, Any]:
    return _app_state(user)


@app.get("/api/customers")
@app.get("/customers")
def list_customers(user=Depends(current_user)) -> dict[str, Any]:
    services = _services()
    services["storage"].ensure_default_customer(user)
    customers = services["storage"].list_customers(user.user_id)
    return {"customers": [_customer_to_dict(customer, services) for customer in customers]}


@app.post("/api/customers")
@app.post("/customers")
def create_customer(payload: CustomerPayload, user=Depends(current_user)) -> dict[str, Any]:
    from app.models.user import Customer

    services = _services()
    customer = Customer.create(
        customer_id=services["storage"].generate_customer_id(),
        owner_user_id=user.user_id,
        name=payload.name.strip(),
        note=payload.note.strip(),
    )
    services["storage"].save_customer(customer)
    return {"message": "客户已创建。", **_app_state(user, customer.customer_id)}


@app.get("/api/customers/{customer_id}")
@app.get("/customers/{customer_id}")
def get_customer(customer_id: str, user=Depends(current_user)) -> dict[str, Any]:
    return _app_state(user, customer_id)


@app.patch("/api/customers/{customer_id}")
@app.patch("/customers/{customer_id}")
def update_customer(customer_id: str, payload: CustomerUpdatePayload, user=Depends(current_user)) -> dict[str, Any]:
    services = _services()
    customer = services["storage"].get_customer(user.user_id, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="客户不存在或无权访问。")
    if payload.name is not None:
        customer.name = payload.name.strip()
    if payload.note is not None:
        customer.note = payload.note.strip()
    if payload.status is not None:
        customer.status = payload.status
    services["storage"].save_customer(customer)
    return {"message": "客户已更新。", **_app_state(user, customer.customer_id)}


@app.get("/api/questionnaires")
@app.get("/questionnaires")
def questionnaires(customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    completed = set(services["storage"].list_completed_levels(entity_id))
    return {
        "questionnaires": [
            _questionnaire_to_dict(qn, qn.level in completed)
            for qn in services["questionnaire_svc"].get_all_questionnaires()
        ]
    }


@app.post("/api/questionnaires/{level}")
@app.post("/questionnaires/{level}")
def submit_questionnaire(
    level: str,
    payload: QuestionnaireSubmit,
    customer_id: str | None = None,
    user=Depends(current_user),
) -> dict[str, Any]:
    level = level.upper()
    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    try:
        qn = services["questionnaire_svc"].get_questionnaire(level)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown questionnaire level: {level}") from None

    missing = [
        q.q_id
        for q in qn.questions
        if q.q_id not in payload.answers
        or payload.answers[q.q_id] in ("", None)
        or (isinstance(payload.answers[q.q_id], list) and not payload.answers[q.q_id])
    ]
    if missing:
        raise HTTPException(status_code=400, detail=f"请回答所有问题。未回答: {', '.join(missing)}")

    score_result = services["questionnaire_svc"].score_answers(level, payload.answers)
    services["storage"].save_questionnaire_results(entity_id, level, payload.answers)

    profile = services["profile_svc"].get_profile(entity_id)
    if profile is None:
        profile = services["profile_svc"].create_profile_from_questionnaire(entity_id, score_result)
    else:
        profile.beta = score_result["beta"]
        profile.risk_tolerance = score_result["risk_tolerance"]
        profile.initial_capital = score_result["initial_capital"]
        for key, value in score_result["features"].items():
            profile.features[key] = value
        profile.questionnaire_scores[level] = score_result
        profile.last_updated = _now()
        services["storage"].save_profile(profile)

    return {"message": "问卷提交成功，客户画像已更新。", **_app_state(user, entity_id)}


@app.post("/api/trades/upload")
@app.post("/trades/upload")
async def upload_trades(
    file: UploadFile = File(...),
    window: str = "all",
    customer_id: str | None = None,
    user=Depends(current_user),
) -> dict[str, Any]:
    from app.config import ROLLING_WINDOW_OPTIONS

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        trades_df = _read_upload(file, content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取交易文件: {exc}") from exc

    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    services["storage"].save_trades(entity_id, trades_df)
    profile = services["profile_svc"].update_profile_with_trades(
        entity_id,
        trades_df,
        window_days=ROLLING_WINDOW_OPTIONS.get(window, None),
    )

    lstm_account = None
    if services["lstm_available"]:
        lstm_account = services["lstm_backend"].assign_account(entity_id)

    return {
        "message": "上传成功，画像已更新。",
        "filename": file.filename,
        "tradeCount": len(trades_df),
        "profile": _profile_to_dict(profile),
        "lstmAccount": lstm_account,
        **_app_state(user, entity_id),
    }


@app.get("/api/profile")
@app.get("/profile")
def profile(customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    return _app_state(user, customer_id)


@app.post("/api/recommend")
@app.post("/recommend")
def recommend(payload: RecommendPayload, customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    customer = services["storage"].get_customer(user.user_id, entity_id)
    profile = services["profile_svc"].get_profile(entity_id)
    if profile is None or not profile.features:
        raise HTTPException(status_code=400, detail="请先完成问卷以获取推荐。")

    backend = payload.backend
    if backend not in services["registry"].list_active():
        backend = "fusion" if "fusion" in services["registry"].list_active() else "statistical"

    result = services["recommendation_svc"].recommend(
        entity_id,
        profile.features,
        profile,
        backend_name=backend,
        top_n=payload.top_n,
    )
    recommendations = [
        _recommendation_to_dict(item, result, index)
        for index, item in enumerate(result.top_n)
    ]
    return {
        "customer": {
            "id": entity_id,
            "username": customer.name if customer else entity_id,
            "trade_count": sum(upload["trade_count"] for upload in services["storage"].list_trade_uploads(entity_id)),
            "buy_ratio": 0.0,
            "active_days": 0,
            "turnover_proxy": 0.0,
            "concentration": 0.0,
            "themes": [profile.source],
            "top_symbols": [],
        },
        "backend": result.backend,
        "metricUsed": result.metric_used,
        "popupText": result.popup_text,
        "explanation": result.explanation,
        "recommendations": recommendations,
        "pca": {"explained_variance": []},
    }


@app.post("/api/trends")
@app.post("/trends")
def trends(payload: TrendPayload, customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    from app.services.trend_service import compute_user_trend

    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    profile = services["profile_svc"].get_profile(entity_id)
    customer_trend = None
    user_trades = services["storage"].load_trades(entity_id)
    if user_trades is not None and not user_trades.empty:
        customer_trend = compute_user_trend(
            entity_id,
            user_trades,
            initial_capital=_safe_float(getattr(profile, "initial_capital", 0.0), 0.0),
        )

    strategy_trends = {}
    for strategy_id in payload.strategy_ids:
        strategy_trends[strategy_id] = _cached_strategy_trend(strategy_id)

    return {
        "customerTrend": customer_trend,
        "strategyTrends": strategy_trends,
    }


@app.get("/api/stability")
@app.get("/stability")
def stability(customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    from app.models.user import UserProfile

    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    profile = services["profile_svc"].get_profile(entity_id)
    if profile is None:
        raise HTTPException(status_code=400, detail="请先完成问卷。")

    trades_df = services["storage"].load_trades(entity_id)
    if trades_df is None or len(trades_df) < 10:
        return {
            "ready": False,
            "message": "请先上传足够的交易数据（至少 10 笔）。",
            "windows": [],
            "backendComparison": [],
            "conclusion": "",
        }

    windows = {"全量": None, "最近 120 天": 120, "最近 60 天": 60, "最近 30 天": 30}
    window_rows = []
    for label, days in windows.items():
        try:
            filtered = trades_df
            if days is not None:
                date_col = trades_df.columns[0]
                filtered = trades_df.copy()
                filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
                max_date = filtered[date_col].max()
                cutoff = max_date - pd.Timedelta(days=days)
                filtered = filtered[filtered[date_col] >= cutoff]
                if len(filtered) < 5:
                    window_rows.append({"window": label, "top1": "数据不足", "similarity": None, "count": len(filtered)})
                    continue

            features = services["extractor"].extract_user_features(filtered)
            temp_profile = UserProfile(user_id=entity_id, beta=profile.beta, features=features, confidence_level="high")
            rec = services["recommendation_svc"].recommend(entity_id, features, temp_profile)
            top1 = rec.top_n[0] if rec.top_n else None
            window_rows.append({
                "window": label,
                "top1": top1["strategy"] if top1 else "—",
                "similarity": round(top1["similarity"] * 100, 2) if top1 else None,
                "count": len(filtered),
            })
        except Exception as exc:
            window_rows.append({"window": label, "top1": f"错误: {exc}", "similarity": None, "count": 0})

    comparison_rows = []
    active_backends = services["registry"].list_active()
    features = services["extractor"].extract_user_features(trades_df)
    temp_profile = UserProfile(user_id=entity_id, beta=profile.beta, features=features, confidence_level="high")
    for backend_name in active_backends:
        try:
            rec = services["recommendation_svc"].recommend(
                entity_id,
                features,
                temp_profile,
                backend_name=backend_name,
                top_n=3,
            )
            for item in rec.top_n:
                comparison_rows.append({
                    "backend": backend_name,
                    "backendLabel": BACKEND_DISPLAY.get(backend_name, backend_name),
                    "rank": item["rank"],
                    "strategy": item["strategy"],
                    "similarity": round(item["similarity"] * 100, 2),
                })
        except Exception as exc:
            comparison_rows.append({
                "backend": backend_name,
                "backendLabel": BACKEND_DISPLAY.get(backend_name, backend_name),
                "rank": None,
                "strategy": f"错误: {exc}",
                "similarity": None,
            })

    top1s = [row["top1"] for row in window_rows if row["top1"] not in ("数据不足", "—") and not str(row["top1"]).startswith("错误")]
    if len(set(top1s)) > 1:
        conclusion = f"不同窗口的 Top-1 策略发生变化：{', '.join(sorted(set(top1s)))}。"
    elif top1s:
        conclusion = f"各窗口下 Top-1 策略一致：{top1s[0]}。"
    else:
        conclusion = "暂无足够结果形成稳定性结论。"

    return {
        "ready": True,
        "message": "",
        "windows": window_rows,
        "backendComparison": comparison_rows,
        "conclusion": conclusion,
    }


@app.patch("/api/settings")
@app.patch("/settings")
def update_settings(payload: SettingsPayload, customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    services = _services()
    entity_id = _customer_entity_id(user, customer_id)
    profile = services["profile_svc"].get_profile(entity_id)
    if profile is None:
        raise HTTPException(status_code=400, detail="请先完成问卷。")

    if payload.beta is not None:
        profile = services["profile_svc"].update_beta(entity_id, payload.beta)
    if payload.backend is not None:
        if payload.backend not in services["registry"].list_active():
            raise HTTPException(status_code=400, detail=f"未知匹配后端: {payload.backend}")
        profile.matching_backend = payload.backend
        services["storage"].save_profile(profile)
    if payload.fusion_alpha is not None:
        services["fusion_backend"].set_alpha(payload.fusion_alpha)

    return {"message": "设置已更新。", **_app_state(user, entity_id)}


@app.post("/api/trades/clear")
@app.post("/trades/clear")
def clear_trades(customer_id: str | None = None, user=Depends(current_user)) -> dict[str, Any]:
    entity_id = _customer_entity_id(user, customer_id)
    _services()["profile_svc"].clear_trade_data(entity_id)
    return {"message": "交易数据已清除。", **_app_state(user, entity_id)}
