"""
用户画像-投资策略匹配：Streamlit 网页应用

启动方式: streamlit run app.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import streamlit as st

# ============================================================
# 初始化服务（缓存到 st.session_state）
# ============================================================

@st.cache_resource
def init_services():
    """初始化所有服务，全局单例"""
    from app.services.storage import StorageService
    from app.services.auth import AuthService
    from app.services.questionnaire import QuestionnaireService
    from app.services.feature_extractor import FeatureExtractor
    from app.services.profile import ProfileService
    from app.services.matching_backend import BackendRegistry
    from app.services.backends.statistical import StatisticalBackend
    from app.services.backends.lstm import LSTMBackend
    from app.services.backends.fusion import FusionBackend
    from app.services.recommendation import RecommendationService
    from app.services.popup_generator import PopupGenerator

    # 基础服务
    storage = StorageService()
    auth = AuthService(storage)
    extractor = FeatureExtractor()

    # 加载策略数据
    strategy_features, strategy_nav = _load_strategy_data(extractor)

    # 问卷服务
    feature_means = extractor.get_feature_means(strategy_features)
    questionnaire_svc = QuestionnaireService(strategy_mean_features=feature_means)

    # 画像服务
    profile_svc = ProfileService(storage, extractor)

    # 匹配后端注册
    registry = BackendRegistry()
    stat_backend = StatisticalBackend()
    stat_backend.fit(strategy_features, strategy_nav)
    registry.register(stat_backend)

    # LSTM 后端
    lstm_backend = LSTMBackend()
    lstm_available = True
    try:
        lstm_backend.fit(strategy_features, strategy_nav)
    except FileNotFoundError:
        lstm_available = False
    registry.register(lstm_backend)

    # 融合后端
    fusion_backend = FusionBackend(stat_backend, lstm_backend)
    if lstm_available:
        fusion_backend.fit(strategy_features, strategy_nav)
    registry.register(fusion_backend)

    # 推荐服务
    popup_gen = PopupGenerator()
    recommendation_svc = RecommendationService(registry, popup_gen)

    # 设置策略净值信息（用于弹窗话术）
    nav_info = _compute_strategy_nav_info(strategy_nav)
    recommendation_svc.set_strategy_nav_info(nav_info)

    return {
        "storage": storage,
        "auth": auth,
        "questionnaire_svc": questionnaire_svc,
        "extractor": extractor,
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
        "nav_info": nav_info,
    }


def _load_strategy_data(extractor):
    """加载策略特征和净值数据"""
    from app.config import STRATEGY_DATA_DIR
    import pandas as pd

    strategy_features = {}
    strategy_nav = {}

    if not STRATEGY_DATA_DIR.exists():
        return strategy_features, strategy_nav

    for dir_path in sorted(STRATEGY_DATA_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        strategy_id = dir_path.name

        # 加载净值
        dv_file = dir_path / "daily_value.csv"
        if dv_file.exists():
            nav_df = pd.read_csv(dv_file)
            nav_df["date"] = pd.to_datetime(nav_df["date"])
            nav_df = nav_df.sort_values("date").reset_index(drop=True)
            strategy_nav[strategy_id] = nav_df

        # 加载交易
        trades_file = dir_path / "trades.csv"
        if trades_file.exists() and strategy_id in strategy_nav:
            trades_df = pd.read_csv(trades_file)
            trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"])
            try:
                features = extractor.extract_strategy_features(
                    strategy_nav[strategy_id], trades_df
                )
                strategy_features[strategy_id] = features
            except Exception:
                pass

    return strategy_features, strategy_nav


def _compute_strategy_nav_info(strategy_nav: dict) -> dict:
    """计算策略年化收益和最大回撤"""
    nav_info = {}
    for sid, nav_df in strategy_nav.items():
        if "nav" not in nav_df.columns or len(nav_df) < 10:
            nav_info[sid] = {"annual_return": 0.0, "max_drawdown": 0.0}
            continue

        nav = nav_df["nav"].values
        # 年化收益
        n_days = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days
        total_ret = (nav[-1] - nav[0]) / nav[0]
        ann_ret = ((1 + total_ret) ** (365 / max(n_days, 1)) - 1) * 100

        # 最大回撤
        peak = nav[0]
        max_dd = 0.0
        for v in nav:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)

        nav_info[sid] = {
            "annual_return": ann_ret,
            "max_drawdown": max_dd * 100,
        }

    return nav_info


# 初始化服务
services = init_services()


# ============================================================
# 页面路由
# ============================================================

def main():
    st.set_page_config(
        page_title="投资策略匹配推荐系统",
        page_icon="📈",
        layout="wide",
    )

    # 检查登录状态
    if "current_user" not in st.session_state:
        show_login_page()
        return

    # 侧边栏
    user = st.session_state["current_user"]
    st.sidebar.title("策略匹配推荐系统")
    st.sidebar.success(f"登录中: {user.username}")

    # 获取画像置信度
    profile = services["profile_svc"].get_profile(user.user_id)
    if profile:
        conf_map = {"low": "低", "medium": "中", "high": "高"}
        st.sidebar.info(f"画像置信度: {conf_map.get(profile.confidence_level, '无')}")

    # 使用 session_state 管理当前页面，支持按钮跳转
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "📋 首页"

    page = st.sidebar.radio(
        "导航",
        ["📋 首页", "📝 完善问卷", "📤 上传交易数据", "📊 我的画像", "🎯 推荐策略", "📐 匹配稳定性", "⚙️ 设置"],
        index=["📋 首页", "📝 完善问卷", "📤 上传交易数据", "📊 我的画像", "🎯 推荐策略", "📐 匹配稳定性", "⚙️ 设置"].index(st.session_state["nav_page"]),
        key="nav_radio",
    )
    st.session_state["nav_page"] = page

    if page == "📋 首页":
        show_dashboard()
    elif page == "📝 完善问卷":
        show_questionnaire_page()
    elif page == "📤 上传交易数据":
        show_upload_page()
    elif page == "📊 我的画像":
        show_profile_page()
    elif page == "🎯 推荐策略":
        show_recommendation_page()
    elif page == "📐 匹配稳定性":
        show_stability_page()
    elif page == "⚙️ 设置":
        show_settings_page()

    # 退出登录
    st.sidebar.divider()
    if st.sidebar.button("🚪 退出登录"):
        del st.session_state["current_user"]
        st.rerun()


# ============================================================
# 登录/注册页面
# ============================================================

def show_login_page():
    st.title("投资策略匹配推荐系统")
    st.caption("基于行为特征与 PCA 的策略推荐引擎")

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", use_container_width=True):
            ok, msg, user = services["auth"].login(username, password)
            if ok:
                st.session_state["current_user"] = user
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_register:
        new_user = st.text_input("用户名", key="reg_username")
        new_pass = st.text_input("密码", type="password", key="reg_password")
        new_pass2 = st.text_input("确认密码", type="password", key="reg_password2")
        if st.button("注册", use_container_width=True):
            if new_pass != new_pass2:
                st.error("两次密码不一致")
            else:
                ok, msg, user = services["auth"].register(new_user, new_pass)
                if ok:
                    st.success(msg)
                    st.info("请前往「登录」标签登录")
                else:
                    st.error(msg)


# ============================================================
# Dashboard
# ============================================================

def show_dashboard():
    user = st.session_state["current_user"]
    st.title(f"欢迎回来，{user.username}！")

    profile = services["profile_svc"].get_profile(user.user_id)
    completed_levels = services["storage"].list_completed_levels(user.user_id)
    trade_uploads = services["storage"].list_trade_uploads(user.user_id)

    # 当前状态
    st.subheader("当前状态")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        levels_text = " | ".join([
            f"L1 {'✓' if 'L1' in completed_levels else '✗'}",
            f"L2 {'✓' if 'L2' in completed_levels else '✗'}",
            f"L3 {'✓' if 'L3' in completed_levels else '✗'}",
        ])
        st.metric("问卷完成度", levels_text)

    with col2:
        if profile:
            st.metric("β 值", f"{profile.beta:.2f}")
        else:
            st.metric("β 值", "未设置")

    with col3:
        if profile:
            conf_map = {"low": "低", "medium": "中", "high": "高"}
            st.metric("画像置信度", conf_map.get(profile.confidence_level, "无"))
        else:
            st.metric("画像置信度", "无")

    with col4:
        st.metric("上传次数", len(trade_uploads))

    # 快速开始
    st.subheader("快速开始")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 完善问卷", use_container_width=True):
            st.session_state["nav_page"] = "📝 完善问卷"
            st.rerun()
    with col2:
        if st.button("📤 上传交易数据", use_container_width=True):
            st.session_state["nav_page"] = "📤 上传交易数据"
            st.rerun()
    with col3:
        if st.button("🎯 查看推荐", use_container_width=True):
            st.session_state["nav_page"] = "🎯 推荐策略"
            st.rerun()

    # 快速推荐
    if profile and profile.features:
        st.subheader("快速推荐（基于当前画像）")
        result = services["recommendation_svc"].recommend(
            user.user_id, profile.features, profile, backend_name="statistical"
        )
        if result.top_n:
            for rec in result.top_n:
                sim_pct = rec["similarity"] * 100
                if sim_pct >= 0:
                    st.info(f"#{rec['rank']} {rec['strategy']} — 匹配度 {sim_pct:.1f}%")
                else:
                    st.warning(f"#{rec['rank']} {rec['strategy']} — 匹配度 {sim_pct:.1f}%（风格相反）")

            # 弹窗话术
            st.divider()
            st.caption("弹窗话术预览：")
            st.info(result.popup_text)

            # 完整话术按钮
            if st.button("复制话术到剪贴板"):
                st.toast("话术已展示，请手动复制")
    else:
        st.info("完成 Level 1 问卷后即可查看推荐。")


# ============================================================
# 问卷页面
# ============================================================

def show_questionnaire_page():
    user = st.session_state["current_user"]
    st.title("完善投资问卷")

    completed_levels = services["storage"].list_completed_levels(user.user_id)
    questionnaire_svc = services["questionnaire_svc"]

    levels = ["L1", "L2", "L3"]
    level_titles = {
        "L1": "基础风险评估",
        "L2": "投资偏好深化",
        "L3": "交易行为进阶",
    }
    level_times = {"L1": "2分钟", "L2": "5分钟", "L3": "8分钟"}
    level_required = {"L1": True, "L2": False, "L3": False}

    for level in levels:
        is_done = level in completed_levels
        required = level_required[level]

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            label = f"{level}: {level_titles[level]}"
            if is_done:
                st.success(f"{label} ✓ 已完成")
            else:
                st.info(f"{label} {'(必填)' if required else '(可选)'} ({level_times[level]})")

        with col3:
            btn_label = "重新填写" if is_done else "开始填写"
            if st.button(btn_label, key=f"btn_{level}"):
                st.session_state[f"show_{level}"] = True

        # 展开问卷
        if st.session_state.get(f"show_{level}", False) or (not is_done and required):
            render_questionnaire(level, questionnaire_svc, is_done)


def render_questionnaire(level, questionnaire_svc, is_done):
    """渲染单个问卷"""
    qn = questionnaire_svc.get_questionnaire(level)
    st.subheader(qn.title)
    st.caption(qn.description)

    answers = {}
    for q in qn.questions:
        st.markdown(f"**{q.q_id}. {q.text}**")

        if q.q_type == "single_choice":
            choice = st.radio(
                "请选择",
                q.options,
                key=f"{q.q_id}_{level}",
                horizontal=False,
                label_visibility="collapsed",
                index=None,
            )
            # 提取选项字母 (A, B, C...)
            answers[q.q_id] = choice[0].upper() if choice else ""

        elif q.q_type == "slider":
            if "亏损容忍度" in q.text:
                val = st.slider("百分比 (%)", 1, 20, 10, key=f"{q.q_id}_{level}")
                answers[q.q_id] = val
            else:
                val = st.slider("百分比 (%)", 0, 100, 50, key=f"{q.q_id}_{level}")
                answers[q.q_id] = val

        elif q.q_type == "number_input":
            val = st.number_input("请输入数值", min_value=0, max_value=999, value=5, key=f"{q.q_id}_{level}")
            answers[q.q_id] = val

        elif q.q_type == "multi_select":
            selected = st.multiselect("请选择（可多选）", q.options, key=f"{q.q_id}_{level}")
            answers[q.q_id] = [s[0].upper() for s in selected]

        st.divider()

    if st.button("提交问卷", key=f"submit_{level}"):
        # 检查必填
        missing = []
        for q in qn.questions:
            if q.q_id not in answers or (
                isinstance(answers.get(q.q_id), str) and not answers[q.q_id]
            ):
                missing.append(q.q_id)

        if missing:
            st.error(f"请回答所有问题。未回答: {', '.join(missing)}")
            return

        # 评分
        score_result = questionnaire_svc.score_answers(level, answers)

        # 保存问卷结果
        services["storage"].save_questionnaire_results(
            st.session_state["current_user"].user_id, level, answers
        )

        # 更新或创建画像
        profile = services["profile_svc"].get_profile(
            st.session_state["current_user"].user_id
        )

        if profile is None:
            services["profile_svc"].create_profile_from_questionnaire(
                st.session_state["current_user"].user_id, score_result
            )
        else:
            # 更新 beta 和 risk_tolerance
            profile.beta = score_result["beta"]
            profile.risk_tolerance = score_result["risk_tolerance"]
            profile.initial_capital = score_result["initial_capital"]

            # 更新问卷已覆盖的特征
            for k, v in score_result["features"].items():
                profile.features[k] = v

            profile.questionnaire_scores[level] = score_result
            profile.last_updated = profile.last_updated  # 保持原时间
            services["storage"].save_profile(profile)

        st.success("问卷提交成功！画像已更新。")
        st.session_state[f"show_{level}"] = False
        st.rerun()


# ============================================================
# 上传交易数据页面
# ============================================================

def show_upload_page():
    user = st.session_state["current_user"]
    st.title("上传交易数据")

    st.caption("支持格式: Excel (.xlsx) 或 CSV")
    st.caption("要求列: 交易日期、操作类型、股票代码、价格、数量")

    uploaded_file = st.file_uploader(
        "选择文件",
        type=["xlsx", "csv"],
        key="trade_upload",
    )

    # 窗口选择
    window_options = {"全部数据": None, "最近 120 天": 120, "最近 60 天": 60, "最近 30 天": 30}
    selected_window = st.selectbox(
        "分析时间窗口",
        list(window_options.keys()),
        help="选择用于分析的交易数据时间范围",
    )

    if uploaded_file and st.button("上传并分析", use_container_width=True):
        with st.spinner("正在处理..."):
            try:
                # 读取文件
                if uploaded_file.name.endswith(".csv"):
                    trades_df = pd.read_csv(uploaded_file)
                else:
                    trades_df = pd.read_excel(uploaded_file)

                # 保存交易数据
                services["storage"].save_trades(user.user_id, trades_df)

                # 提取特征并更新画像
                profile = services["profile_svc"].update_profile_with_trades(
                    user.user_id,
                    trades_df,
                    window_days=window_options[selected_window],
                )

                # 分配 LSTM 账户名（首次上传时自动分配）
                lstm_account_msg = ""
                if services["lstm_available"]:
                    assigned = services["lstm_backend"].assign_account(user.user_id)
                    if assigned:
                        lstm_account_msg = f"，已分配 LSTM 账户: {assigned}"
                    else:
                        lstm_account_msg = "（LSTM 账户槽位已满，仅支持前 3 个上传用户）"

                st.success(
                    f"上传成功！画像已更新（第 {profile.update_count} 次更新，"
                    f"置信度: {profile.confidence_level}）{lstm_account_msg}"
                )

                # 显示提取的特征
                with st.expander("查看提取的特征"):
                    st.json(profile.features, expanded=False)

                st.rerun()

            except Exception as e:
                st.error(f"处理失败: {e}")

    # 上传历史
    trade_uploads = services["storage"].list_trade_uploads(user.user_id)
    if trade_uploads:
        st.divider()
        st.subheader("上传历史")
        for upload in trade_uploads:
            st.text(f"{upload['upload_date']}  |  {upload['trade_count']} 笔  |  已处理 ✓")

    profile = services["profile_svc"].get_profile(user.user_id)
    if profile:
        st.caption(f"当前画像更新次数: {profile.update_count} 次 (置信度: {profile.confidence_level})")


# ============================================================
# 画像页面
# ============================================================

def show_profile_page():
    user = st.session_state["current_user"]
    st.title("我的投资画像")

    profile = services["profile_svc"].get_profile(user.user_id)
    if profile is None:
        st.info("请先完成问卷以初始化您的画像。")
        return

    # 画像来源
    st.subheader("画像信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("β 值 (行为特征权重)", f"{profile.beta:.2f}")
    with col2:
        conf_map = {"low": "低", "medium": "中", "high": "高"}
        st.metric("置信度", conf_map.get(profile.confidence_level, "无"))
    with col3:
        st.metric("画像来源", profile.source)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("风险承受能力", f"{profile.risk_tolerance}/5")
    with col5:
        st.metric("画像更新次数", profile.update_count)
    with col6:
        st.metric("匹配后端", profile.matching_backend)

    # 特征雷达图
    st.subheader("12 维特征对比")
    if profile.features:
        _plot_feature_radar(profile)

    # 变化轨迹
    if profile.history and len(profile.history) > 1:
        st.subheader("画像变化轨迹")
        _plot_history_trajectory(profile)


def _plot_feature_radar(profile):
    """绘制特征雷达图"""
    import matplotlib.pyplot as plt
    import numpy as np

    strategy_features = services["strategy_features"]
    from pipeline import MATCH_FEATURES

    name_map = {
        "holding_period": "持仓周期", "turnover_rate": "换手率",
        "buy_sell_ratio": "买卖对称", "hhi_concentration": "持仓集中",
        "disposition_effect": "处置效应", "positive_trade_ratio": "胜率",
        "etf_ratio": "ETF偏好", "avg_price_preference": "价格偏好",
        "position_uniformity": "分仓均匀", "avg_loss_magnitude": "亏损幅度",
        "vol_preference": "波动偏好", "trend_preference": "趋势偏好",
    }

    # 归一化到 [0, 1]
    all_vals = [v for sf in strategy_features.values() for v in sf.values()]
    all_vals += list(profile.features.values())
    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1
    range_val = max_val - min_val if max_val != min_val else 1

    user_vals = [(profile.features.get(f, 0) - min_val) / range_val for f in MATCH_FEATURES]
    avg_vals = [
        (np.mean([sf.get(f, 0) for sf in strategy_features.values()]) - min_val) / range_val
        for f in MATCH_FEATURES
    ]

    # 极坐标图
    angles = np.linspace(0, 2 * np.pi, len(MATCH_FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    user_vals += user_vals[:1]
    avg_vals += avg_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.plot(angles, user_vals, "o-", linewidth=2, label="我的画像")
    ax.fill(angles, user_vals, alpha=0.15)
    ax.plot(angles, avg_vals, "--", linewidth=1, label="策略平均", alpha=0.6)

    short_names = [name_map.get(f, f[:8]) for f in MATCH_FEATURES]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, fontsize=8)
    ax.legend(loc="upper right", fontsize=8)

    st.pyplot(fig)
    plt.close()


def _plot_history_trajectory(profile):
    """绘制画像变化轨迹"""
    import matplotlib.pyplot as plt

    history = profile.history
    updates = [h["update"] for h in history]

    # 选择前3个主成分方向（用特征值近似）
    key_features = ["holding_period", "etf_ratio", "turnover_rate"]
    name_map = {"holding_period": "持仓周期", "etf_ratio": "ETF占比", "turnover_rate": "换手率"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, feat in enumerate(key_features):
        vals = [h["features"].get(feat, 0) for h in history]
        axes[i].plot(updates, vals, "o-", linewidth=2)
        axes[i].set_title(name_map[feat])
        axes[i].set_xlabel("更新次数")
        axes[i].grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()


# ============================================================
# 推荐策略页面
# ============================================================

_BACKEND_DISPLAY = {
    "statistical": "PCA 统计方法",
    "lstm": "LSTM 序列风格匹配",
    "fusion": "融合推荐 (统计 + LSTM)",
}


def _display_backend_label(backend_name: str) -> str:
    return _BACKEND_DISPLAY.get(backend_name, backend_name)


def show_recommendation_page():
    user = st.session_state["current_user"]
    st.title("策略推荐")

    profile = services["profile_svc"].get_profile(user.user_id)
    if profile is None or not profile.features:
        st.info("请先完成问卷以获取推荐。")
        return

    # 后端选择
    available_backends = services["registry"].list_active()
    # Default to fusion if available, else first available
    default_backend = "fusion" if "fusion" in available_backends else (available_backends[0] if available_backends else None)
    default_idx = 0
    if default_backend and default_backend in available_backends:
        default_idx = available_backends.index(default_backend)

    backend_choice = st.selectbox(
        "匹配算法",
        available_backends,
        index=default_idx,
        format_func=_display_backend_label,
    )

    if backend_choice:
        result = services["recommendation_svc"].recommend(
            user.user_id, profile.features, profile, backend_name=backend_choice
        )

        st.divider()

        if not result.top_n:
            if backend_choice == "lstm" and not services["lstm_available"]:
                st.warning(
                    "LSTM 后端不可用：DLMethod 预训练文件未找到。\n"
                    "请先在 DLMethod/ 目录下运行 step1-step7 流水线。"
                )
            else:
                st.info("暂无匹配结果。")
            return

        # LSTM 账户映射状态提示
        if backend_choice == "lstm":
            assigned = services["lstm_backend"].get_assigned_accounts()
            st.caption(f"已分配 LSTM 账户: {assigned}")
            if not assigned:
                st.info(
                    "提示：请先上传交易数据以分配 LSTM 账户名。"
                    "系统最多支持 3 个账户的 LSTM 匹配。"
                )

        # 推荐结果
        st.subheader("推荐结果")
        for rec in result.top_n:
            sim_pct = rec["similarity"] * 100
            explanation = result.explanation

            # 融合模式下展示双源排名
            extra_info = ""
            if backend_choice == "fusion" and result.stat_score and result.ml_score:
                stat_s = result.stat_score.get(rec["strategy"], 0)
                ml_s = result.ml_score.get(rec["strategy"], 0)
                stat_r = result.phase1_rank.get(rec["strategy"], "—")
                ml_r = result.phase2_rank.get(rec["strategy"], "—")
                extra_info = f"\n    统计得分: {stat_s * 100:.1f}% (Phase1 排名 #{stat_r})"
                extra_info += f" | LSTM 得分: {ml_s * 100:.1f}% (Phase2 排名 #{ml_r})"
            elif backend_choice == "lstm":
                ml_r = result.phase2_rank.get(rec["strategy"], "—")
                p1_r = result.phase1_rank.get(rec["strategy"], "—")
                extra_info = f"\n    LSTM 排名: #{ml_r} | 特征基线排名: #{p1_r}"

            if sim_pct >= 0:
                with st.expander(f"🥇{'🥈' if rec['rank'] == 2 else '🥉' if rec['rank'] == 3 else ''} #{rec['rank']} {rec['strategy']} — 匹配度 {sim_pct:.1f}%{extra_info}", expanded=(rec['rank'] == 1)):
                    st.markdown(f"**策略**: {rec['strategy']}")
                    st.markdown(f"**匹配度**: {sim_pct:.1f}%")

                    # 融合模式额外信息
                    if backend_choice == "fusion":
                        alpha = explanation.get("fusion_alpha", 0.7)
                        st.caption(
                            f"融合权重: 统计 {alpha * 100:.0f}% + LSTM {(1 - alpha) * 100:.0f}%"
                        )

                    # LSTM SHAP 归因
                    if explanation.get("lstm_shap_top_features"):
                        st.markdown("**LSTM 驱动特征 (SHAP)**:")
                        for sf in explanation["lstm_shap_top_features"]:
                            st.markdown(f"- {sf['feature']} (|SHAP|={sf['mean_abs_shap']:.4f})")

                    # 维度对比
                    if explanation.get("most_similar_dimensions"):
                        st.markdown("**与您最相似的维度**:")
                        for d in explanation["most_similar_dimensions"]:
                            st.markdown(f"- {d['feature']} (差异: {d['diff']:.4f})")

                    if explanation.get("most_different_dimensions"):
                        st.markdown("**差异最大的维度**:")
                        for d in explanation["most_different_dimensions"]:
                            st.markdown(f"- {d['feature']} (差异: {d['diff']:.4f})")
            else:
                st.warning(f"#{rec['rank']} {rec['strategy']} — 匹配度 {sim_pct:.1f}%（风格方向相反）")

        # 弹窗话术
        st.divider()
        st.subheader("弹窗话术")
        st.info(result.popup_text)


# ============================================================
# 匹配稳定性页面
# ============================================================

def show_stability_page():
    user = st.session_state["current_user"]
    st.title("匹配稳定性分析")

    profile = services["profile_svc"].get_profile(user.user_id)
    if profile is None:
        st.info("请先完成问卷。")
        return

    # 加载用户的交易数据
    trades_df = services["storage"].load_trades(user.user_id)
    if trades_df is None or len(trades_df) < 10:
        st.info("请先上传足够的交易数据（至少 10 笔）。")
        return

    st.caption("使用不同时间窗口的交易数据计算推荐结果，观察匹配是否稳定。")

    windows = {"全量": None, "最近 120 天": 120, "最近 60 天": 60, "最近 30 天": 30}

    results = []
    for label, days in windows.items():
        try:
            if days is not None:
                date_col = trades_df.columns[0]
                trades_df[date_col] = pd.to_datetime(trades_df[date_col], errors="coerce")
                max_date = trades_df[date_col].max()
                cutoff = max_date - pd.Timedelta(days=days)
                filtered = trades_df[trades_df[date_col] >= cutoff]
                if len(filtered) < 5:
                    results.append({"window": label, "top1": "数据不足", "similarity": "—", "count": len(filtered)})
                    continue
                features = services["extractor"].extract_user_features(filtered)
            else:
                features = services["extractor"].extract_user_features(trades_df)

            # 临时 profile 用于推荐
            from app.models.user import UserProfile
            temp_profile = UserProfile(
                user_id=user.user_id,
                beta=profile.beta,
                features=features,
                confidence_level="high",
            )

            rec = services["recommendation_svc"].recommend(
                user.user_id, features, temp_profile
            )
            top1 = rec.top_n[0] if rec.top_n else None
            results.append({
                "window": label,
                "top1": top1["strategy"] if top1 else "—",
                "similarity": f"{top1['similarity'] * 100:.1f}%" if top1 else "—",
                "count": len(filtered) if days else len(trades_df),
            })
        except Exception as e:
            results.append({"window": label, "top1": f"错误: {e}", "similarity": "—", "count": 0})

    # 展示结果
    st.subheader("多窗口推荐对比")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # 趋势图
    st.subheader("匹配度趋势")
    valid_results = [r for r in results if r["similarity"] != "—"]
    if valid_results:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        sims = [float(r["similarity"].rstrip("%")) for r in valid_results]
        labels = [r["window"] for r in valid_results]
        ax.bar(labels, sims, color="steelblue")
        ax.set_ylabel("匹配度 (%)")
        ax.set_title("不同窗口下的 Top-1 匹配度")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close()

    # 结论
    top1_strategies = [r["top1"] for r in results if r["top1"] != "数据不足" and r["top1"] != "—"]
    if len(set(top1_strategies)) > 1:
        st.warning(
            f"⚠️ 不同窗口的 Top-1 策略发生变化 ({', '.join(set(top1_strategies))})，"
            f"建议关注近期交易风格是否有显著变化。"
        )
    elif top1_strategies:
        st.success(f"✅ 各窗口下 Top-1 策略一致: {top1_strategies[0]}，匹配稳定。")

    # 多算法路线对比
    st.divider()
    st.subheader("三路线推荐对比")
    st.caption("对比统计方法、LSTM 序列风格匹配、融合推荐三种算法的 Top-3 结果。")

    active_backends = services["registry"].list_active()
    if len(active_backends) >= 2:
        from app.models.user import UserProfile
        features = services["extractor"].extract_user_features(trades_df)
        features["_user_id"] = user.user_id
        temp_profile = UserProfile(
            user_id=user.user_id,
            beta=profile.beta,
            features=features,
            confidence_level="high",
        )

        comparison_rows = []
        for backend_name in active_backends:
            try:
                rec = services["recommendation_svc"].recommend(
                    user.user_id, features, temp_profile,
                    backend_name=backend_name, top_n=3,
                )
                for i, item in enumerate(rec.top_n):
                    comparison_rows.append({
                        "算法": _display_backend_label(backend_name),
                        "排名": f"#{item['rank']}",
                        "策略": item["strategy"],
                        "匹配度": f"{item['similarity'] * 100:.1f}%",
                    })
            except Exception as e:
                comparison_rows.append({
                    "算法": _display_backend_label(backend_name),
                    "排名": "—",
                    "策略": f"错误: {e}",
                    "匹配度": "—",
                })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True)

            # 跨算法 Top-1 一致性检查
            top1_per_backend = {}
            for row in comparison_rows:
                if row["排名"] == "#1":
                    top1_per_backend[row["算法"]] = row["策略"]

            unique_top1 = set(top1_per_backend.values())
            if len(unique_top1) == 1:
                st.success(f"✅ 所有算法 Top-1 一致: {list(unique_top1)[0]}")
            else:
                details = ", ".join(f"{algo}: {s}" for algo, s in top1_per_backend.items())
                st.info(f"不同算法推荐了不同的策略: {details}")


# ============================================================
# 设置页面
# ============================================================

def show_settings_page():
    user = st.session_state["current_user"]
    st.title("设置")

    profile = services["profile_svc"].get_profile(user.user_id)
    if profile is None:
        st.info("请先完成问卷。")
        return

    # 匹配算法选择
    st.subheader("匹配算法")
    available_backends = services["registry"].list_active()
    current_backend = profile.matching_backend if profile.matching_backend in available_backends else available_backends[0]
    selected_backend = st.selectbox(
        "选择默认匹配后端",
        available_backends,
        index=available_backends.index(current_backend) if current_backend in available_backends else 0,
        format_func=_display_backend_label,
    )
    if selected_backend != profile.matching_backend:
        if st.button("应用后端切换"):
            profile.matching_backend = selected_backend
            services["storage"].save_profile(profile)
            st.success(f"默认后端已切换为: {_display_backend_label(selected_backend)}")
            st.rerun()

    # 融合权重（仅融合模式）
    if "fusion" in available_backends:
        st.divider()
        st.subheader("融合权重 α")
        from app.config import FUSION_ALPHA_OPTIONS
        alpha_labels = list(FUSION_ALPHA_OPTIONS.keys())
        alpha_values = list(FUSION_ALPHA_OPTIONS.values())
        current_alpha = services["fusion_backend"]._alpha
        # Find closest label
        closest_idx = min(range(len(alpha_values)), key=lambda i: abs(alpha_values[i] - current_alpha))
        selected_label = st.selectbox(
            "统计方法与 LSTM 的权重比例",
            alpha_labels,
            index=closest_idx,
        )
        new_alpha = FUSION_ALPHA_OPTIONS[selected_label]
        if new_alpha != current_alpha:
            if st.button("应用融合权重"):
                services["fusion_backend"].set_alpha(new_alpha)
                st.success(f"融合权重已更新: 统计 {new_alpha * 100:.0f}% + LSTM {(1 - new_alpha) * 100:.0f}%")
                st.rerun()

    # LSTM 状态
    if services["lstm_available"]:
        st.divider()
        st.subheader("LSTM 后端状态")
        assigned = services["lstm_backend"].get_assigned_accounts()
        if assigned:
            st.caption(f"已分配 LSTM 账户: {assigned}")
        else:
            st.caption("尚无用户分配 LSTM 账户。用户首次上传交易数据后将自动分配。")
        st.caption(
            "LSTM 方法说明：本系统的 LSTM 匹配模块基于小样本弱监督学习训练，"
            "训练标签来自交易画像相似度生成的伪标签。它适合作为统计方法的辅助补充，"
            "用于捕捉交易序列风格的相似性。最终推荐仍结合统计方法加权得出。"
        )

    # β 调整
    st.divider()
    st.subheader("β 超参数")
    st.caption(f"当前: {profile.beta:.2f} (行为特征权重 {profile.beta * 100:.0f}%)")

    new_beta = st.slider(
        "手动调整 β",
        0.0, 1.0, profile.beta, 0.05,
        help="β=0.0: 仅关注资产偏好; β=1.0: 仅关注行为特征",
    )
    if st.button("应用 β 值"):
        services["profile_svc"].update_beta(user.user_id, new_beta)
        st.success(f"β 已更新为 {new_beta:.2f}")
        st.rerun()

    # 数据管理
    st.divider()
    st.subheader("数据管理")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("导出画像数据"):
            st.json(profile.to_dict(), expanded=False)
    with col2:
        if st.button("清除交易数据"):
            services["profile_svc"].clear_trade_data(user.user_id)
            st.success("交易数据已清除。")
            st.rerun()

    # 关于
    st.divider()
    st.subheader("关于")
    st.markdown(f"""
- **统计方法**: PCA + 径向惩罚余弦 (λ=1.0)
- **LSTM 方法**: BiLSTM 128 维序列风格匹配 (DLMethod 团队)
- **融合公式**: final = α × stat_norm + (1-α) × ml_norm
- **参考文档**: `report.md`, `DLMethod/ML_INTEGRATION_GUIDE.md`
- **版本**: App V2.0 / Pipeline V2.0
    """)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
