"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  AlertCircle,
  BarChart3,
  ClipboardList,
  FileSpreadsheet,
  Home,
  LineChart as LineChartIcon,
  LogOut,
  RefreshCw,
  Settings,
  SlidersHorizontal,
  Trash2,
  Upload,
  UserRound,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  clearTrades,
  fetchQuestionnaires,
  fetchRecommendations,
  fetchSession,
  fetchStability,
  fetchTrends,
  login,
  logout,
  register,
  submitQuestionnaire,
  updateSettings,
  uploadTrades,
  type AppState,
  type Question,
  type Questionnaire,
  type RecommendResponse,
  type StabilityResponse,
  type TrendComparisonResponse,
} from "@/lib/api";
import { CollapsibleSidebar } from "@/components/CollapsibleSidebar";
import { FontSizeControl, getFontScale, type FontScaleKey } from "@/components/FontSizeControl";
import { RadarChart } from "@/components/RadarChart";
import { RecommendationBoard } from "@/components/RecommendationBoard";

type PageKey = "home" | "questionnaire" | "upload" | "profile" | "recommend" | "stability" | "settings";

const NAV_ITEMS: Array<{ key: PageKey; label: string; icon: React.ReactNode }> = [
  { key: "home", label: "首页", icon: <Home size={16} /> },
  { key: "questionnaire", label: "完善问卷", icon: <ClipboardList size={16} /> },
  { key: "upload", label: "上传交易", icon: <Upload size={16} /> },
  { key: "profile", label: "我的画像", icon: <UserRound size={16} /> },
  { key: "recommend", label: "推荐策略", icon: <LineChartIcon size={16} /> },
  { key: "stability", label: "匹配稳定性", icon: <BarChart3 size={16} /> },
  { key: "settings", label: "设置", icon: <Settings size={16} /> },
];

const CONFIDENCE_LABELS: Record<string, string> = {
  low: "低",
  medium: "中",
  high: "高",
};

const TREND_COLORS = ["#0f766e", "#c48112", "#c2415b", "#6d5bd0", "#0f9f6e", "#64748b"];
const QUESTIONNAIRE_LEVELS: Array<Questionnaire["level"]> = ["L1", "L2", "L3"];

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function pickDefaultBackend(state: AppState | null): string {
  if (!state || state.backends.length === 0) return "statistical";
  const preferred = state.profile?.matchingBackend;
  if (preferred && state.backends.some((item) => item.name === preferred)) return preferred;
  if (state.backends.some((item) => item.name === "fusion")) return "fusion";
  return state.backends[0].name;
}

function getDefaultAnswer(question: Question): string | number | string[] | undefined {
  if (question.type === "slider") {
    return question.text.includes("亏损容忍度") ? 10 : 50;
  }
  if (question.type === "number_input") {
    return 5;
  }
  return undefined;
}

function buildQuestionnairePayload(
  questionnaire: Questionnaire,
  answers: Record<string, string | number | string[]>,
): Record<string, string | number | string[]> {
  return questionnaire.questions.reduce<Record<string, string | number | string[]>>((payload, question) => {
    const answer = answers[question.id] ?? getDefaultAnswer(question);
    if (answer !== undefined) payload[question.id] = answer;
    return payload;
  }, {});
}

function findNextQuestionnaire(
  questionnaires: Questionnaire[],
  completedLevels: string[],
): Questionnaire | null {
  const completed = new Set(completedLevels);
  return QUESTIONNAIRE_LEVELS
    .map((level) => questionnaires.find((item) => item.level === level))
    .find((item): item is Questionnaire => Boolean(item && !completed.has(item.level))) ?? null;
}

function AuthScreen({
  onAuthed,
}: {
  onAuthed: (state: AppState) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    setMessage("");
    try {
      const result = mode === "login"
        ? await login(username, password)
        : await register(username, password);
      onAuthed(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "认证失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-screen">
      <section className="auth-panel">
        <p className="eyebrow">Investment Matching</p>
        <h1>投资策略匹配推荐系统</h1>
        <div className="auth-tabs">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>登录</button>
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>注册</button>
        </div>
        <label className="field-label">
          用户名
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </label>
        <label className="field-label">
          密码
          <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} />
        </label>
        {message ? <div className="error-strip"><AlertCircle size={16} />{message}</div> : null}
        <button className="primary-button" type="button" onClick={submit} disabled={busy}>
          {busy ? "处理中..." : mode === "login" ? "登录" : "注册并进入"}
        </button>
      </section>
    </main>
  );
}

function TrendComparisonChart({
  data,
  selectedStrategyIds,
}: {
  data: TrendComparisonResponse | null;
  selectedStrategyIds: string[];
}) {
  const chart = useMemo(() => {
    const rows = new Map<string, Record<string, string | number | null>>();
    const ensure = (date: string) => {
      if (!rows.has(date)) rows.set(date, { date });
      return rows.get(date)!;
    };

    for (const point of data?.customerTrend?.trend ?? []) {
      ensure(point.date).customer = point.cumulativeReturn;
    }
    selectedStrategyIds.forEach((strategyId, index) => {
      const series = data?.strategyTrends[strategyId];
      for (const point of series?.trend ?? []) {
        ensure(point.date)[`strategy_${index}`] = point.cumulativeReturn;
      }
    });

    return Array.from(rows.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [data, selectedStrategyIds]);

  const hasCustomer = Boolean(data?.customerTrend?.trend?.length);
  const hasStrategy = selectedStrategyIds.some((id) => (data?.strategyTrends[id]?.trend?.length ?? 0) > 0);

  if (!hasCustomer && !hasStrategy) {
    return (
      <div className="trend-chart-container trend-empty">
        <LineChartIcon size={36} strokeWidth={1.5} />
        <p>暂无可展示的收益曲线。</p>
      </div>
    );
  }

  return (
    <div className="trend-chart-container">
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chart} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8edf5" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#667085" }} tickLine={false} minTickGap={60} />
          <YAxis tick={{ fontSize: 11, fill: "#667085" }} tickFormatter={(value: number) => `${value}%`} width={54} />
          <Tooltip formatter={(value) => formatPct(typeof value === "number" ? value : Number(value))} />
          <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
          {hasCustomer ? (
            <Line type="monotone" dataKey="customer" name="我的账户" stroke="#2563eb" strokeWidth={3} dot={false} connectNulls />
          ) : null}
          {selectedStrategyIds.map((strategyId, index) => (
            <Line
              key={strategyId}
              type="monotone"
              dataKey={`strategy_${index}`}
              name={strategyId}
              stroke={TREND_COLORS[index % TREND_COLORS.length]}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function HomePage() {
  const [appState, setAppState] = useState<AppState | null | undefined>(undefined);
  const [questionnaires, setQuestionnaires] = useState<Questionnaire[]>([]);
  const [activePage, setActivePage] = useState<PageKey>("home");
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [selectedLevel, setSelectedLevel] = useState<"L1" | "L2" | "L3">("L1");
  const [answers, setAnswers] = useState<Record<string, string | number | string[]>>({});
  const [uploadWindow, setUploadWindow] = useState("all");
  const [backend, setBackend] = useState("statistical");
  const [topN, setTopN] = useState(5);
  const [recommendation, setRecommendation] = useState<RecommendResponse | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null);
  const [trendStrategyIds, setTrendStrategyIds] = useState<string[]>([]);
  const [trendData, setTrendData] = useState<TrendComparisonResponse | null>(null);
  const [stability, setStability] = useState<StabilityResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const trendSectionRef = useRef<HTMLElement | null>(null);
  const [fontScaleKey, setFontScaleKey] = useState<FontScaleKey>(() => {
    if (typeof window === "undefined") return "md";
    return (window.localStorage.getItem("investment-font-scale") as FontScaleKey) ?? "md";
  });

  useEffect(() => {
    async function load() {
      try {
        const state = await fetchSession();
        setAppState(state);
        setBackend(pickDefaultBackend(state));
      } catch {
        setAppState(null);
      }
    }
    void load();
  }, []);

  useEffect(() => {
    if (!appState) return;
    void fetchQuestionnaires().then((data) => setQuestionnaires(data.questionnaires));
  }, [appState?.user.userId]);

  useEffect(() => {
    if (!appState) return;
    setBackend(pickDefaultBackend(appState));
  }, [appState?.profile?.matchingBackend, appState?.backends.length]);

  useEffect(() => {
    if (trendStrategyIds.length === 0 && !recommendation) return;
    let cancelled = false;
    async function loadTrend() {
      try {
        const data = await fetchTrends(trendStrategyIds);
        if (!cancelled) setTrendData(data);
      } catch {
        if (!cancelled) setTrendData(null);
      }
    }
    void loadTrend();
    return () => {
      cancelled = true;
    };
  }, [trendStrategyIds, recommendation]);

  useEffect(() => {
    if (activePage !== "recommend" || !recommendation) return;
    const timer = window.setTimeout(() => {
      trendSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [activePage, recommendation, trendStrategyIds.join("|")]);

  function setFlash(nextMessage: string, nextError = "") {
    setMessage(nextMessage);
    setError(nextError);
  }

  function setFontScale(next: FontScaleKey) {
    setFontScaleKey(next);
    window.localStorage.setItem("investment-font-scale", next);
  }

  async function refreshState() {
    try {
      const state = await fetchSession();
      setAppState(state);
      setFlash("数据已刷新。");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "刷新失败");
    }
  }

  async function refreshRecommendationAndTrends(nextBackend = backend, nextTopN = topN) {
    const data = await fetchRecommendations({ backend: nextBackend, topN: nextTopN });
    const first = data.recommendations[0]?.strategy_id ?? null;
    const nextTrendIds = first ? [first] : [];
    setRecommendation(data);
    setSelectedStrategyId(first);
    setTrendStrategyIds(nextTrendIds);
    setTrendData(await fetchTrends(nextTrendIds));
    return data;
  }

  async function handleLogout() {
    await logout();
    setAppState(null);
    setRecommendation(null);
    setTrendData(null);
  }

  const selectedQuestionnaire = questionnaires.find((item) => item.level === selectedLevel) ?? questionnaires[0];
  const sortedRecommendations = recommendation?.recommendations ?? [];
  const profile = appState?.profile ?? null;
  const completedText = appState?.completedLevels.length
    ? appState.completedLevels.join(" / ")
    : "未完成";

  async function handleQuestionnaireSubmit() {
    if (!selectedQuestionnaire) return;
    setBusy(true);
    setFlash("");
    try {
      const payload = buildQuestionnairePayload(selectedQuestionnaire, answers);
      const state = await submitQuestionnaire(selectedQuestionnaire.level, payload);
      setAppState(state);
      const fresh = await fetchQuestionnaires();
      setQuestionnaires(fresh.questionnaires);
      setAnswers({});
      const next = findNextQuestionnaire(fresh.questionnaires, state.completedLevels);
      if (next) {
        setSelectedLevel(next.level);
        setActivePage("questionnaire");
        setFlash(`${state.message} 请继续完成 ${next.level}。`);
      } else {
        setActivePage("upload");
        setFlash(`${state.message} 三份问卷已完成，请上传交易数据。`);
      }
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "问卷提交失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setFlash("");
    try {
      const state = await uploadTrades(file, uploadWindow);
      setAppState(state);
      if (state.profile) {
        try {
          await refreshRecommendationAndTrends(pickDefaultBackend(state), topN);
          setBackend(pickDefaultBackend(state));
          setActivePage("recommend");
          setFlash(`${state.filename}: ${state.message} 已自动刷新推荐和收益曲线。`);
        } catch (recommendError) {
          setFlash(
            `${state.filename}: ${state.message}`,
            recommendError instanceof Error ? recommendError.message : "推荐和收益曲线自动刷新失败",
          );
        }
      } else {
        setFlash(`${state.filename}: ${state.message}`);
      }
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function runRecommendation() {
    setBusy(true);
    setFlash("");
    try {
      await refreshRecommendationAndTrends();
      setActivePage("recommend");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "推荐计算失败");
    } finally {
      setBusy(false);
    }
  }

  async function loadStability() {
    setBusy(true);
    setFlash("");
    try {
      const data = await fetchStability();
      setStability(data);
      setActivePage("stability");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "稳定性分析失败");
    } finally {
      setBusy(false);
    }
  }

  async function applySettings(next: { beta?: number; backend?: string; fusionAlpha?: number }) {
    setBusy(true);
    setFlash("");
    try {
      const state = await updateSettings(next);
      setAppState(state);
      setFlash(state.message);
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "设置更新失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleClearTrades() {
    setBusy(true);
    try {
      const state = await clearTrades();
      setAppState(state);
      setFlash(state.message);
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "清除失败");
    } finally {
      setBusy(false);
    }
  }

  if (appState === undefined) {
    return <main className="auth-screen"><section className="auth-panel"><h1>加载中...</h1></section></main>;
  }

  if (appState === null) {
    return <AuthScreen onAuthed={(state) => { setAppState(state); setBackend(pickDefaultBackend(state)); }} />;
  }

  return (
    <div className="dashboard-layout" style={{ "--font-scale": getFontScale(fontScaleKey) } as CSSProperties}>
      <CollapsibleSidebar expanded={sidebarExpanded} onToggle={() => setSidebarExpanded((prev) => !prev)}>
        <div className="sidebar-section-title"><FileSpreadsheet size={16} />策略匹配推荐系统</div>
        <div className="user-strip">
          <strong>{appState.user.username}</strong>
          <span>{profile ? `画像置信度：${CONFIDENCE_LABELS[profile.confidenceLevel] ?? profile.confidenceLevel}` : "尚未初始化画像"}</span>
        </div>
        <nav className="nav-stack" aria-label="主导航">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              type="button"
              className={activePage === item.key ? "nav-button active" : "nav-button"}
              onClick={() => {
                setActivePage(item.key);
                if (item.key === "stability") void loadStability();
              }}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <FontSizeControl value={fontScaleKey} onChange={setFontScale} />
        <button className="secondary-button full" type="button" onClick={handleLogout}><LogOut size={16} />退出登录</button>
      </CollapsibleSidebar>

      <div className="main-area">
        <div className="main-area-inner">
          <header className="topbar compact">
            <div>
              <p className="eyebrow">Next.js + FastAPI</p>
              <h1>客户交易画像策略推荐系统</h1>
            </div>
            <button className="icon-button" type="button" onClick={() => void refreshState()} aria-label="刷新数据">
              <RefreshCw size={18} />
            </button>
          </header>

          {message ? <div className="success-strip">{message}</div> : null}
          {error ? <div className="error-strip"><AlertCircle size={16} />{error}</div> : null}

          {activePage === "home" ? (
            <section className="workspace-grid">
              <div className="workspace-panel">
                <div className="panel-head"><Home size={18} /><h2>当前状态</h2></div>
                <div className="status-grid">
                  <div><span>问卷完成度</span><strong>{completedText}</strong></div>
                  <div><span>上传次数</span><strong>{appState.uploads.length}</strong></div>
                  <div><span>β 值</span><strong>{profile ? profile.beta.toFixed(2) : "--"}</strong></div>
                  <div><span>默认后端</span><strong>{profile?.matchingBackend ?? pickDefaultBackend(appState)}</strong></div>
                </div>
              </div>
              <div className="workspace-panel">
                <div className="panel-head"><SlidersHorizontal size={18} /><h2>快速操作</h2></div>
                <div className="action-row">
                  <button className="primary-button" type="button" onClick={() => setActivePage("questionnaire")}>完善问卷</button>
                  <button className="secondary-button" type="button" onClick={() => setActivePage("upload")}>上传交易</button>
                  <button className="secondary-button" type="button" onClick={() => void runRecommendation()} disabled={!profile || busy}>生成推荐</button>
                </div>
              </div>
            </section>
          ) : null}

          {activePage === "questionnaire" && selectedQuestionnaire ? (
            <section className="workspace-panel">
              <div className="panel-head"><ClipboardList size={18} /><h2>完善投资问卷</h2></div>
              <div className="segmented level-tabs">
                {questionnaires.map((item) => (
                  <button key={item.level} type="button" className={selectedLevel === item.level ? "active" : ""} onClick={() => { setSelectedLevel(item.level); setAnswers({}); }}>
                    {item.level}{item.completed ? " ✓" : ""}
                  </button>
                ))}
              </div>
              <div className="questionnaire-head">
                <h3>{selectedQuestionnaire.title}</h3>
                <p>{selectedQuestionnaire.description}（约 {selectedQuestionnaire.estimatedMinutes} 分钟）</p>
              </div>
              <div className="question-list">
                {selectedQuestionnaire.questions.map((question) => (
                  <div className="question-card" key={question.id}>
                    <h3>{question.id}. {question.text}</h3>
                    {question.type === "single_choice" ? (
                      <div className="option-list">
                        {question.options.map((option) => {
                          const value = option[0]?.toUpperCase() ?? option;
                          return (
                            <label key={option} className="option-row">
                              <input type="radio" name={question.id} checked={answers[question.id] === value} onChange={() => setAnswers((current) => ({ ...current, [question.id]: value }))} />
                              {option}
                            </label>
                          );
                        })}
                      </div>
                    ) : null}
                    {question.type === "multi_select" ? (
                      <div className="chip-group">
                        {question.options.map((option) => {
                          const value = option[0]?.toUpperCase() ?? option;
                          const selected = Array.isArray(answers[question.id]) && (answers[question.id] as string[]).includes(value);
                          return (
                            <button
                              key={option}
                              type="button"
                              className={selected ? "chip selected" : "chip"}
                              onClick={() => setAnswers((current) => {
                                const prev = Array.isArray(current[question.id]) ? current[question.id] as string[] : [];
                                return { ...current, [question.id]: selected ? prev.filter((item) => item !== value) : [...prev, value] };
                              })}
                            >
                              {option}
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
                    {question.type === "slider" ? (
                      <label className="range-row">
                        <input
                          type="range"
                          min={question.text.includes("亏损容忍度") ? 1 : 0}
                          max={question.text.includes("亏损容忍度") ? 20 : 100}
                          step={1}
                          value={Number(answers[question.id] ?? (question.text.includes("亏损容忍度") ? 10 : 50))}
                          onChange={(event) => setAnswers((current) => ({ ...current, [question.id]: Number(event.target.value) }))}
                        />
                        <strong>{Number(answers[question.id] ?? (question.text.includes("亏损容忍度") ? 10 : 50))}%</strong>
                      </label>
                    ) : null}
                    {question.type === "number_input" ? (
                      <input className="number-field" type="number" min={0} max={999} value={Number(answers[question.id] ?? 5)} onChange={(event) => setAnswers((current) => ({ ...current, [question.id]: Number(event.target.value) }))} />
                    ) : null}
                  </div>
                ))}
              </div>
              <button className="primary-button" type="button" onClick={() => void handleQuestionnaireSubmit()} disabled={busy}>提交问卷</button>
            </section>
          ) : null}

          {activePage === "upload" ? (
            <section className="workspace-panel">
              <div className="panel-head"><Upload size={18} /><h2>上传交易数据</h2></div>
              <label className="field-label">分析时间窗口
                <select value={uploadWindow} onChange={(event) => setUploadWindow(event.target.value)}>
                  <option value="all">全部数据</option>
                  <option value="120d">最近 120 天</option>
                  <option value="60d">最近 60 天</option>
                  <option value="30d">最近 30 天</option>
                </select>
              </label>
              <label className="upload-zone">
                <Upload size={22} />
                <span>选择 Excel / CSV 交易记录</span>
                <input type="file" accept=".xlsx,.xls,.csv" onChange={(event) => void handleUpload(event.target.files?.[0] ?? null)} />
              </label>
              <h3>上传历史</h3>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>文件</th><th>时间</th><th>交易笔数</th></tr></thead>
                  <tbody>
                    {appState.uploads.map((upload) => (
                      <tr key={upload.filename}><td>{upload.filename}</td><td>{upload.upload_date}</td><td>{upload.trade_count}</td></tr>
                    ))}
                    {appState.uploads.length === 0 ? <tr><td colSpan={3}>暂无上传记录</td></tr> : null}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          {activePage === "profile" ? (
            <section className="workspace-panel">
              <div className="panel-head"><UserRound size={18} /><h2>我的投资画像</h2></div>
              {profile ? (
                <>
                  <div className="status-grid">
                    <div><span>β 值</span><strong>{profile.beta.toFixed(2)}</strong></div>
                    <div><span>置信度</span><strong>{CONFIDENCE_LABELS[profile.confidenceLevel]}</strong></div>
                    <div><span>画像来源</span><strong>{profile.source}</strong></div>
                    <div><span>更新次数</span><strong>{profile.updateCount}</strong></div>
                  </div>
                  {appState.featureChart.length ? <RadarChart values={appState.featureChart.map((item) => ({ label: item.label, value: item.value }))} /> : null}
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>特征</th><th>画像值</th><th>标准化展示</th></tr></thead>
                      <tbody>
                        {appState.featureChart.map((item) => (
                          <tr key={item.key}><td>{item.label}</td><td>{item.rawValue.toFixed(4)}</td><td>{item.value.toFixed(1)}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : <p className="status-text">请先完成 Level 1 问卷以初始化画像。</p>}
            </section>
          ) : null}

          {activePage === "recommend" ? (
            <section className="recommend-page">
              <div className="workspace-panel">
                <div className="panel-head"><LineChartIcon size={18} /><h2>策略推荐</h2></div>
                <div className="recommend-controls">
                  <label className="field-label">匹配算法
                    <select value={backend} onChange={(event) => setBackend(event.target.value)}>
                      {appState.backends.map((item) => <option key={item.name} value={item.name}>{item.label}</option>)}
                    </select>
                  </label>
                  <label className="field-label">Top N
                    <input type="number" min={1} max={20} value={topN} onChange={(event) => setTopN(Math.max(1, Math.min(20, Number(event.target.value))))} />
                  </label>
                  <button className="primary-button" type="button" onClick={() => void runRecommendation()} disabled={!profile || busy}>生成推荐</button>
                </div>
                {!profile ? <p className="status-text">请先完成问卷以获取推荐。</p> : null}
              </div>

              {recommendation ? (
                <>
                  <RecommendationBoard
                    recommendations={sortedRecommendations}
                    isLoading={false}
                    isRecommending={busy}
                    selectedStrategyId={selectedStrategyId}
                    onSelectStrategy={setSelectedStrategyId}
                    customerProfile={recommendation.customer}
                    pcaVariance={recommendation.pca.explained_variance.map((v) => `${Math.round(v * 100)}%`).join(" / ")}
                  />
                  <section className="trend-section" ref={trendSectionRef}>
                    <div className="results-head">
                      <div>
                        <h2>收益曲线对比</h2>
                        <p>默认展示 Top1，可勾选 TopN 中任意策略。</p>
                      </div>
                    </div>
                    <div className="chip-group">
                      {sortedRecommendations.map((item) => {
                        const selected = trendStrategyIds.includes(item.strategy_id);
                        return (
                          <button
                            key={item.strategy_id}
                            className={selected ? "chip selected" : "chip"}
                            type="button"
                            onClick={() => setTrendStrategyIds((current) => selected ? current.filter((id) => id !== item.strategy_id) : [...current, item.strategy_id])}
                          >
                            {item.strategy_name}
                          </button>
                        );
                      })}
                    </div>
                    <TrendComparisonChart data={trendData} selectedStrategyIds={trendStrategyIds} />
                    <div className="table-wrap">
                      <table>
                        <thead><tr><th>对象</th><th>最终收益</th><th>覆盖率</th><th>质量</th><th>提示</th></tr></thead>
                        <tbody>
                          {trendData?.customerTrend ? (
                            <tr><td>我的账户</td><td>{formatPct(trendData.customerTrend.meta.finalReturn)}</td><td>{trendData.customerTrend.meta.coverageRate.toFixed(1)}%</td><td>{trendData.customerTrend.meta.dataQuality}</td><td>{trendData.customerTrend.meta.warnings.slice(0, 1).join("；")}</td></tr>
                          ) : <tr><td>我的账户</td><td colSpan={4}>尚未上传交易记录；上传后可查看个人账户对比。</td></tr>}
                          {trendStrategyIds.map((id) => {
                            const series = trendData?.strategyTrends[id];
                            return <tr key={id}><td>{id}</td><td>{formatPct(series?.meta.finalReturn)}</td><td>{(series?.meta.coverageRate ?? 0).toFixed(1)}%</td><td>{series?.meta.dataQuality ?? "--"}</td><td>{series?.meta.warnings.slice(0, 1).join("；")}</td></tr>;
                          })}
                        </tbody>
                      </table>
                    </div>
                    <div className="workspace-panel subtle-panel">
                      <h3>弹窗话术</h3>
                      <p>{recommendation.popupText}</p>
                    </div>
                  </section>
                </>
              ) : null}
            </section>
          ) : null}

          {activePage === "stability" ? (
            <section className="workspace-panel">
              <div className="panel-head"><BarChart3 size={18} /><h2>匹配稳定性分析</h2></div>
              <button className="primary-button" type="button" onClick={() => void loadStability()} disabled={busy}>重新计算</button>
              {stability && !stability.ready ? <p className="status-text">{stability.message}</p> : null}
              {stability?.ready ? (
                <>
                  <h3>多窗口推荐对比</h3>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>窗口</th><th>Top1</th><th>匹配度</th><th>交易笔数</th></tr></thead>
                      <tbody>{stability.windows.map((row) => <tr key={row.window}><td>{row.window}</td><td>{row.top1}</td><td>{formatPct(row.similarity)}</td><td>{row.count}</td></tr>)}</tbody>
                    </table>
                  </div>
                  <p className="status-text">{stability.conclusion}</p>
                  <h3>三路线推荐对比</h3>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>算法</th><th>排名</th><th>策略</th><th>匹配度</th></tr></thead>
                      <tbody>{stability.backendComparison.map((row, index) => <tr key={`${row.backend}-${index}`}><td>{row.backendLabel}</td><td>{row.rank ? `#${row.rank}` : "--"}</td><td>{row.strategy}</td><td>{formatPct(row.similarity)}</td></tr>)}</tbody>
                    </table>
                  </div>
                </>
              ) : null}
            </section>
          ) : null}

          {activePage === "settings" ? (
            <section className="workspace-panel">
              <div className="panel-head"><Settings size={18} /><h2>设置</h2></div>
              {profile ? (
                <>
                  <label className="field-label">默认匹配后端
                    <select value={profile.matchingBackend} onChange={(event) => void applySettings({ backend: event.target.value })}>
                      {appState.backends.map((item) => <option key={item.name} value={item.name}>{item.label}</option>)}
                    </select>
                  </label>
                  <label className="range-row">β 超参数
                    <input type="range" min={0} max={1} step={0.05} value={profile.beta} onChange={(event) => void applySettings({ beta: Number(event.target.value) })} />
                    <strong>{profile.beta.toFixed(2)}</strong>
                  </label>
                  <label className="range-row">融合权重 α
                    <input type="range" min={0.5} max={0.9} step={0.1} value={appState.fusionAlpha} onChange={(event) => void applySettings({ fusionAlpha: Number(event.target.value) })} />
                    <strong>{appState.fusionAlpha.toFixed(1)}</strong>
                  </label>
                  <button className="danger-button" type="button" onClick={() => void handleClearTrades()}><Trash2 size={16} />清除交易数据</button>
                </>
              ) : <p className="status-text">请先完成问卷。</p>}
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
