"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Clock,
  FileSpreadsheet,
  LineChart as LineChartIcon,
  LogOut,
  Menu,
  RefreshCw,
  Search,
  Settings,
  SlidersHorizontal,
  Trash2,
  Upload,
  UserPlus,
  UserRound,
  Users,
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
  createCustomer,
  fetchCustomerState,
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
  type Customer,
  type Question,
  type Questionnaire,
  type RecommendResponse,
  type StabilityResponse,
  type TrendComparisonResponse,
} from "@/lib/api";
import { CollapsibleSidebar } from "@/components/CollapsibleSidebar";
import { FontSizeControl, getFontScale, normalizeFontScaleKey, type FontScaleKey } from "@/components/FontSizeControl";
import { RadarChart } from "@/components/RadarChart";
import { RecommendationBoard } from "@/components/RecommendationBoard";

type PageKey = "customers" | "detail" | "recommend" | "stability" | "settings";
type FocusTargetKey = PageKey | "trend";

const NAV_ITEMS: Array<{ key: PageKey; label: string; icon: React.ReactNode }> = [
  { key: "customers", label: "客户池", icon: <Users size={16} /> },
  { key: "detail", label: "客户详情", icon: <UserRound size={16} /> },
  { key: "recommend", label: "推荐方案", icon: <LineChartIcon size={16} /> },
  { key: "stability", label: "策略对比", icon: <BarChart3 size={16} /> },
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

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
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

function groupCustomersByStatus(customers: Customer[]): Array<{ key: string; label: string; customers: Customer[] }> {
  const order = [
    ["needs_questionnaire", "待补资料"],
    ["needs_trades", "待上传交易"],
    ["needs_profile", "待生成画像"],
    ["ready_to_recommend", "可生成推荐"],
  ];
  return order.map(([key, label]) => ({
    key,
    label,
    customers: customers.filter((customer) => customer.status === key),
  }));
}

const WORKFLOW_STEPS: Array<{
  key: Customer["workflowStep"];
  label: string;
  helper: string;
  icon: React.ReactNode;
}> = [
  { key: "questionnaire", label: "资料", helper: "完成三层问卷", icon: <ClipboardList size={16} /> },
  { key: "trades", label: "交易", helper: "上传流水", icon: <Upload size={16} /> },
  { key: "profile", label: "画像", helper: "校验特征", icon: <UserRound size={16} /> },
  { key: "recommendation", label: "推荐", helper: "沟通方案", icon: <LineChartIcon size={16} /> },
];

function getWorkflowStepIndex(customer: Customer | null): number {
  if (!customer) return 0;
  return Math.max(0, WORKFLOW_STEPS.findIndex((item) => item.key === customer.workflowStep));
}

function WorkflowProgress({ customer }: { customer: Customer | null }) {
  const activeIndex = getWorkflowStepIndex(customer);
  const progress = Math.max(0, Math.min(100, customer?.workflowProgress ?? 0));

  return (
    <div className="workflow-panel" aria-label="客户流程进度">
      <div className="workflow-panel-head">
        <div>
          <span>流程进度</span>
          <strong>{progress}%</strong>
        </div>
        <span className="status-pill">{customer?.statusLabel ?? "--"}</span>
      </div>
      <div className="workflow-track" aria-hidden="true">
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="workflow-steps">
        {WORKFLOW_STEPS.map((step, index) => {
          const state = index < activeIndex || progress === 100
            ? "complete"
            : index === activeIndex
              ? "active"
              : "pending";
          return (
            <div className={`workflow-step ${state}`} key={step.key}>
              <span className="workflow-step-icon">
                {state === "complete" ? <CheckCircle2 size={16} /> : step.icon}
              </span>
              <span>
                <strong>{step.label}</strong>
                <small>{step.helper}</small>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CustomerCard({
  customer,
  active,
  onOpen,
}: {
  customer: Customer;
  active: boolean;
  onOpen: (customer: Customer) => void;
}) {
  return (
    <button
      className={active ? "customer-card active" : "customer-card"}
      type="button"
      onClick={() => onOpen(customer)}
    >
      <span className="customer-card-main">
        <strong>{customer.name}</strong>
        <span className="status-pill">{customer.statusLabel}</span>
      </span>
      <span className="customer-next-action">{customer.nextAction}</span>
      <span className="customer-progress-row">
        <span className="customer-progress-track"><span style={{ width: `${customer.workflowProgress}%` }} /></span>
        <b>{customer.workflowProgress}%</b>
      </span>
      <span className="customer-meta-grid">
        <span><b>{customer.completedLevels.length}/3</b>问卷</span>
        <span><b>{customer.tradeCount}</b>交易</span>
        <span><b>{customer.confidenceLevel ? CONFIDENCE_LABELS[customer.confidenceLevel] : "--"}</b>置信度</span>
      </span>
      <span className="customer-card-action">
        {customer.primaryActionLabel}
        <ArrowRight size={14} />
      </span>
      <span className="customer-card-foot">更新：{formatDateTime(customer.lastUpdated)}</span>
    </button>
  );
}

function RecommendationReadiness({
  customer,
  hasProfile,
  uploadsCount,
  completedText,
}: {
  customer: Customer | null;
  hasProfile: boolean;
  uploadsCount: number;
  completedText: string;
}) {
  const rows = [
    { label: "问卷", value: completedText, ready: (customer?.completedLevels.length ?? 0) >= 3 },
    { label: "交易", value: `${uploadsCount} 次上传`, ready: uploadsCount > 0 },
    { label: "画像", value: hasProfile ? "已生成" : "未生成", ready: hasProfile },
  ];

  return (
    <div className="readiness-grid" aria-label="推荐准备状态">
      {rows.map((row) => (
        <div className={row.ready ? "ready" : "blocked"} key={row.label}>
          {row.ready ? <CheckCircle2 size={16} /> : <Clock size={16} />}
          <span>{row.label}</span>
          <strong>{row.value}</strong>
        </div>
      ))}
    </div>
  );
}

function AuthScreen({
  initialMessage,
  onAuthed,
}: {
  initialMessage?: string;
  onAuthed: (state: AppState) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState(initialMessage ?? "");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setMessage(initialMessage ?? "");
  }, [initialMessage]);

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
            <Line type="monotone" dataKey="customer" name="客户账户" stroke="#2563eb" strokeWidth={3} dot={false} connectNulls />
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
  const [activePage, setActivePage] = useState<PageKey>("customers");
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [selectedLevel, setSelectedLevel] = useState<"L1" | "L2" | "L3">("L1");
  const [answers, setAnswers] = useState<Record<string, string | number | string[]>>({});
  const [customerSearch, setCustomerSearch] = useState("");
  const [newCustomerName, setNewCustomerName] = useState("");
  const [newCustomerNote, setNewCustomerNote] = useState("");
  const [uploadWindow, setUploadWindow] = useState("all");
  const [backend, setBackend] = useState("statistical");
  const [topN, setTopN] = useState(5);
  const [recommendation, setRecommendation] = useState<RecommendResponse | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null);
  const [trendStrategyIds, setTrendStrategyIds] = useState<string[]>([]);
  const [trendData, setTrendData] = useState<TrendComparisonResponse | null>(null);
  const [stability, setStability] = useState<StabilityResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [authNotice, setAuthNotice] = useState("");
  const mainContentRef = useRef<HTMLDivElement | null>(null);
  const contentFocusRefs = useRef<Partial<Record<FocusTargetKey, HTMLElement | null>>>({});
  const [focusRequest, setFocusRequest] = useState<{ target: FocusTargetKey; nonce: number } | null>(null);
  const [fontScaleKey, setFontScaleKey] = useState<FontScaleKey>(() => {
    if (typeof window === "undefined") return "md";
    return normalizeFontScaleKey(window.localStorage.getItem("investment-font-scale"));
  });

  function setContentFocusRef(target: FocusTargetKey) {
    return (node: HTMLElement | null) => {
      contentFocusRefs.current[target] = node;
    };
  }

  function requestContentFocus(target: FocusTargetKey) {
    setFocusRequest({ target, nonce: Date.now() });
  }

  function navigateToPage(nextPage: PageKey) {
    setActivePage(nextPage);
    requestContentFocus(nextPage);
  }

  const currentCustomer = appState?.currentCustomer ?? null;
  const customers = appState?.customers ?? [];
  const currentCustomerId = currentCustomer?.customerId;

  useEffect(() => {
    async function load() {
      try {
        const state = await fetchSession({ suppressAuthEvent: true });
        setAppState(state);
        setBackend(pickDefaultBackend(state));
      } catch {
        setAppState(null);
      }
    }
    void load();
  }, []);

  useEffect(() => {
    window.localStorage.setItem("investment-font-scale", fontScaleKey);
  }, [fontScaleKey]);

  useEffect(() => {
    if (window.innerWidth <= 768) setSidebarExpanded(false);
  }, []);

  useEffect(() => {
    function handleUnauthorized(event: Event) {
      const detail = event instanceof CustomEvent && typeof event.detail === "string"
        ? event.detail
        : "登录状态已失效";
      setAppState(null);
      setRecommendation(null);
      setTrendData(null);
      setTrendStrategyIds([]);
      setSelectedStrategyId(null);
      setStability(null);
      setAnswers({});
      setActivePage("customers");
      setAuthNotice(detail === "Not authenticated" ? "请先登录后继续操作。" : detail);
      setFlash("");
    }

    window.addEventListener("investment:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("investment:unauthorized", handleUnauthorized);
  }, []);

  useEffect(() => {
    if (!appState || !currentCustomerId) return;
    void fetchQuestionnaires(currentCustomerId).then((data) => {
      setQuestionnaires(data.questionnaires);
      const next = findNextQuestionnaire(data.questionnaires, appState.completedLevels);
      setSelectedLevel(next?.level ?? "L3");
    });
  }, [appState, currentCustomerId]);

  useEffect(() => {
    if (!appState) return;
    setBackend(pickDefaultBackend(appState));
  }, [appState?.profile?.matchingBackend, appState?.backends.length]);

  useEffect(() => {
    if (trendStrategyIds.length === 0 && !recommendation) return;
    let cancelled = false;
    async function loadTrend() {
      try {
        const data = await fetchTrends(trendStrategyIds, currentCustomerId);
        if (!cancelled) setTrendData(data);
      } catch {
        if (!cancelled) setTrendData(null);
      }
    }
    void loadTrend();
    return () => {
      cancelled = true;
    };
  }, [trendStrategyIds, recommendation, currentCustomerId]);

  useEffect(() => {
    if (activePage !== "recommend" || !recommendation) return;
    requestContentFocus("trend");
  }, [activePage, recommendation, trendStrategyIds.join("|")]);

  useEffect(() => {
    if (!focusRequest) return;
    const timer = window.setTimeout(() => {
      const target = contentFocusRefs.current[focusRequest.target]
        ?? contentFocusRefs.current[activePage]
        ?? mainContentRef.current;
      if (!target) return;
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      target.focus({ preventScroll: true });
      target.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "start",
        inline: "nearest",
      });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [focusRequest, activePage, selectedLevel, recommendation]);

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
      const state = currentCustomerId ? await fetchCustomerState(currentCustomerId) : await fetchSession();
      setAppState(state);
      setFlash("数据已刷新。");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "刷新失败");
    }
  }

  async function refreshRecommendationAndTrends(nextBackend = backend, nextTopN = topN, nextCustomerId = currentCustomerId) {
    const data = await fetchRecommendations({ backend: nextBackend, topN: nextTopN, customerId: nextCustomerId });
    const first = data.recommendations[0]?.strategy_id ?? null;
    const nextTrendIds = first ? [first] : [];
    setRecommendation(data);
    setSelectedStrategyId(first);
    setTrendStrategyIds(nextTrendIds);
    setTrendData(await fetchTrends(nextTrendIds, nextCustomerId));
    return data;
  }

  async function handleLogout() {
    await logout();
    setAppState(null);
    setRecommendation(null);
    setTrendData(null);
  }

  async function selectCustomer(customerId: string, nextPage: PageKey = "detail") {
    setBusy(true);
    setFlash("");
    try {
      const state = await fetchCustomerState(customerId);
      setAppState(state);
      setRecommendation(null);
      setTrendData(null);
      setTrendStrategyIds([]);
      setSelectedStrategyId(null);
      setAnswers({});
      setActivePage(nextPage);
      requestContentFocus(nextPage);
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "客户切换失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateCustomer() {
    const name = newCustomerName.trim();
    if (!name) {
      setFlash("", "请输入客户姓名。");
      return;
    }
    setBusy(true);
    setFlash("");
    try {
      const state = await createCustomer({ name, note: newCustomerNote });
      setAppState(state);
      setNewCustomerName("");
      setNewCustomerNote("");
      setRecommendation(null);
      setTrendData(null);
      setTrendStrategyIds([]);
      setSelectedStrategyId(null);
      setAnswers({});
      setActivePage("detail");
      requestContentFocus("detail");
      setFlash("客户已创建，请继续补齐资料。");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "客户创建失败");
    } finally {
      setBusy(false);
    }
  }

  const selectedQuestionnaire = questionnaires.find((item) => item.level === selectedLevel) ?? questionnaires[0];
  const sortedRecommendations = recommendation?.recommendations ?? [];
  const profile = appState?.profile ?? null;
  const completedText = appState?.completedLevels.length
    ? appState.completedLevels.join(" / ")
    : "未完成";
  const filteredCustomers = customers.filter((customer) => {
    const query = customerSearch.trim().toLowerCase();
    if (!query) return true;
    return `${customer.name} ${customer.note} ${customer.statusLabel} ${customer.primaryActionLabel} ${customer.blockedReason ?? ""}`
      .toLowerCase()
      .includes(query);
  });
  const customerGroups = groupCustomersByStatus(filteredCustomers);
  const averageWorkflowProgress = customers.length
    ? Math.round(customers.reduce((sum, customer) => sum + customer.workflowProgress, 0) / customers.length)
    : 0;
  const canRunRecommendation = Boolean(profile && currentCustomer?.workflowStep === "recommendation");

  function openCustomerFromPool(customer: Customer) {
    void selectCustomer(customer.customerId, customer.primaryActionPage);
  }

  function handlePrimaryAction() {
    if (!currentCustomer) return;
    if (currentCustomer.workflowStep === "recommendation" && profile) {
      void runRecommendation();
      return;
    }
    navigateToPage(currentCustomer.primaryActionPage);
  }

  async function handleQuestionnaireSubmit() {
    if (!selectedQuestionnaire) return;
    setBusy(true);
    setFlash("");
    try {
      const payload = buildQuestionnairePayload(selectedQuestionnaire, answers);
      const state = await submitQuestionnaire(selectedQuestionnaire.level, payload, currentCustomerId);
      setAppState(state);
      const freshCustomerId = state.currentCustomer?.customerId;
      const fresh = freshCustomerId
        ? await fetchQuestionnaires(freshCustomerId)
        : { questionnaires };
      setQuestionnaires(fresh.questionnaires);
      setAnswers({});
      const next = findNextQuestionnaire(fresh.questionnaires, state.completedLevels);
      if (next) {
        setSelectedLevel(next.level);
        setActivePage("detail");
        requestContentFocus("detail");
        setFlash(`${state.message} 请继续完成 ${next.level}。`);
      } else {
        setActivePage("detail");
        requestContentFocus("detail");
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
      const state = await uploadTrades(file, uploadWindow, currentCustomerId);
      setAppState(state);
      if (state.profile && state.currentCustomer?.customerId) {
        try {
          await refreshRecommendationAndTrends(pickDefaultBackend(state), topN, state.currentCustomer.customerId);
          setBackend(pickDefaultBackend(state));
          setActivePage("recommend");
          requestContentFocus("trend");
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
      await refreshRecommendationAndTrends(backend, topN, currentCustomerId);
      setActivePage("recommend");
      requestContentFocus("trend");
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "推荐计算失败");
    } finally {
      setBusy(false);
    }
  }

  async function loadStability() {
    setBusy(true);
    setFlash("");
    setActivePage("stability");
    requestContentFocus("stability");
    try {
      const data = await fetchStability(currentCustomerId);
      setStability(data);
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
      const state = await updateSettings({ ...next, customerId: currentCustomerId });
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
      const state = await clearTrades(currentCustomerId);
      setAppState(state);
      setFlash(state.message);
    } catch (err) {
      setFlash("", err instanceof Error ? err.message : "清除失败");
    } finally {
      setBusy(false);
    }
  }

  function renderQuestionnairePanel() {
    return (
      <section className="workspace-panel task-panel">
        <div className="panel-head"><ClipboardList size={18} /><h2>客户问卷</h2></div>
        {selectedQuestionnaire ? (
          <>
            <div className="segmented level-tabs">
              {questionnaires.map((item) => (
                <button
                  key={item.level}
                  type="button"
                  className={selectedLevel === item.level ? "active" : ""}
                  onClick={() => {
                    setSelectedLevel(item.level);
                    setAnswers({});
                    requestContentFocus("detail");
                  }}
                >
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
            <button className="primary-button" type="button" onClick={() => void handleQuestionnaireSubmit()} disabled={busy}>提交客户问卷</button>
          </>
        ) : (
          <p className="status-text">问卷配置暂不可用。</p>
        )}
      </section>
    );
  }

  function renderUploadPanel() {
    if (!appState) return null;
    return (
      <section className="workspace-panel task-panel">
        <div className="panel-head"><Upload size={18} /><h2>客户交易数据</h2></div>
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
    );
  }

  function renderProfilePanel() {
    if (!appState) return null;
    return (
      <section className="workspace-panel task-panel">
        <div className="panel-head"><UserRound size={18} /><h2>客户画像</h2></div>
        {profile ? (
          <>
            <div className="status-grid compact-status-grid">
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
        ) : <p className="status-text">请先完成客户 L1 问卷以初始化画像。</p>}
      </section>
    );
  }

  if (appState === undefined) {
    return <main className="auth-screen"><section className="auth-panel"><h1>加载中...</h1></section></main>;
  }

  if (appState === null) {
    return <AuthScreen initialMessage={authNotice} onAuthed={(state) => { setAuthNotice(""); setAppState(state); setBackend(pickDefaultBackend(state)); }} />;
  }

  return (
    <div className="dashboard-layout" style={{ "--font-scale": getFontScale(fontScaleKey) } as CSSProperties}>
      <CollapsibleSidebar expanded={sidebarExpanded} onToggle={() => setSidebarExpanded((prev) => !prev)}>
        <div className="sidebar-section-title"><FileSpreadsheet size={16} />销售客户工作台</div>
        <div className="user-strip">
          <strong>{appState.user.username}</strong>
          <span>当前客户：{currentCustomer?.name ?? "未选择客户"}</span>
          <span>{currentCustomer ? `流程进度：${currentCustomer.workflowProgress}%` : "尚未选择客户"}</span>
          <span>{profile ? `画像置信度：${CONFIDENCE_LABELS[profile.confidenceLevel] ?? profile.confidenceLevel}` : "客户画像未初始化"}</span>
        </div>
        <nav className="nav-stack" aria-label="主导航">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              type="button"
              className={activePage === item.key ? "nav-button active" : "nav-button"}
              onClick={() => {
                if (item.key === "stability") void loadStability();
                else navigateToPage(item.key);
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
        <div className="main-area-inner" ref={mainContentRef}>
          <header className="topbar compact">
            <div>
              <p className="eyebrow">CRM Workflow</p>
              <h1>营业部销售客户工作台</h1>
            </div>
            <div className="topbar-actions">
              <button className="sidebar-mobile-trigger" type="button" onClick={() => setSidebarExpanded(true)}>
                <Menu size={16} />
                导航
              </button>
              <button className="icon-button" type="button" onClick={() => void refreshState()} aria-label="刷新数据">
                <RefreshCw size={18} />
              </button>
            </div>
          </header>

          {message ? <div className="success-strip">{message}</div> : null}
          {error ? <div className="error-strip"><AlertCircle size={16} />{error}</div> : null}

          {activePage === "customers" ? (
            <section className="customer-pool-page">
              <div className="workspace-panel customer-pool-toolbar crm-toolbar">
                <div>
                  <div className="panel-head"><Users size={18} /><h2 className="content-focus-target" tabIndex={-1} ref={setContentFocusRef("customers")}>客户池</h2></div>
                  <p className="status-text">按客户推进状态组织待办，优先处理阻塞推荐生成的客户。</p>
                </div>
                <div className="customer-toolbar-actions">
                  <label className="field-label customer-search">
                    <span>搜索客户</span>
                    <span className="input-with-icon"><Search size={16} /><input value={customerSearch} onChange={(event) => setCustomerSearch(event.target.value)} placeholder="姓名、备注、状态或下一步" /></span>
                  </label>
                  <div className="customer-create-box">
                    <label className="field-label">
                      新建客户
                      <input value={newCustomerName} onChange={(event) => setNewCustomerName(event.target.value)} placeholder="客户姓名" />
                    </label>
                    <label className="field-label">
                      备注
                      <input value={newCustomerNote} onChange={(event) => setNewCustomerNote(event.target.value)} placeholder="可选，例如风险偏好或跟进事项" />
                    </label>
                    <button className="primary-button" type="button" onClick={() => void handleCreateCustomer()} disabled={busy}><UserPlus size={16} />创建客户</button>
                  </div>
                </div>
              </div>

              <div className="status-grid customer-pool-summary">
                <div><span>客户总数</span><strong>{customers.length}</strong></div>
                <div><span>平均进度</span><strong>{averageWorkflowProgress}%</strong></div>
                <div><span>待补资料</span><strong>{customers.filter((customer) => customer.status === "needs_questionnaire").length}</strong></div>
                <div><span>可生成推荐</span><strong>{customers.filter((customer) => customer.status === "ready_to_recommend").length}</strong></div>
              </div>

              <div className="customer-group-stack">
                {customerGroups.map((group) => (
                  <section className={`workspace-panel customer-group ${group.key}`} key={group.key}>
                    <div className="customer-group-head">
                      <div>
                        <h3>{group.label}</h3>
                        <p>{group.customers.length} 位客户需要处理</p>
                      </div>
                      {group.key === "ready_to_recommend" ? <CheckCircle2 size={18} /> : <Clock size={18} />}
                    </div>
                    {group.customers.length ? (
                      <div className="customer-card-grid">
                        {group.customers.map((customer) => (
                          <CustomerCard
                            key={customer.customerId}
                            customer={customer}
                            active={customer.customerId === currentCustomerId}
                            onOpen={openCustomerFromPool}
                          />
                        ))}
                      </div>
                    ) : (
                      <p className="status-text">暂无客户处于该状态。</p>
                    )}
                  </section>
                ))}
                {filteredCustomers.length === 0 ? (
                  <section className="workspace-panel empty-customer-state">
                    <Search size={24} />
                    <p>没有匹配的客户。</p>
                  </section>
                ) : null}
              </div>
            </section>
          ) : null}

          {activePage === "detail" ? (
            <section className="customer-detail-page">
              <div className="workspace-panel customer-hero">
                <div className="customer-hero-top">
                  <div>
                    <p className="eyebrow">当前客户</p>
                    <h2 className="content-focus-target customer-name-title" tabIndex={-1} ref={setContentFocusRef("detail")}>{currentCustomer?.name ?? "未选择客户"}</h2>
                    <p className="status-text">{currentCustomer?.note || "暂无客户备注。"}</p>
                  </div>
                  <span className="status-pill hero-status">{currentCustomer?.statusLabel ?? "--"}</span>
                </div>
                <WorkflowProgress customer={currentCustomer} />
                {currentCustomer?.blockedReason ? (
                  <div className="workflow-alert">
                    <AlertCircle size={16} />
                    <span>{currentCustomer.blockedReason}</span>
                  </div>
                ) : null}
                <div className="status-grid customer-detail-summary">
                  <div><span>下一步</span><strong>{currentCustomer?.primaryActionLabel ?? "--"}</strong></div>
                  <div><span>问卷完成度</span><strong>{completedText}</strong></div>
                  <div><span>上传次数</span><strong>{appState.uploads.length}</strong></div>
                  <div><span>画像置信度</span><strong>{profile ? CONFIDENCE_LABELS[profile.confidenceLevel] : "--"}</strong></div>
                </div>
                <div className="action-row customer-hero-actions">
                  <button className="secondary-button" type="button" onClick={() => navigateToPage("customers")}><Users size={16} />返回客户池</button>
                  <button className="primary-button" type="button" onClick={handlePrimaryAction} disabled={!currentCustomer || busy || (currentCustomer.workflowStep === "recommendation" && !profile)}>
                    <ArrowRight size={16} />
                    {currentCustomer?.primaryActionLabel ?? "继续处理"}
                  </button>
                </div>
              </div>
              <div className="task-flow-label">
                <span>客户推进任务</span>
                <p>按资料、交易、画像、推荐顺序处理；已满足条件后可直接生成推荐。</p>
              </div>
              <div className="task-grid">
                {renderQuestionnairePanel()}
                {renderUploadPanel()}
                {renderProfilePanel()}
              </div>
            </section>
          ) : null}

          {activePage === "recommend" ? (
            <section className="recommend-page">
              <div className="workspace-panel recommend-command-panel">
                <div className="panel-head"><LineChartIcon size={18} /><h2 className="content-focus-target" tabIndex={-1} ref={setContentFocusRef("recommend")}>推荐方案</h2></div>
                <div className="recommend-context">
                  <div>
                    <p className="eyebrow">当前客户</p>
                    <h3>{currentCustomer?.name ?? "--"}</h3>
                    <p className="status-text">{currentCustomer?.nextAction ?? "选择客户后生成推荐方案。"}</p>
                  </div>
                  <RecommendationReadiness
                    customer={currentCustomer}
                    hasProfile={Boolean(profile)}
                    uploadsCount={appState.uploads.length}
                    completedText={completedText}
                  />
                </div>
                {currentCustomer?.blockedReason ? (
                  <div className="workflow-alert compact">
                    <AlertCircle size={16} />
                    <span>{currentCustomer.blockedReason}</span>
                  </div>
                ) : null}
                <div className="recommend-controls">
                  <label className="field-label">匹配算法
                    <select value={backend} onChange={(event) => setBackend(event.target.value)}>
                      {appState.backends.map((item) => <option key={item.name} value={item.name}>{item.label}</option>)}
                    </select>
                  </label>
                  <label className="field-label">Top N
                    <input type="number" min={1} max={20} value={topN} onChange={(event) => setTopN(Math.max(1, Math.min(20, Number(event.target.value))))} />
                  </label>
                  <button className="primary-button" type="button" onClick={() => void runRecommendation()} disabled={!canRunRecommendation || busy}>生成推荐方案</button>
                </div>
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
                  <section className="trend-section">
                    <div className="results-head">
                      <div>
                        <h2 className="content-focus-target" tabIndex={-1} ref={setContentFocusRef("trend")}>收益曲线对比</h2>
                        <p>默认展示 Top1，可勾选 TopN 中任意策略与客户账户对比。</p>
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
                            <tr><td>客户账户</td><td>{formatPct(trendData.customerTrend.meta.finalReturn)}</td><td>{trendData.customerTrend.meta.coverageRate.toFixed(1)}%</td><td>{trendData.customerTrend.meta.dataQuality}</td><td>{trendData.customerTrend.meta.warnings.slice(0, 1).join("；")}</td></tr>
                          ) : <tr><td>客户账户</td><td colSpan={4}>尚未上传交易记录；上传后可查看客户账户对比。</td></tr>}
                          {trendStrategyIds.map((id) => {
                            const series = trendData?.strategyTrends[id];
                            return <tr key={id}><td>{id}</td><td>{formatPct(series?.meta.finalReturn)}</td><td>{(series?.meta.coverageRate ?? 0).toFixed(1)}%</td><td>{series?.meta.dataQuality ?? "--"}</td><td>{series?.meta.warnings.slice(0, 1).join("；")}</td></tr>;
                          })}
                        </tbody>
                      </table>
                    </div>
                    <div className="workspace-panel subtle-panel">
                      <h3>推荐话术</h3>
                      <p>{recommendation.popupText}</p>
                    </div>
                  </section>
                </>
              ) : (
                <section className="workspace-panel empty-recommend-state">
                  <LineChartIcon size={28} />
                  <div>
                    <h3>{canRunRecommendation ? "尚未生成推荐方案" : "推荐条件尚未满足"}</h3>
                    <p>{canRunRecommendation ? "选择算法和 Top N 后生成推荐，系统会同步加载收益曲线。" : currentCustomer?.blockedReason ?? "请先选择客户并完成前置任务。"}</p>
                  </div>
                </section>
              )}
            </section>
          ) : null}

          {activePage === "stability" ? (
            <section className="workspace-panel">
              <div className="panel-head"><BarChart3 size={18} /><h2 className="content-focus-target" tabIndex={-1} ref={setContentFocusRef("stability")}>策略对比</h2></div>
              <p className="status-text">当前客户：{currentCustomer?.name ?? "--"}</p>
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
              <div className="panel-head"><SlidersHorizontal size={18} /><h2 className="content-focus-target" tabIndex={-1} ref={setContentFocusRef("settings")}>客户设置</h2></div>
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
