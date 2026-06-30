export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

function apiUrl(path: string): string {
  if (!API_BASE) return path;
  const backendPath = path.replace(/^\/api/, "");
  return `${API_BASE}${backendPath}`;
}

type RequestOptions = RequestInit & { json?: unknown };

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  let body = options.body;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }

  const response = await fetch(apiUrl(path), {
    ...options,
    headers,
    body,
    credentials: "include",
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      message = typeof data.detail === "string" ? data.detail : message;
    } catch {
      const text = await response.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export type User = {
  userId: string;
  username: string;
  createdAt: string;
  lastLogin: string;
  onboardingStatus: string;
};

export type Profile = {
  userId: string;
  beta: number;
  riskTolerance: number;
  initialCapital: number;
  features: Record<string, number>;
  industryVector: Record<string, number>;
  updateCount: number;
  confidenceLevel: "low" | "medium" | "high";
  source: string;
  lastUpdated: string;
  matchingBackend: string;
  history: Array<{ update: number; features: Record<string, number>; timestamp: string }>;
};

export type UploadRecord = {
  filename: string;
  upload_date: string;
  trade_count: number;
};

export type BackendOption = {
  name: string;
  label: string;
};

export type FeatureChartItem = {
  key: string;
  label: string;
  value: number;
  rawValue: number;
  strategyAverage: number;
};

export type AppState = {
  user: User;
  profile: Profile | null;
  completedLevels: string[];
  uploads: UploadRecord[];
  backends: BackendOption[];
  lstmAvailable: boolean;
  lstmAssignedAccounts: Record<string, string>;
  fusionAlpha: number;
  featureChart: FeatureChartItem[];
};

export type Question = {
  id: string;
  text: string;
  type: "single_choice" | "slider" | "number_input" | "multi_select";
  options: string[];
};

export type Questionnaire = {
  level: "L1" | "L2" | "L3";
  title: string;
  description: string;
  estimatedMinutes: number;
  completed: boolean;
  questions: Question[];
};

export type DimensionRanks = Record<"style" | "performance" | "risk" | "preference" | "factor" | "cluster" | "cca", number>;

export type Recommendation = {
  strategy_id: string;
  strategy_name: string;
  score: number;
  rank_sum: number;
  rank_score: number;
  final_rank_score: number;
  dimension_ranks: DimensionRanks;
  style_similarity: number;
  mahalanobis_style_score: number;
  factor_score: number;
  cluster_score: number;
  cca_score: number;
  cluster_label: string;
  performance_score: number;
  risk_score: number;
  preference_score: number;
  symbol_overlap: number;
  theme_overlap: number;
  themes: string[];
  top_symbols: string[];
  factor_label: string;
  return_2025: number;
  annual_return: number;
  max_drawdown_2025: number;
  explanation: string;
};

export type RecommendResponse = {
  customer: {
    id: string;
    username: string;
    trade_count: number;
    buy_ratio: number;
    active_days: number;
    turnover_proxy: number;
    concentration: number;
    themes: string[];
    top_symbols: string[];
  };
  backend: string;
  metricUsed: string;
  popupText: string;
  explanation: Record<string, unknown>;
  recommendations: Recommendation[];
  pca: { explained_variance: number[] };
};

export type TrendPoint = {
  date: string;
  totalAsset: number;
  cumulativeReturn: number;
};

export type TrendMeta = {
  initialCapital?: number;
  initialAsset?: number;
  startDate: string | null;
  endDate: string | null;
  finalReturn: number | null;
  dataQuality: string;
  missingSymbols: string[];
  fallbackSymbols: string[];
  coverageRate: number;
  warnings: string[];
};

export type TrendSeries = {
  label: string;
  trend: TrendPoint[];
  meta: TrendMeta;
};

export type TrendComparisonResponse = {
  customerTrend: TrendSeries | null;
  strategyTrends: Record<string, TrendSeries>;
};

export type CustomerTrendPoint = TrendPoint;
export type CustomerTrendMeta = TrendMeta;
export type MergedTrendPoint = {
  date: string;
  customerReturn: number | null;
  strategyReturn: number | null;
};
export type AlignedTrendPoint = MergedTrendPoint;
export type MergedTrendResponse = {
  customerTrend: CustomerTrendPoint[];
  customerMeta: CustomerTrendMeta | null;
  strategyTrend: CustomerTrendPoint[] | null;
  strategyMeta: CustomerTrendMeta | null;
  mergedTrend: MergedTrendPoint[] | null;
};

export type StabilityResponse = {
  ready: boolean;
  message: string;
  windows: Array<{ window: string; top1: string; similarity: number | null; count: number }>;
  backendComparison: Array<{
    backend: string;
    backendLabel: string;
    rank: number | null;
    strategy: string;
    similarity: number | null;
  }>;
  conclusion: string;
};

export async function fetchSession(): Promise<AppState> {
  return request<AppState>("/api/auth/me");
}

export async function login(username: string, password: string): Promise<AppState & { message: string }> {
  return request("/api/auth/login", { method: "POST", json: { username, password } });
}

export async function register(username: string, password: string): Promise<AppState & { message: string }> {
  return request("/api/auth/register", { method: "POST", json: { username, password } });
}

export async function logout(): Promise<{ message: string }> {
  return request("/api/auth/logout", { method: "POST" });
}

export async function fetchQuestionnaires(): Promise<{ questionnaires: Questionnaire[] }> {
  return request("/api/questionnaires");
}

export async function submitQuestionnaire(
  level: string,
  answers: Record<string, string | number | string[]>,
): Promise<AppState & { message: string }> {
  return request(`/api/questionnaires/${level}`, { method: "POST", json: { answers } });
}

export async function uploadTrades(file: File, window: string): Promise<AppState & { message: string; filename: string; tradeCount: number; lstmAccount: string | null }> {
  const form = new FormData();
  form.append("file", file);
  return request(`/api/trades/upload?window=${encodeURIComponent(window)}`, {
    method: "POST",
    body: form,
  });
}

export async function fetchRecommendations(input: {
  backend: string;
  topN: number;
}): Promise<RecommendResponse> {
  return request("/api/recommend", {
    method: "POST",
    json: { backend: input.backend, top_n: input.topN },
  });
}

export async function fetchTrends(strategyIds: string[]): Promise<TrendComparisonResponse> {
  return request("/api/trends", {
    method: "POST",
    json: { strategy_ids: strategyIds },
  });
}

export async function fetchStability(): Promise<StabilityResponse> {
  return request("/api/stability");
}

export async function updateSettings(input: {
  beta?: number;
  backend?: string;
  fusionAlpha?: number;
}): Promise<AppState & { message: string }> {
  return request("/api/settings", {
    method: "PATCH",
    json: {
      beta: input.beta,
      backend: input.backend,
      fusion_alpha: input.fusionAlpha,
    },
  });
}

export async function clearTrades(): Promise<AppState & { message: string }> {
  return request("/api/trades/clear", { method: "POST" });
}
