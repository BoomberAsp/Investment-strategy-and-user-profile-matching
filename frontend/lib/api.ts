export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
const AUTH_PATHS = new Set(["/api/auth/login", "/api/auth/register"]);

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function apiUrl(path: string): string {
  if (!API_BASE) return path;
  const backendPath = path.replace(/^\/api/, "");
  return `${API_BASE}${backendPath}`;
}

type RequestOptions = RequestInit & { json?: unknown; suppressAuthEvent?: boolean };

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { json, suppressAuthEvent, ...init } = options;
  const headers = new Headers(options.headers);
  let body = options.body;
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }

  const response = await fetch(apiUrl(path), {
    ...init,
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
    if (response.status === 401 && !suppressAuthEvent && !AUTH_PATHS.has(path) && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("investment:unauthorized", { detail: message }));
    }
    throw new ApiError(message, response.status);
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

export type CustomerStatus =
  | "new"
  | "needs_questionnaire"
  | "needs_trades"
  | "needs_profile"
  | "ready_to_recommend";

export type Customer = {
  customerId: string;
  ownerUserId: string;
  name: string;
  status: CustomerStatus;
  statusLabel: string;
  nextAction: string;
  note: string;
  createdAt: string;
  lastUpdated: string;
  completedLevels: string[];
  uploadCount: number;
  tradeCount: number;
  hasProfile: boolean;
  confidenceLevel: "low" | "medium" | "high" | null;
};

export type AppState = {
  user: User;
  currentCustomer: Customer;
  customers: Customer[];
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

export async function fetchSession(options: { suppressAuthEvent?: boolean } = {}): Promise<AppState> {
  return request<AppState>("/api/auth/me", { suppressAuthEvent: options.suppressAuthEvent });
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

function customerQuery(customerId?: string): string {
  return customerId ? `?customer_id=${encodeURIComponent(customerId)}` : "";
}

export async function fetchCustomers(): Promise<{ customers: Customer[] }> {
  return request("/api/customers");
}

export async function createCustomer(input: { name: string; note?: string }): Promise<AppState & { message: string }> {
  return request("/api/customers", {
    method: "POST",
    json: { name: input.name, note: input.note ?? "" },
  });
}

export async function fetchCustomerState(customerId: string): Promise<AppState> {
  return request(`/api/customers/${encodeURIComponent(customerId)}`);
}

export async function updateCustomer(input: { customerId: string; name?: string; note?: string; status?: string }): Promise<AppState & { message: string }> {
  return request(`/api/customers/${encodeURIComponent(input.customerId)}`, {
    method: "PATCH",
    json: { name: input.name, note: input.note, status: input.status },
  });
}

export async function fetchQuestionnaires(customerId?: string): Promise<{ questionnaires: Questionnaire[] }> {
  return request(`/api/questionnaires${customerQuery(customerId)}`);
}

export async function submitQuestionnaire(
  level: string,
  answers: Record<string, string | number | string[]>,
  customerId?: string,
): Promise<AppState & { message: string }> {
  return request(`/api/questionnaires/${level}${customerQuery(customerId)}`, { method: "POST", json: { answers } });
}

export async function uploadTrades(file: File, window: string, customerId?: string): Promise<AppState & { message: string; filename: string; tradeCount: number; lstmAccount: string | null }> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ window });
  if (customerId) params.set("customer_id", customerId);
  return request(`/api/trades/upload?${params.toString()}`, {
    method: "POST",
    body: form,
  });
}

export async function fetchRecommendations(input: {
  backend: string;
  topN: number;
  customerId?: string;
}): Promise<RecommendResponse> {
  return request(`/api/recommend${customerQuery(input.customerId)}`, {
    method: "POST",
    json: { backend: input.backend, top_n: input.topN },
  });
}

export async function fetchTrends(strategyIds: string[], customerId?: string): Promise<TrendComparisonResponse> {
  return request(`/api/trends${customerQuery(customerId)}`, {
    method: "POST",
    json: { strategy_ids: strategyIds },
  });
}

export async function fetchStability(customerId?: string): Promise<StabilityResponse> {
  return request(`/api/stability${customerQuery(customerId)}`);
}

export async function updateSettings(input: {
  beta?: number;
  backend?: string;
  fusionAlpha?: number;
  customerId?: string;
}): Promise<AppState & { message: string }> {
  return request(`/api/settings${customerQuery(input.customerId)}`, {
    method: "PATCH",
    json: {
      beta: input.beta,
      backend: input.backend,
      fusion_alpha: input.fusionAlpha,
    },
  });
}

export async function clearTrades(customerId?: string): Promise<AppState & { message: string }> {
  return request(`/api/trades/clear${customerQuery(customerId)}`, { method: "POST" });
}
