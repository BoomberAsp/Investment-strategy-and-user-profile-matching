import { BadgePercent, ShieldCheck, TrendingUp } from "lucide-react";
import type { Recommendation } from "@/lib/api";
import { MetricBar } from "./MetricBar";
import { RadarChart } from "./RadarChart";

type RecommendationCardProps = {
  recommendation: Recommendation;
  rank: number;
};

export function RecommendationCard({ recommendation, rank }: RecommendationCardProps) {
  const radarValues = [
    { label: "风格", value: recommendation.style_similarity },
    { label: "绩效", value: recommendation.performance_score },
    { label: "风险", value: recommendation.risk_score },
    { label: "偏好", value: recommendation.preference_score },
    { label: "因子", value: recommendation.factor_score },
    { label: "聚类", value: recommendation.cluster_score },
    { label: "CCA", value: recommendation.cca_score },
  ];

  return (
    <article className="recommendation-card">
      <div className="card-title-row">
        <div>
          <span className="rank">Top {rank}</span>
          <h3>{recommendation.strategy_name}</h3>
          <p>{recommendation.factor_label}</p>
        </div>
        <div className="score-pill">
          <span>{recommendation.score.toFixed(1)}</span>
          <small>综合分</small>
        </div>
      </div>

      <div className="card-main">
        <RadarChart values={radarValues} />
        <div className="metric-stack">
          <MetricBar label="风格匹配" value={recommendation.style_similarity} tone="blue" />
          <MetricBar label="策略绩效" value={recommendation.performance_score} tone="green" />
          <MetricBar label="风险控制" value={recommendation.risk_score} tone="amber" />
          <MetricBar label="主动偏好" value={recommendation.preference_score} tone="red" />
          <MetricBar label="因子匹配" value={recommendation.factor_score} tone="blue" />
          <MetricBar label="聚类归属" value={recommendation.cluster_score} tone="green" />
          <MetricBar label="CCA匹配" value={recommendation.cca_score} tone="amber" />
        </div>
      </div>

      <div className="facts-grid">
        <div>
          <TrendingUp size={18} aria-hidden="true" />
          <span>25年收益</span>
          <strong>{recommendation.return_2025.toFixed(1)}%</strong>
        </div>
        <div>
          <ShieldCheck size={18} aria-hidden="true" />
          <span>最大回撤</span>
          <strong>{recommendation.max_drawdown_2025.toFixed(1)}%</strong>
        </div>
        <div>
          <BadgePercent size={18} aria-hidden="true" />
          <span>主题重合</span>
          <strong>{recommendation.theme_overlap.toFixed(1)}%</strong>
        </div>
      </div>

      <p className="explanation">{recommendation.explanation}</p>

      <div className="tag-row" aria-label="策略主题">
        {recommendation.themes.map((theme) => (
          <span key={theme}>{theme}</span>
        ))}
      </div>
    </article>
  );
}
