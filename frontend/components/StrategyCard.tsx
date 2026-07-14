import type { Recommendation } from "@/lib/api";
import { RecommendationCard } from "./RecommendationCard";
import { InfoTooltip } from "./InfoTooltip";

type StrategyCardVariant = "compact" | "focused" | "side-left" | "side-right";

interface StrategyCardProps {
  recommendation: Recommendation;
  rank: number;
  isSelected: boolean;
  onSelect: (strategyId: string) => void;
  variant?: StrategyCardVariant;
  /** Customer profile summary (used in focused mode to show context) */
  customerProfile?: {
    trade_count: number;
    buy_ratio: number;
    themes: string[];
    top_symbols: string[];
  } | null;
  /** PCA explained variance for context */
  pcaVariance?: string | null;
}

export function StrategyCard({
  recommendation,
  rank,
  isSelected,
  onSelect,
  variant = "compact",
  customerProfile,
  pcaVariance,
}: StrategyCardProps) {
  function handleClick(e: React.MouseEvent) {
    e.stopPropagation();
    onSelect(recommendation.strategy_id);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      onSelect(recommendation.strategy_id);
    }
  }

  return (
    <div
      className={`strategy-card ${variant} ${isSelected ? "selected" : ""}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-label={`选择策略 ${recommendation.strategy_name}`}
    >
      {variant === "focused" ? (
        /* ======== Focused dashboard layout ======== */
        <article className="recommendation-card recommendation-card-focused">
          {/* ── Top identity bar ── */}
          <div className="focused-top">
            <div className="focused-top-left">
              <span className="rank">Top {rank}</span>
              <h3>{recommendation.strategy_name}</h3>
              <span className="focused-factor">
                {recommendation.factor_label}
              </span>
            </div>
            <div className="score-pill focused-score">
              <span>{recommendation.score.toFixed(1)}</span>
              <small>
                综合分
                <InfoTooltip
                  size="sm"
                  text="综合分由风格匹配、策略绩效、风险控制、主动偏好、因子匹配、聚类归属和CCA匹配七个维度按当前权重加权得到，用于综合排序推荐策略。"
                />
              </small>
            </div>
          </div>

          {/* ── Recommendation reason ── */}
          <div className="focused-section focused-reason">
            <h4>
              推荐理由
            </h4>
            <p>{recommendation.explanation}</p>
          </div>

          {/* ── Two-column body ── */}
          <div className="focused-body">
            {/* Left column: Profile matching analysis */}
            <div className="focused-col">
              <h4 className="focused-section-title">
                画像匹配分析
              </h4>

              <div className="focused-metric-cards">
                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    风格相似度
                    <InfoTooltip text="风格相似度基于客户与策略的交易画像特征计算，例如交易频率、买入比例、换手活跃度、持仓集中度、平均单笔金额和股票数量等。数值越高，说明交易行为越接近。" />
                  </div>
                  <div className="focused-metric-card-value blue">
                    {recommendation.style_similarity.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`风格相似度 ${recommendation.style_similarity.toFixed(1)}分`}>
                    <span className="meter-fill blue" style={{ width: `${recommendation.style_similarity}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    股票重合度
                    <InfoTooltip text="股票重合度表示客户持仓股票与策略持仓股票的交集比例（Jaccard相似度），反映在具体标的选择上的重叠程度。" />
                  </div>
                  <div className="focused-metric-card-value green">
                    {recommendation.symbol_overlap.toFixed(1)}%
                  </div>
                  <div className="meter" aria-label={`股票重合度 ${recommendation.symbol_overlap.toFixed(1)}%`}>
                    <span className="meter-fill green" style={{ width: `${recommendation.symbol_overlap}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    主题重合度
                    <InfoTooltip text="主题重合度表示策略主题标签与客户历史或主动选择主题之间的重合程度（Jaccard相似度）。" />
                  </div>
                  <div className="focused-metric-card-value amber">
                    {recommendation.theme_overlap.toFixed(1)}%
                  </div>
                  <div className="meter" aria-label={`主题重合度 ${recommendation.theme_overlap.toFixed(1)}%`}>
                    <span className="meter-fill amber" style={{ width: `${recommendation.theme_overlap}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    PCA 风格标签
                    <InfoTooltip text={`PCA 解释率表示主成分能够解释原始交易画像特征变异的比例${pcaVariance ? `（当前：${pcaVariance}）` : ""}，用于判断二维风格图是否能较好概括原始特征。风格标签基于客户在主成分空间中的位置划分。`} />
                  </div>
                  <div className="focused-metric-card-value teal">
                    {recommendation.factor_label}
                  </div>
                  {pcaVariance ? (
                    <div className="focused-pca-note">解释率: {pcaVariance}</div>
                  ) : null}
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    因子匹配
                    <InfoTooltip text="因子匹配基于因子分析（Factor Analysis）将6个交易特征压缩为少数潜在风格因子，在因子空间中计算客户与策略的余弦相似度，消除了特征间的冗余相关。" />
                  </div>
                  <div className="focused-metric-card-value blue">
                    {recommendation.factor_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`因子匹配 ${recommendation.factor_score.toFixed(1)}分`}>
                    <span className="meter-fill blue" style={{ width: `${recommendation.factor_score}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    聚类归属
                    <InfoTooltip text={`聚类归属基于K-Means将策略划分为不同风格簇，客户被分配到一个簇，同簇策略得分更高。当前策略归属：${recommendation.cluster_label}。`} />
                  </div>
                  <div className="focused-metric-card-value green">
                    ({recommendation.cluster_label}) {recommendation.cluster_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`聚类归属 ${recommendation.cluster_score.toFixed(1)}分`}>
                    <span className="meter-fill green" style={{ width: `${recommendation.cluster_score}%` }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Right column: Performance + Risk + Preference */}
            <div className="focused-col">
              <h4 className="focused-section-title">
                收益、风险与多维匹配
              </h4>

              <div className="focused-metric-cards">
                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    策略绩效
                    <InfoTooltip text="策略绩效反映策略近一年收益表现。通常收益越高，该得分越高，但具体得分经过归一化处理，便于跨策略比较。" />
                  </div>
                  <div className="focused-metric-card-value green">
                    {recommendation.performance_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`策略绩效 ${recommendation.performance_score.toFixed(1)}分`}>
                    <span className="meter-fill green" style={{ width: `${recommendation.performance_score}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    风险控制
                    <InfoTooltip text="风险控制主要反映策略的回撤和波动风险。最大回撤越小、风险越低，风险控制得分通常越高。" />
                  </div>
                  <div className="focused-metric-card-value amber">
                    {recommendation.risk_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`风险控制 ${recommendation.risk_score.toFixed(1)}分`}>
                    <span className="meter-fill amber" style={{ width: `${recommendation.risk_score}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    偏好匹配
                    <InfoTooltip text="偏好匹配反映策略主题与用户主动选择的主题、股票或偏好权重之间的匹配程度。命中更高权重的偏好时，得分更高。" />
                  </div>
                  <div className="focused-metric-card-value red">
                    {recommendation.preference_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`偏好匹配 ${recommendation.preference_score.toFixed(1)}分`}>
                    <span className="meter-fill red" style={{ width: `${recommendation.preference_score}%` }} />
                  </div>
                </div>

                <div className="focused-metric-card">
                  <div className="focused-metric-card-label">
                    CCA 匹配
                    <InfoTooltip text="CCA匹配通过典型相关分析学习交易风格与绩效指标之间的最大相关方向，衡量客户的交易风格在多大程度上接近那些历史上表现优异的策略。" />
                  </div>
                  <div className="focused-metric-card-value amber">
                    {recommendation.cca_score.toFixed(1)}
                  </div>
                  <div className="meter" aria-label={`CCA匹配 ${recommendation.cca_score.toFixed(1)}分`}>
                    <span className="meter-fill amber" style={{ width: `${recommendation.cca_score}%` }} />
                  </div>
                </div>
              </div>

              {/* Return & drawdown facts */}
              <div className="focused-fact-row">
                <div className="focused-fact-item">
                  <span>25年收益</span>
                  <strong className={recommendation.return_2025 >= 0 ? "positive" : "negative"}>
                    {recommendation.return_2025 >= 0 ? "+" : ""}{recommendation.return_2025.toFixed(1)}%
                  </strong>
                </div>
                <div className="focused-fact-item">
                  <span>
                    最大回撤
                    <InfoTooltip
                      size="sm"
                      text="最大回撤表示策略净值从历史高点下跌到低点的最大跌幅，用于衡量极端亏损风险。回撤越小，策略抗风险能力越强。"
                    />
                  </span>
                  <strong className="negative">
                    {recommendation.max_drawdown_2025.toFixed(1)}%
                  </strong>
                </div>
                <div className="focused-fact-item">
                  <span>年化收益</span>
                  <strong className={recommendation.annual_return >= 0 ? "positive" : "negative"}>
                    {recommendation.annual_return >= 0 ? "+" : ""}{recommendation.annual_return.toFixed(1)}%
                  </strong>
                </div>
              </div>
            </div>
          </div>

          {/* ── Bottom: theme tags + top symbols ── */}
          <div className="focused-footer">
            <div className="focused-footer-row">
              <div className="focused-footer-block">
                <span className="focused-footer-label">策略主题</span>
                <div className="tag-row">
                  {recommendation.themes.map((t) => (
                    <span key={t}>{t}</span>
                  ))}
                </div>
              </div>
              {recommendation.top_symbols.length > 0 && (
                <div className="focused-footer-block">
                  <span className="focused-footer-label">策略持仓（部分）</span>
                  <div className="tag-row">
                    {recommendation.top_symbols.slice(0, 8).map((s) => (
                      <span key={s} className="symbol-tag">{s}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {customerProfile && (
              <div className="focused-footer-row">
                <div className="focused-footer-block">
                  <span className="focused-footer-label">客户主题</span>
                  <div className="tag-row">
                    {customerProfile.themes.length > 0
                      ? customerProfile.themes.map((t) => (
                          <span key={t} className="customer-tag">{t}</span>
                        ))
                      : <span className="no-data">暂无数据</span>}
                  </div>
                </div>
                {customerProfile.top_symbols.length > 0 && (
                  <div className="focused-footer-block">
                    <span className="focused-footer-label">客户持仓（部分）</span>
                    <div className="tag-row">
                      {customerProfile.top_symbols.slice(0, 8).map((s) => (
                        <span key={s} className="symbol-tag customer-symbol">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>
      ) : (
        /* ======== Compact / Side: full RecommendationCard ======== */
        <RecommendationCard recommendation={recommendation} rank={rank} />
      )}
    </div>
  );
}
