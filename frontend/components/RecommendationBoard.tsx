"use client";

import { useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import type { Recommendation } from "@/lib/api";
import { StrategyCard } from "./StrategyCard";

interface RecommendationBoardProps {
  recommendations: Recommendation[];
  isLoading: boolean;
  isRecommending: boolean;
  selectedStrategyId: string | null;
  onSelectStrategy: (strategyId: string) => void;
  /** Customer profile summary passed through to focused card for richer context */
  customerProfile?: {
    trade_count: number;
    buy_ratio: number;
    themes: string[];
    top_symbols: string[];
  } | null;
  pcaVariance?: string | null;
}

export function RecommendationBoard({
  recommendations,
  isLoading,
  isRecommending,
  selectedStrategyId,
  onSelectStrategy,
  customerProfile,
  pcaVariance,
}: RecommendationBoardProps) {
  const [isFocusMode, setIsFocusMode] = useState(false);
  const boardRef = useRef<HTMLDivElement>(null);

  const selectedIndex = recommendations.findIndex(
    (r) => r.strategy_id === selectedStrategyId,
  );

  function enterFocusMode(strategyId: string) {
    onSelectStrategy(strategyId);
    setIsFocusMode(true);
  }

  function exitFocusMode() {
    setIsFocusMode(false);
  }

  function handleStageKeyDown(e: React.KeyboardEvent) {
    if (!isFocusMode) return;
    if (e.key === "Escape") {
      e.preventDefault();
      exitFocusMode();
      return;
    }
    if (e.key === "ArrowLeft" && selectedIndex > 0) {
      e.preventDefault();
      onSelectStrategy(recommendations[selectedIndex - 1].strategy_id);
      return;
    }
    if (e.key === "ArrowRight" && selectedIndex < recommendations.length - 1) {
      e.preventDefault();
      onSelectStrategy(recommendations[selectedIndex + 1].strategy_id);
    }
  }

  function handleStageClick(e: React.MouseEvent) {
    if (!isFocusMode) return;
    if (e.target === e.currentTarget) {
      exitFocusMode();
    }
  }

  // Build visible cards for focus mode: up to 3 (prev, current, next)
  const focusCards: Array<{
    rec: Recommendation;
    index: number;
    variant: "focused" | "side-left" | "side-right";
  }> = [];
  if (isFocusMode && selectedIndex >= 0) {
    if (selectedIndex > 0) {
      focusCards.push({
        rec: recommendations[selectedIndex - 1],
        index: selectedIndex - 1,
        variant: "side-left",
      });
    }
    focusCards.push({
      rec: recommendations[selectedIndex],
      index: selectedIndex,
      variant: "focused",
    });
    if (selectedIndex < recommendations.length - 1) {
      focusCards.push({
        rec: recommendations[selectedIndex + 1],
        index: selectedIndex + 1,
        variant: "side-right",
      });
    }
  }

  return (
    <section className="recommendation-board" aria-label="推荐策略列表">
      <div className="results-head">
        <div>
          <h2>推荐结果</h2>
          <p>
            {isRecommending
              ? "正在计算..."
              : `${recommendations.length} 个策略候选`}
          </p>
        </div>
      </div>

      {isLoading ? (
        <div className="empty-state">
          <Loader2 size={24} className="spin" aria-hidden="true" />
          <span>加载数据中</span>
        </div>
      ) : isFocusMode ? (
        /* ======== Focus Mode ======== */
        <div
          className="focus-stage"
          ref={boardRef}
          onClick={handleStageClick}
          onKeyDown={handleStageKeyDown}
          tabIndex={0}
          role="region"
          aria-label="聚焦浏览策略"
        >
          {focusCards.map(({ rec, index, variant }) => (
            <StrategyCard
              key={rec.strategy_id}
              recommendation={rec}
              rank={index + 1}
              isSelected={rec.strategy_id === selectedStrategyId}
              onSelect={(id) => {
                if (id === selectedStrategyId) return;
                onSelectStrategy(id);
              }}
              variant={variant}
              customerProfile={
                variant === "focused" ? customerProfile : null
              }
              pcaVariance={variant === "focused" ? pcaVariance : null}
            />
          ))}
        </div>
      ) : (
        /* ======== Normal Mode ======== */
        <div className="strategy-scroll">
          <div className="strategy-scroll-inner">
            {recommendations.map((rec, index) => (
              <StrategyCard
                key={rec.strategy_id}
                recommendation={rec}
                rank={index + 1}
                isSelected={rec.strategy_id === selectedStrategyId}
                onSelect={enterFocusMode}
                variant="compact"
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
