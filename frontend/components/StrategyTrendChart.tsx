/**
 * DEPRECATED: Old trend chart component. Do not use for new trend display.
 *
 * Replaced by CustomerTrendChart.tsx — which only shows customer account curves
 * and does no financial computation in the component.
 *
 * Kept for reference only.
 */
"use client";

import { useMemo, useState } from "react";
import { TrendingUp } from "lucide-react";
import type { AlignedTrendPoint } from "@/lib/api";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface StrategyTrendChartProps {
  title: string;
  data: AlignedTrendPoint[] | null;
  isLoading: boolean;
  hasCustomerTrend: boolean;
  hasStrategyTrend: boolean;
  warnings: string[];
}

function formatPct(value: number | null): string {
  if (value == null) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: AlignedTrendPoint }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload;
  return (
    <div className="trend-tooltip">
      <div className="trend-tooltip-date">{point.date}</div>
      <div className="trend-tooltip-row trend-tooltip-customer">
        <span className="trend-tooltip-dot trend-tooltip-dot-customer" />
        客户账户 {formatPct(point.customerReturn)}
      </div>
      <div className="trend-tooltip-row trend-tooltip-strategy">
        <span className="trend-tooltip-dot trend-tooltip-dot-strategy" />
        当前策略 {formatPct(point.strategyReturn)}
      </div>
    </div>
  );
}

export function StrategyTrendChart({
  title,
  data,
  isLoading,
  hasCustomerTrend,
  hasStrategyTrend,
  warnings,
}: StrategyTrendChartProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const hasData = data && data.length > 0;

  const { yDomain, ticks } = useMemo(() => {
    if (!hasData)
      return {
        yDomain: [-20, 20] as [number, number],
        ticks: [-20, -10, 0, 10, 20],
      };
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const pt of data) {
      if (pt.customerReturn != null) {
        minVal = Math.min(minVal, pt.customerReturn);
        maxVal = Math.max(maxVal, pt.customerReturn);
      }
      if (pt.strategyReturn != null) {
        minVal = Math.min(minVal, pt.strategyReturn);
        maxVal = Math.max(maxVal, pt.strategyReturn);
      }
    }
    if (!isFinite(minVal)) {
      minVal = -20;
      maxVal = 20;
    }
    const pad = Math.max((maxVal - minVal) * 0.1, 2);
    const lo = Math.floor((minVal - pad) / 5) * 5;
    const hi = Math.ceil((maxVal + pad) / 5) * 5;
    const step = Math.max(Math.ceil((hi - lo) / 6 / 5) * 5, 5);
    const t: number[] = [];
    for (let v = lo; v <= hi; v += step) t.push(v);
    return { yDomain: [lo, hi] as [number, number], ticks: t };
  }, [hasData, data]);

  const activePoint =
    activeIndex != null && data ? data[activeIndex] : null;
  const crosshairY =
    activePoint != null
      ? (activePoint.strategyReturn ?? activePoint.customerReturn) ?? 0
      : 0;

  if (isLoading) {
    return (
      <section className="trend-section" aria-label="走势对比">
        <h2>{title}</h2>
        <div className="trend-chart-container trend-empty">
          <TrendingUp size={40} strokeWidth={1.5} aria-hidden="true" />
          <p>正在加载走势数据...</p>
        </div>
      </section>
    );
  }

  if (!hasData) {
    return (
      <section className="trend-section" aria-label="走势对比">
        <h2>{title}</h2>
        <div className="trend-chart-container trend-empty">
          <TrendingUp size={40} strokeWidth={1.5} aria-hidden="true" />
          <p>缺少日度股价数据，无法可靠绘制收益曲线。</p>
          {warnings.length > 0 && (
            <p className="trend-hint">{warnings[0]}</p>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className="trend-section" aria-label="走势对比">
      <h2>{title}</h2>

      {warnings.length > 0 && (
        <div className="trend-warning">
          {warnings.map((w, i) => (
            <p key={i} style={{ margin: i > 0 ? "4px 0 0" : 0 }}>
              {w}
            </p>
          ))}
        </div>
      )}

      <div className="trend-chart-container">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={data}
            margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            onMouseMove={(e) => {
              if (e.activeTooltipIndex != null) {
                setActiveIndex(Number(e.activeTooltipIndex));
              }
            }}
            onMouseLeave={() => setActiveIndex(null)}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e8edf5" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: "#667085" }}
              tickLine={false}
              axisLine={{ stroke: "#d8dee9" }}
              interval="preserveStartEnd"
              minTickGap={60}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#667085" }}
              tickLine={false}
              axisLine={false}
              domain={yDomain}
              ticks={ticks}
              tickFormatter={(v: number) => `${v}%`}
              width={50}
              label={{
                value: "累计收益率（%）",
                angle: -90,
                position: "insideLeft",
                offset: 0,
                style: { fontSize: 12, fill: "#667085", textAnchor: "middle" },
              }}
            />
            <Tooltip content={<CustomTooltip />} cursor={false} />
            <Legend
              iconType="line"
              wrapperStyle={{ fontSize: 13, paddingTop: 8 }}
            />
            {hasCustomerTrend && (
              <Line
                type="monotone"
                dataKey="customerReturn"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: "#2563eb",
                  stroke: "#fff",
                  strokeWidth: 2,
                }}
                name="客户账户"
                connectNulls
              />
            )}
            {hasStrategyTrend && (
              <Line
                type="monotone"
                dataKey="strategyReturn"
                stroke="#0f766e"
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: "#0f766e",
                  stroke: "#fff",
                  strokeWidth: 2,
                }}
                name="当前策略"
                connectNulls
              />
            )}

            {/* Crosshair */}
            {activePoint != null && (
              <>
                <ReferenceLine
                  x={activePoint.date}
                  stroke="#94a3b8"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
                <ReferenceLine
                  y={crosshairY}
                  stroke="#94a3b8"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
                {activePoint.customerReturn != null && (
                  <ReferenceDot
                    x={activePoint.date}
                    y={activePoint.customerReturn}
                    r={5}
                    fill="#2563eb"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
                {activePoint.strategyReturn != null && (
                  <ReferenceDot
                    x={activePoint.date}
                    y={activePoint.strategyReturn}
                    r={5}
                    fill="#0f766e"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
