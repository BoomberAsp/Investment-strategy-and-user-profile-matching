"use client";

import { useMemo, useState } from "react";
import { TrendingUp } from "lucide-react";
import type { CustomerTrendMeta, MergedTrendPoint, MergedTrendResponse } from "@/lib/api";
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

interface CustomerTrendChartProps {
  accountId: string;
  customerData: { trend: { date: string; cumulativeReturn: number }[]; meta: CustomerTrendMeta | null } | null;
  /** Merged trend with both customer and strategy returns */
  mergedData: MergedTrendPoint[] | null;
  /** Whether strategy trend is available */
  hasStrategy: boolean;
  strategyName?: string;
  strategyMeta?: CustomerTrendMeta | null;
  isLoading: boolean;
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
  payload?: Array<{ payload: MergedTrendPoint }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload;
  return (
    <div className="trend-tooltip">
      <div className="trend-tooltip-date">{point.date}</div>
      {point.customerReturn != null && (
        <div className="trend-tooltip-row trend-tooltip-customer">
          <span className="trend-tooltip-dot trend-tooltip-dot-customer" />
          客户账户 {formatPct(point.customerReturn)}
        </div>
      )}
      {point.strategyReturn != null && (
        <div className="trend-tooltip-row trend-tooltip-strategy">
          <span className="trend-tooltip-dot trend-tooltip-dot-strategy" />
          当前策略 {formatPct(point.strategyReturn)}
        </div>
      )}
    </div>
  );
}

export function CustomerTrendChart({
  accountId,
  customerData,
  mergedData,
  hasStrategy,
  strategyName,
  strategyMeta,
  isLoading,
}: CustomerTrendChartProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  // Use mergedData if available, else build from customerData alone
  const chartData = mergedData ?? (
    customerData?.trend?.map((p) => ({
      date: p.date,
      customerReturn: p.cumulativeReturn,
      strategyReturn: null as number | null,
    })) ?? []
  );

  const hasData = chartData.length > 0;
  const customerMeta = customerData?.meta ?? null;
  const customerTrendEmpty = !customerData?.trend || customerData.trend.length === 0;
  const customerInsufficient = customerMeta?.dataQuality === "insufficient_price_data";

  const { yDomain, ticks } = useMemo(() => {
    if (!hasData) {
      return {
        yDomain: [-20, 20] as [number, number],
        ticks: [-20, -10, 0, 10, 20],
      };
    }
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const pt of chartData) {
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
  }, [hasData, chartData]);

  const activePoint =
    activeIndex != null && chartData.length > 0 ? chartData[activeIndex] : null;
  const crosshairY =
    activePoint != null
      ? (activePoint.strategyReturn ?? activePoint.customerReturn) ?? 0
      : 0;

  if (isLoading) {
    return (
      <section className="trend-section" aria-label="收益曲线">
        <h2>客户账户 {accountId} 收益曲线</h2>
        <div className="trend-chart-container trend-empty">
          <TrendingUp size={40} strokeWidth={1.5} aria-hidden="true" />
          <p>正在加载收益数据...</p>
        </div>
      </section>
    );
  }

  // Customer data insufficient
  if (customerInsufficient) {
    return (
      <section className="trend-section" aria-label="收益曲线">
        <h2>客户账户 {accountId} 收益曲线</h2>
        <div className="trend-chart-container trend-empty">
          <TrendingUp size={40} strokeWidth={1.5} aria-hidden="true" />
          <p>缺失股票价格数据比例过高，无法可靠绘制收益曲线。</p>
          {customerMeta?.warnings && customerMeta.warnings.length > 0 && (
            <p className="trend-hint">{customerMeta.warnings[0]}</p>
          )}
        </div>
      </section>
    );
  }

  // No data at all
  if (!hasData && customerTrendEmpty) {
    return (
      <section className="trend-section" aria-label="收益曲线">
        <h2>客户账户 {accountId} 收益曲线</h2>
        <div className="trend-chart-container trend-empty">
          <TrendingUp size={40} strokeWidth={1.5} aria-hidden="true" />
          <p>暂无收益数据。</p>
          {customerMeta?.warnings && customerMeta.warnings.length > 0 && (
            <p className="trend-hint">{customerMeta.warnings[0]}</p>
          )}
        </div>
      </section>
    );
  }

  // Title
  const title = strategyName
    ? `客户账户 ${accountId} vs ${strategyName}`
    : `客户账户 ${accountId} 收益曲线`;

  return (
    <section className="trend-section" aria-label="收益曲线">
      <h2>{title}</h2>

      {/* Customer warnings */}
      {customerMeta?.warnings && customerMeta.warnings.length > 0 && (
        <div className="trend-warning">
          {customerMeta.warnings.map((w, i) => (
            <p key={i} style={{ margin: i > 0 ? "4px 0 0" : 0 }}>
              {w}
            </p>
          ))}
        </div>
      )}

      {/* Customer partial missing prices */}
      {customerMeta?.dataQuality === "partial_missing_prices" && (
        <div className="trend-warning">
          <p>部分股票缺少价格数据，已按买入价格固定估值，曲线为近似结果。</p>
        </div>
      )}

      {/* Strategy warnings */}
      {hasStrategy && strategyMeta?.warnings && strategyMeta.warnings.length > 0 && (
        <div className="trend-warning">
          {strategyMeta.warnings.map((w, i) => (
            <p key={i} style={{ margin: i > 0 ? "4px 0 0" : 0, color: "#b45309" }}>
              {w}
            </p>
          ))}
        </div>
      )}

      {/* Strategy status messages */}
      {hasStrategy && strategyMeta?.dataQuality === "insufficient_price_data" && (
        <div className="trend-warning">
          <p>当前策略缺失股票价格数据比例过高，暂无法可靠绘制策略收益曲线。</p>
        </div>
      )}
      {hasStrategy && strategyMeta?.dataQuality === "insufficient_trade_data" && (
        <div className="trend-warning">
          <p>当前策略缺少数量或金额字段，暂无法可靠重建收益曲线。</p>
        </div>
      )}
      {hasStrategy && strategyMeta?.dataQuality === "partial_missing_prices" && (
        <div className="trend-warning">
          <p>当前策略部分股票缺少价格数据，已按成交价固定估值，曲线为近似结果。</p>
        </div>
      )}

      {!hasStrategy && (
        <p className="trend-hint" style={{ marginBottom: 8 }}>
          选择推荐策略后可对比收益曲线。
        </p>
      )}

      <div className="trend-chart-container">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={chartData}
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

            {/* Customer line */}
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
              name={`客户账户 ${accountId}`}
              connectNulls
            />

            {/* Strategy line */}
            {hasStrategy && (
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
                name={strategyName ?? "当前策略"}
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
