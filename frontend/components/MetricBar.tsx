type MetricBarProps = {
  label: string;
  value: number;
  tone?: "blue" | "green" | "amber" | "red";
};

export function MetricBar({ label, value, tone = "blue" }: MetricBarProps) {
  const safeValue = Math.max(0, Math.min(100, value));

  return (
    <div className="metric-row">
      <div className="metric-row-head">
        <span>{label}</span>
        <strong>{safeValue.toFixed(1)}</strong>
      </div>
      <div className="meter" aria-label={`${label} ${safeValue.toFixed(1)}分`}>
        <span className={`meter-fill ${tone}`} style={{ width: `${safeValue}%` }} />
      </div>
    </div>
  );
}
