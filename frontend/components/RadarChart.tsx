type RadarChartProps = {
  values: {
    label: string;
    value: number;
  }[];
};

export function RadarChart({ values }: RadarChartProps) {
  const center = 86;
  const radius = 54;
  const count = values.length;
  const points = values
    .map((item, index) => {
      const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
      const distance = radius * Math.max(0, Math.min(100, item.value)) / 100;
      return `${center + Math.cos(angle) * distance},${center + Math.sin(angle) * distance}`;
    })
    .join(" ");

  return (
    <svg className="radar" viewBox="0 0 172 172" role="img" aria-label="推荐指标雷达图">
      {[0.33, 0.66, 1].map((scale) => (
        <circle
          key={scale}
          cx={center}
          cy={center}
          r={radius * scale}
          className="radar-grid"
        />
      ))}
      {values.map((item, index) => {
        const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
        const x = center + Math.cos(angle) * radius;
        const y = center + Math.sin(angle) * radius;
        const labelX = center + Math.cos(angle) * (radius + 22);
        const labelY = center + Math.sin(angle) * (radius + 22);
        return (
          <g key={item.label}>
            <line x1={center} y1={center} x2={x} y2={y} className="radar-axis" />
            <text x={labelX} y={labelY} textAnchor="middle" dominantBaseline="middle" className="radar-label">
              {item.label}
            </text>
          </g>
        );
      })}
      <polygon points={points} className="radar-area" />
      <polyline points={`${points} ${points.split(" ")[0]}`} className="radar-line" />
    </svg>
  );
}
