"use client";

const OPTIONS = [
  { label: "默认", value: "md", scale: 1 },
  { label: "大", value: "lg", scale: 1.08 },
  { label: "超大", value: "xl", scale: 1.18 },
] as const;

export type FontScaleKey = (typeof OPTIONS)[number]["value"];

export function normalizeFontScaleKey(value: string | null): FontScaleKey {
  if (value === "lg" || value === "xl") return value;
  return "md";
}

export function getFontScale(key: FontScaleKey): number {
  return OPTIONS.find((o) => o.value === key)?.scale ?? 1;
}

interface FontSizeControlProps {
  value: FontScaleKey;
  onChange: (key: FontScaleKey) => void;
}

export function FontSizeControl({ value, onChange }: FontSizeControlProps) {
  return (
    <div>
      <div className="sidebar-section-title">显示设置</div>
      <div
        className="font-size-control"
        role="radiogroup"
        aria-label="字体大小"
      >
        {OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={`font-size-btn ${value === opt.value ? "active" : ""}`}
            onClick={() => onChange(opt.value)}
            type="button"
            role="radio"
            aria-checked={value === opt.value}
            aria-label={`字体大小 ${opt.label}`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
