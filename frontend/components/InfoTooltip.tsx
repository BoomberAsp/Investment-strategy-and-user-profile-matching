"use client";

interface InfoTooltipProps {
  text: string;
  /** Icon size: "sm" for compact areas (score pill, fact items), "md" for metric labels */
  size?: "sm" | "md";
}

const SIZE_CLASS = {
  sm: "info-tooltip-sm",
  md: "info-tooltip-md",
} as const;

export function InfoTooltip({ text, size = "md" }: InfoTooltipProps) {
  return (
    <span
      className={`info-tooltip ${SIZE_CLASS[size]}`}
      tabIndex={0}
      role="tooltip"
      aria-label={text}
    >
      <span className="info-tooltip-icon" aria-hidden="true">
        ?
      </span>
      <span className="info-tooltip-popup">{text}</span>
    </span>
  );
}
