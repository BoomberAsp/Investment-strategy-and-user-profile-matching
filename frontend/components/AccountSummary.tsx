import { Loader2 } from "lucide-react";

interface AccountSummaryProps {
  customerId: string;
  tradeCount: number | string;
  buyRatio: string;
  concentration: string;
  pcaVariance: string;
  isLoading: boolean;
}

const STATS = [
  { key: "account", label: "账户" },
  { key: "trades", label: "交易笔数" },
  { key: "buyRatio", label: "买入占比" },
  { key: "concentration", label: "集中度" },
  { key: "pca", label: "PCA 解释率" },
] as const;

export function AccountSummary({
  customerId,
  tradeCount,
  buyRatio,
  concentration,
  pcaVariance,
  isLoading,
}: AccountSummaryProps) {
  const values: Record<string, string> = {
    account: customerId,
    trades: String(tradeCount),
    buyRatio,
    concentration,
    pca: pcaVariance,
  };

  return (
    <section className="account-summary" aria-label="账户概览">
      {STATS.map(({ key, label }) => (
        <div key={key} className="account-stat">
          <span className="account-stat-label">{label}</span>
          {isLoading ? (
            <Loader2 size={20} className="spin" aria-hidden="true" />
          ) : (
            <strong className="account-stat-value">
              {values[key] || "--"}
            </strong>
          )}
        </div>
      ))}
    </section>
  );
}
