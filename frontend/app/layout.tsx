import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "投资策略匹配推荐系统",
  description: "客户交易画像驱动的投资策略匹配推荐系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
