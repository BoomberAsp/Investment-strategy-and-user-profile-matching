"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface CollapsibleSidebarProps {
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

export function CollapsibleSidebar({
  expanded,
  onToggle,
  children,
}: CollapsibleSidebarProps) {
  return (
    <>
      <div
        className={`sidebar-backdrop ${expanded ? "visible" : ""}`}
        onClick={onToggle}
        aria-hidden="true"
      />

      <aside
        className={`sidebar ${expanded ? "expanded" : "collapsed"}`}
        aria-label="控制面板"
      >
        <button
          className="sidebar-toggle"
          onClick={onToggle}
          type="button"
          aria-label={expanded ? "折叠侧边栏" : "展开侧边栏"}
        >
          {expanded ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>

        <div className="sidebar-inner">{children}</div>
      </aside>
    </>
  );
}
