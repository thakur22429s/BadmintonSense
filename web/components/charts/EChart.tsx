"use client";
import dynamic from "next/dynamic";
import type { EChartsOption } from "echarts";

const ReactECharts = dynamic(() => import("echarts-for-react").then((m) => m.default), {
  ssr: false,
  loading: () => <div className="h-[320px] grid place-items-center text-sm text-muted-foreground">Loading chart…</div>,
});

export function EChart({
  option,
  className = "",
  height = 320,
}: {
  option: EChartsOption;
  className?: string;
  height?: number;
}) {
  return (
    <div className={className} style={{ height }}>
      <ReactECharts
        option={option}
        style={{ height: "100%", width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge
      />
    </div>
  );
}
