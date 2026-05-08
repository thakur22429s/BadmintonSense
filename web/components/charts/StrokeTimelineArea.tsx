"use client";
import { EChart } from "./EChart";
import { CHART_PALETTE, CHART_LABEL, CHART_TOOLTIP_BG, CHART_GRID, CHART_AXIS } from "@/lib/styles";

export function StrokeTimelineArea({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  classNames: _classNames,
  series,
  setBoundaries,
}: {
  classNames: string[];
  series: { name: string; values: number[] }[];
  setBoundaries: number[];
}) {
  return (
    <EChart
      height={340}
      option={{
        tooltip: { trigger: "axis", backgroundColor: CHART_TOOLTIP_BG, borderColor: "#2a323e", textStyle: { color: CHART_LABEL } },
        legend: { data: series.map((s) => s.name), textStyle: { color: CHART_LABEL }, top: 0, type: "scroll" },
        grid: { left: 50, right: 30, top: 50, bottom: 40 },
        xAxis: {
          type: "category",
          data: series[0]?.values.map((_, i) => `${i + 1}`) ?? [],
          axisLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS, fontSize: 10 },
          splitLine: { show: false },
        },
        yAxis: {
          type: "value",
          axisLine: { lineStyle: { color: CHART_GRID } },
          splitLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS },
        },
        color: CHART_PALETTE,
        series: series.map((s) => ({
          name: s.name,
          type: "line",
          stack: "total",
          areaStyle: { opacity: 0.7 },
          lineStyle: { width: 0 },
          showSymbol: false,
          data: s.values,
        })),
        markLine: setBoundaries.length
          ? {
              symbol: "none",
              silent: true,
              lineStyle: { color: "#5a667a", type: "dashed" },
              data: setBoundaries.map((x) => ({ xAxis: `${x}` })),
            }
          : undefined,
      }}
    />
  );
}
