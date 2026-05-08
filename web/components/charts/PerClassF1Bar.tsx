"use client";
import { EChart } from "./EChart";
import { CHART_LABEL, CHART_TOOLTIP_BG, CHART_GRID, CHART_AXIS, CHART_PALETTE } from "@/lib/styles";

export function PerClassF1Bar({
  classNames,
  series,
}: {
  classNames: string[];
  series: { name: string; values: number[] }[];
}) {
  return (
    <EChart
      height={320}
      option={{
        tooltip: { trigger: "axis", backgroundColor: CHART_TOOLTIP_BG, borderColor: "#2a323e", textStyle: { color: CHART_LABEL } },
        legend: { data: series.map((s) => s.name), textStyle: { color: CHART_LABEL }, top: 0 },
        grid: { left: 50, right: 20, top: 40, bottom: 70 },
        xAxis: {
          type: "category",
          data: classNames,
          axisLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS, rotate: 30, fontSize: 11 },
        },
        yAxis: {
          type: "value",
          min: 0,
          max: 1,
          axisLine: { lineStyle: { color: CHART_GRID } },
          splitLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS },
        },
        color: CHART_PALETTE,
        series: series.map((s) => ({
          name: s.name,
          type: "bar",
          data: s.values,
          itemStyle: { borderRadius: [3, 3, 0, 0] },
        })),
      }}
    />
  );
}
