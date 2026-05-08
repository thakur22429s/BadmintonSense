"use client";
import { EChart } from "./EChart";
import { CHART_PALETTE, CHART_LABEL, CHART_TOOLTIP_BG } from "@/lib/styles";

export function PlayerStyleRadar({
  classNames,
  players,
}: {
  classNames: string[];
  players: { name: string; values: number[] }[];
}) {
  const max = Math.max(1, ...players.flatMap((p) => p.values));
  return (
    <EChart
      height={360}
      option={{
        tooltip: {
          backgroundColor: CHART_TOOLTIP_BG,
          borderColor: "#2a323e",
          textStyle: { color: CHART_LABEL },
        },
        legend: { data: players.map((p) => p.name), textStyle: { color: CHART_LABEL }, top: 0 },
        radar: {
          indicator: classNames.map((c) => ({ name: c, max })),
          axisName: { color: CHART_LABEL, fontSize: 11 },
          splitArea: { areaStyle: { color: ["#0e1218", "#10141b"] } },
          splitLine: { lineStyle: { color: "#1f242c" } },
          axisLine: { lineStyle: { color: "#1f242c" } },
        },
        color: CHART_PALETTE,
        series: [
          {
            type: "radar",
            data: players.map((p, i) => ({
              name: p.name,
              value: p.values,
              areaStyle: { opacity: 0.18, color: CHART_PALETTE[i % CHART_PALETTE.length] },
              lineStyle: { width: 2 },
              symbolSize: 5,
            })),
          },
        ],
      }}
    />
  );
}
