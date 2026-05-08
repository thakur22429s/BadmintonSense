"use client";
import { EChart } from "./EChart";
import { CHART_PALETTE, CHART_LABEL, CHART_TOOLTIP_BG } from "@/lib/styles";

export function StrokeDistributionDonut({
  title,
  classNames,
  counts,
}: {
  title: string;
  classNames: string[];
  counts: number[];
}) {
  const total = counts.reduce((a, b) => a + b, 0);
  const data = classNames
    .map((name, i) => ({ name, value: counts[i] }))
    .filter((d) => d.value > 0);

  return (
    <EChart
      height={300}
      option={{
        title: { text: title, left: "left", textStyle: { color: CHART_LABEL, fontSize: 13, fontWeight: 600 } },
        tooltip: {
          trigger: "item",
          backgroundColor: CHART_TOOLTIP_BG,
          borderColor: "#2a323e",
          textStyle: { color: CHART_LABEL },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter: (p: any) => `${p.name}<br/>${p.value} strokes (${((p.value / total) * 100).toFixed(1)}%)`,
        },
        legend: { show: false },
        color: CHART_PALETTE,
        series: [
          {
            type: "pie",
            radius: ["55%", "82%"],
            center: ["50%", "55%"],
            avoidLabelOverlap: true,
            itemStyle: { borderColor: "#0a0c10", borderWidth: 2 },
            label: { color: CHART_LABEL, formatter: "{b}\n{d}%", fontSize: 11 },
            labelLine: { lineStyle: { color: "#3a4452" } },
            data,
          },
        ],
      }}
    />
  );
}
