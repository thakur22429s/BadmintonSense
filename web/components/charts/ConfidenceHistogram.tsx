"use client";
import { EChart } from "./EChart";
import { CHART_LABEL, CHART_TOOLTIP_BG, CHART_GRID, CHART_AXIS, CHART_PALETTE } from "@/lib/styles";

export function ConfidenceHistogram({ predictions }: { predictions: { confidence: number; correct: boolean }[] }) {
  const bins = 10;
  const correct = new Array(bins).fill(0);
  const wrong = new Array(bins).fill(0);
  for (const p of predictions) {
    const idx = Math.min(bins - 1, Math.floor(p.confidence * bins));
    if (p.correct) correct[idx] += 1;
    else wrong[idx] += 1;
  }
  const labels = Array.from({ length: bins }, (_, i) => `${(i * 10).toFixed(0)}–${((i + 1) * 10).toFixed(0)}%`);

  return (
    <EChart
      height={300}
      option={{
        tooltip: { trigger: "axis", backgroundColor: CHART_TOOLTIP_BG, borderColor: "#2a323e", textStyle: { color: CHART_LABEL } },
        legend: { data: ["Correct", "Wrong"], textStyle: { color: CHART_LABEL }, top: 0 },
        grid: { left: 50, right: 20, top: 40, bottom: 50 },
        xAxis: {
          type: "category",
          data: labels,
          axisLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS, fontSize: 10, rotate: 30 },
        },
        yAxis: {
          type: "value",
          axisLine: { lineStyle: { color: CHART_GRID } },
          splitLine: { lineStyle: { color: CHART_GRID } },
          axisLabel: { color: CHART_AXIS },
        },
        color: [CHART_PALETTE[0], CHART_PALETTE[1]],
        series: [
          { name: "Correct", type: "bar", stack: "x", data: correct, itemStyle: { borderRadius: [2, 2, 0, 0] } },
          { name: "Wrong", type: "bar", stack: "x", data: wrong, itemStyle: { borderRadius: [2, 2, 0, 0] } },
        ],
      }}
    />
  );
}
