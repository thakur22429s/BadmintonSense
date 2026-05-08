"use client";
import { EChart } from "./EChart";
import { CHART_LABEL, CHART_TOOLTIP_BG, CHART_AXIS } from "@/lib/styles";

export function ConfusionMatrixHeatmap({
  classNames,
  matrix,
}: {
  classNames: string[];
  matrix: number[][];
}) {
  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const data: [number, number, number, number][] = [];
  let maxNorm = 0;
  for (let t = 0; t < classNames.length; t++) {
    for (let p = 0; p < classNames.length; p++) {
      const norm = rowSums[t] ? matrix[t][p] / rowSums[t] : 0;
      data.push([p, t, +norm.toFixed(3), matrix[t][p]]);
      if (norm > maxNorm) maxNorm = norm;
    }
  }
  return (
    <EChart
      height={Math.max(320, classNames.length * 42)}
      option={{
        tooltip: {
          backgroundColor: CHART_TOOLTIP_BG,
          borderColor: "#2a323e",
          textStyle: { color: CHART_LABEL },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter: (p: any) => {
            const [px, ty, norm, cnt] = p.value;
            return `Truth: <b>${classNames[ty]}</b><br/>Predicted: <b>${classNames[px]}</b><br/>${cnt} strokes (${(norm * 100).toFixed(1)}%)`;
          },
        },
        grid: { left: 110, right: 30, top: 30, bottom: 80 },
        xAxis: {
          type: "category",
          data: classNames,
          axisLabel: { color: CHART_AXIS, rotate: 35, fontSize: 11 },
          splitArea: { show: true },
          name: "Predicted",
          nameLocation: "middle",
          nameGap: 60,
          nameTextStyle: { color: CHART_LABEL },
        },
        yAxis: {
          type: "category",
          data: classNames,
          axisLabel: { color: CHART_AXIS, fontSize: 11 },
          splitArea: { show: true },
          name: "Truth",
          nameLocation: "middle",
          nameGap: 90,
          nameTextStyle: { color: CHART_LABEL },
        },
        visualMap: {
          min: 0,
          max: Math.max(0.05, maxNorm),
          calculable: false,
          orient: "vertical",
          right: 0,
          top: "middle",
          textStyle: { color: CHART_LABEL },
          inRange: { color: ["#0d1318", "#1c4f3a", "#56b275", "#a6df8c"] },
        },
        series: [
          {
            type: "heatmap",
            data,
            label: {
              show: true,
              color: "#ffffff",
              fontSize: 11,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter: (p: any) => (p.value[3] > 0 ? `${p.value[3]}` : ""),
            },
            itemStyle: { borderColor: "#0a0c10", borderWidth: 1 },
          },
        ],
      }}
    />
  );
}
