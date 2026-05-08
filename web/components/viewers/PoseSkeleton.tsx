"use client";
import { useMemo } from "react";

const CONNECTIONS: [number, number][] = [
  [1, 3], [3, 5],
  [2, 4], [4, 6],
  [1, 2],
  [1, 11], [2, 12],
  [11, 12],
  [11, 13], [12, 14],
];

export function PoseSkeleton({
  keypoints,
  frameIdx,
  width = 720,
  height = 405,
}: {
  keypoints: number[][][];
  frameIdx: number;
  width?: number;
  height?: number;
}) {
  const frame = keypoints[Math.max(0, Math.min(keypoints.length - 1, frameIdx))];

  const { points, ok } = useMemo(() => {
    if (!frame) return { points: [], ok: false };
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of frame) {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        if (x < xMin) xMin = x; if (x > xMax) xMax = x;
        if (y < yMin) yMin = y; if (y > yMax) yMax = y;
      }
    }
    if (!Number.isFinite(xMin)) return { points: [], ok: false };
    const padX = (xMax - xMin) * 0.15 || 0.1;
    const padY = (yMax - yMin) * 0.15 || 0.1;
    const left = xMin - padX, right = xMax + padX;
    const top = yMin - padY, bottom = yMax + padY;
    const sx = width / (right - left);
    const sy = height / (bottom - top);
    const s = Math.min(sx, sy);
    const offX = (width - (right - left) * s) / 2 - left * s;
    const offY = (height - (bottom - top) * s) / 2 - top * s;
    return {
      ok: true,
      points: frame.map(([x, y]) =>
        Number.isFinite(x) && Number.isFinite(y) ? [x * s + offX, y * s + offY] : null
      ),
    };
  }, [frame, width, height]);

  if (!ok) {
    return <div className="grid place-items-center bg-card border border-border rounded-lg" style={{ width, height }}>
      <span className="text-sm text-muted-foreground">No pose detected on this frame</span>
    </div>;
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full bg-card border border-border rounded-lg">
      {CONNECTIONS.map(([a, b], i) => {
        const pa = points[a]; const pb = points[b];
        if (!pa || !pb) return null;
        return <line key={i} x1={pa[0]} y1={pa[1]} x2={pb[0]} y2={pb[1]} stroke="#56b275" strokeWidth={3} strokeLinecap="round" />;
      })}
      {points.map((p, i) => (
        p ? <circle key={i} cx={p[0]} cy={p[1]} r={5} fill="#fff" stroke="#56b275" strokeWidth={2} /> : null
      ))}
    </svg>
  );
}
