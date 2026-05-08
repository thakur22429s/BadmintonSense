"use client";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { Stroke } from "@/lib/types";

export function StrokeSelector({
  strokes,
  value,
  onChange,
}: {
  strokes: Stroke[];
  value: string;
  onChange: (clipId: string) => void;
}) {
  const formatLabel = (s: Stroke) =>
    `Set ${s.setNum} · Rally ${s.rallyNum} · Stroke ${s.ballNum.toFixed(0)} · Player ${s.player} · ${s.truthClass}`;

  return (
    <Select value={value} onValueChange={(v) => onChange(v ?? "")}>
      <SelectTrigger className="w-full md:w-[520px]">
        <SelectValue placeholder="Select a stroke" />
      </SelectTrigger>
      <SelectContent className="max-h-[420px]">
        {strokes.map((s) => (
          <SelectItem key={s.clipId} value={s.clipId}>
            {formatLabel(s)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
