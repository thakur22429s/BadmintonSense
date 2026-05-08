"use client";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ManifestEntry } from "@/lib/types";

export function MatchSelector({
  matches,
  value,
  onChange,
  ariaLabel = "Match",
}: {
  matches: ManifestEntry[];
  value: string;
  onChange: (slug: string) => void;
  ariaLabel?: string;
}) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v ?? "")}>
      <SelectTrigger aria-label={ariaLabel} className="w-full md:w-[520px]">
        <SelectValue placeholder="Select a match" />
      </SelectTrigger>
      <SelectContent>
        {matches.map((m) => (
          <SelectItem key={m.slug} value={m.slug}>
            {m.display}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
