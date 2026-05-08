"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { MatchSelector } from "@/components/match/MatchSelector";
import { StrokeSelector } from "@/components/match/StrokeSelector";
import { ClipPlayer } from "@/components/viewers/ClipPlayer";
import { PoseSkeleton } from "@/components/viewers/PoseSkeleton";
import { pct } from "@/lib/format";
import type { ManifestEntry, MatchData, ModelType } from "@/lib/types";

export function SingleClipClient({ matches, manifest }: { matches: MatchData[]; manifest: ManifestEntry[] }) {
  const [matchSlug, setMatchSlug] = useState(manifest[0]?.slug ?? "");
  const [model, setModel] = useState<ModelType>("lstm");

  const match = matches.find((m) => m.slug === matchSlug) ?? matches[0];
  const browsableStrokes = useMemo(
    () => match.strokes.filter((s) => s.keypoints && s.clipUrl),
    [match]
  );
  const [clipId, setClipId] = useState(browsableStrokes[0]?.clipId ?? "");

  const stroke = browsableStrokes.find((s) => s.clipId === clipId) ?? browsableStrokes[0];
  const pred = stroke ? match.models[model].byClipId[stroke.clipId] : undefined;
  const top5 = useMemo(() => {
    if (!pred) return [] as { name: string; p: number }[];
    return match.classNames
      .map((name, i) => ({ name, p: pred.probs[i] }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);
  }, [pred, match.classNames]);

  const [frameIdx, setFrameIdx] = useState(15);

  return (
    <div className="mt-8 space-y-6">
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2">
          <label className="block text-sm text-muted-foreground mb-1">Match</label>
          <MatchSelector
            matches={manifest}
            value={matchSlug}
            onChange={(s) => {
              setMatchSlug(s);
              const next = matches.find((m) => m.slug === s);
              setClipId(next?.strokes.find((x) => x.keypoints && x.clipUrl)?.clipId ?? "");
            }}
          />
        </div>
        <div>
          <label className="block text-sm text-muted-foreground mb-1">Model</label>
          <Tabs value={model} onValueChange={(v) => setModel(v as ModelType)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="lstm">BiLSTM</TabsTrigger>
              <TabsTrigger value="transformer">Transformer</TabsTrigger>
            </TabsList>
            <TabsContent value="lstm" />
            <TabsContent value="transformer" />
          </Tabs>
        </div>
      </div>

      <div>
        <label className="block text-sm text-muted-foreground mb-1">Stroke</label>
        <StrokeSelector strokes={browsableStrokes} value={clipId} onChange={setClipId} />
      </div>

      {!stroke ? (
        <Card className="p-6">No browsable strokes in this match.</Card>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          <Card className="p-4">
            <h3 className="font-semibold mb-3">Clip</h3>
            {stroke.clipUrl ? <ClipPlayer src={stroke.clipUrl} /> : <p className="text-muted-foreground">Clip not available.</p>}
            <div className="mt-4">
              <label className="block text-sm text-muted-foreground mb-2">Pose at frame {frameIdx + 1}/30</label>
              {stroke.keypoints && <PoseSkeleton keypoints={stroke.keypoints} frameIdx={frameIdx} />}
              <input
                type="range"
                min={0}
                max={(stroke.keypoints?.length ?? 30) - 1}
                value={frameIdx}
                onChange={(e) => setFrameIdx(Number(e.target.value))}
                className="w-full mt-3"
              />
            </div>
          </Card>

          <div className="space-y-6">
            <Card className="p-6">
              <div className="mono text-xs uppercase tracking-wider text-primary">Predicted</div>
              <div className="mt-1 text-3xl font-semibold">
                {pred ? match.classNames[pred.predIdx] : "—"}
              </div>
              <div className="mt-1 text-sm text-muted-foreground">
                {pred ? pct(pred.probs[pred.predIdx]) : "—"} confidence
              </div>
              <div className="mt-3 flex gap-2 items-center">
                <span className="text-sm text-muted-foreground">Truth:</span>
                <span className="font-medium">{stroke.truthClass}</span>
                {pred && (
                  <Badge variant={pred.correct ? "default" : "destructive"}>
                    {pred.correct ? "correct" : "wrong"}
                  </Badge>
                )}
              </div>
              <div className="mt-6 grid grid-cols-3 gap-3 text-sm">
                <div><div className="mono text-xs text-muted-foreground">Frames</div><div className="font-medium">{stroke.keypoints?.length ?? "—"}</div></div>
                <div><div className="mono text-xs text-muted-foreground">Detection</div><div className="font-medium">{pct(stroke.detectionRate)}</div></div>
                <div><div className="mono text-xs text-muted-foreground">Player</div><div className="font-medium">{stroke.playerName}</div></div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="font-semibold mb-4">Top 5 probabilities</h3>
              <div className="space-y-3">
                {top5.map((row, i) => (
                  <div key={row.name}>
                    <div className="flex justify-between text-sm">
                      <span className={i === 0 ? "font-medium" : "text-muted-foreground"}>{row.name}</span>
                      <span className="mono">{pct(row.p)}</span>
                    </div>
                    <div className="mt-1 h-2 rounded bg-secondary overflow-hidden">
                      <div
                        className={i === 0 ? "h-full bg-primary" : "h-full bg-muted-foreground/40"}
                        style={{ width: `${Math.max(2, row.p * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
