"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { MatchSelector } from "@/components/match/MatchSelector";
import { StrokeDistributionDonut } from "@/components/charts/StrokeDistributionDonut";
import { PlayerStyleRadar } from "@/components/charts/PlayerStyleRadar";
import { StrokeTimelineArea } from "@/components/charts/StrokeTimelineArea";
import { ConfusionMatrixHeatmap } from "@/components/charts/ConfusionMatrixHeatmap";
import { PerClassF1Bar } from "@/components/charts/PerClassF1Bar";
import { ConfidenceHistogram } from "@/components/charts/ConfidenceHistogram";
import { pct } from "@/lib/format";
import type { ManifestEntry, MatchData, ModelType } from "@/lib/types";

const TIMELINE_BUCKET = 10;

export function MatchAnalyticsClient({ matches, manifest }: { matches: MatchData[]; manifest: ManifestEntry[] }) {
  const [matchSlug, setMatchSlug] = useState(manifest[0]?.slug ?? "");
  const [model, setModel] = useState<ModelType>("lstm");

  const match = matches.find((m) => m.slug === matchSlug) ?? matches[0];
  const block = match.models[model];

  const predicted = useMemo(() => {
    return match.strokes
      .map((s) => ({ stroke: s, p: block.byClipId[s.clipId] }))
      .filter((x): x is { stroke: typeof match.strokes[0]; p: NonNullable<typeof x.p> } => x.p !== undefined);
  }, [match, block]);

  const playerNames = useMemo(() => Array.from(new Set(match.strokes.map((s) => s.player))), [match]);
  const distByPlayer = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const p of playerNames) out[p] = new Array(match.classNames.length).fill(0);
    for (const x of predicted) {
      if (out[x.stroke.player]) out[x.stroke.player][x.p.predIdx] += 1;
    }
    return out;
  }, [predicted, playerNames, match.classNames]);

  const radarSeries = useMemo(() => {
    return playerNames.map((p) => ({
      name: match.strokes.find((s) => s.player === p)?.playerName ?? `Player ${p}`,
      values: distByPlayer[p],
    }));
  }, [playerNames, distByPlayer, match.strokes]);

  const timelineSeries = useMemo(() => {
    const ordered = [...predicted].sort(
      (a, b) =>
        a.stroke.setNum - b.stroke.setNum ||
        a.stroke.rallyNum - b.stroke.rallyNum ||
        a.stroke.ballNum - b.stroke.ballNum
    );
    const buckets = Math.max(1, Math.ceil(ordered.length / TIMELINE_BUCKET));
    const out = match.classNames.map((c) => ({ name: c, values: new Array(buckets).fill(0) }));
    ordered.forEach((x, idx) => {
      const b = Math.min(buckets - 1, Math.floor(idx / TIMELINE_BUCKET));
      out[x.p.predIdx].values[b] += 1;
    });
    const setBoundaries: number[] = [];
    let lastSet = -1;
    ordered.forEach((x, idx) => {
      if (x.stroke.setNum !== lastSet) {
        if (lastSet !== -1) setBoundaries.push(Math.floor(idx / TIMELINE_BUCKET) + 1);
        lastSet = x.stroke.setNum;
      }
    });
    return { series: out, setBoundaries };
  }, [predicted, match.classNames]);

  const correct = predicted.filter((x) => x.p.correct).length;
  const acc = predicted.length ? correct / predicted.length : 0;
  const avgConf = predicted.length
    ? predicted.reduce((a, b) => a + b.p.probs[b.p.predIdx], 0) / predicted.length
    : 0;

  return (
    <div className="mt-8 space-y-6">
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2">
          <span className="block text-sm text-muted-foreground mb-1">Match</span>
          <MatchSelector matches={manifest} value={matchSlug} onChange={setMatchSlug} />
        </div>
        <div>
          <span className="block text-sm text-muted-foreground mb-1">Model</span>
          <Tabs value={model} onValueChange={(v) => setModel(v as ModelType)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="lstm">BiLSTM</TabsTrigger>
              <TabsTrigger value="transformer">Transformer</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      <Card className="p-5">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
          <div>
            <span className="font-semibold">{match.winner}</span>{" "}
            <span className="text-muted-foreground">def.</span>{" "}
            <span className="font-semibold">{match.loser}</span>
          </div>
          <div className="mono text-sm text-muted-foreground">
            {match.tournament} · {match.year} · {match.round}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Strokes", value: `${match.strokes.length}` },
          { label: "Predicted", value: `${predicted.length}` },
          { label: "Match accuracy", value: pct(acc) },
          { label: "Avg confidence", value: pct(avgConf) },
        ].map((s) => (
          <Card key={s.label} className="p-4">
            <div className="mono text-xs uppercase tracking-wider text-muted-foreground">{s.label}</div>
            <div className="mt-1 text-2xl font-semibold">{s.value}</div>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="story">
        <TabsList>
          <TabsTrigger value="story">Match Story</TabsTrigger>
          <TabsTrigger value="diagnostics">Model Diagnostics</TabsTrigger>
        </TabsList>

        <TabsContent value="story" className="mt-6 space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {playerNames.map((p) => (
              <Card key={p} className="p-5">
                <StrokeDistributionDonut
                  title={match.strokes.find((s) => s.player === p)?.playerName ?? `Player ${p}`}
                  classNames={match.classNames}
                  counts={distByPlayer[p]}
                />
              </Card>
            ))}
          </div>

          <Card className="p-5">
            <h3 className="font-semibold mb-3">Player style radar</h3>
            <PlayerStyleRadar classNames={match.classNames} players={radarSeries} />
          </Card>

          <Card className="p-5">
            <h3 className="font-semibold mb-3">Stroke timeline (chronological, bucketed by {TIMELINE_BUCKET} strokes)</h3>
            <StrokeTimelineArea classNames={match.classNames} series={timelineSeries.series} setBoundaries={timelineSeries.setBoundaries} />
          </Card>
        </TabsContent>

        <TabsContent value="diagnostics" className="mt-6 space-y-6">
          <Card className="p-5">
            <h3 className="font-semibold mb-3">Confusion matrix</h3>
            <ConfusionMatrixHeatmap classNames={match.classNames} matrix={block.confusion} />
          </Card>
          <Card className="p-5">
            <h3 className="font-semibold mb-3">Per-class F1 (this match)</h3>
            <PerClassF1Bar
              classNames={match.classNames}
              series={[{ name: model === "lstm" ? "BiLSTM" : "Transformer", values: match.classNames.map((c) => block.perClassF1[c] ?? 0) }]}
            />
          </Card>
          <Card className="p-5">
            <h3 className="font-semibold mb-3">Confidence distribution (by correctness)</h3>
            <ConfidenceHistogram
              predictions={predicted.map((x) => ({ confidence: x.p.probs[x.p.predIdx], correct: x.p.correct }))}
            />
          </Card>

          <Card className="p-5">
            <h3 className="font-semibold mb-3">Most confident predictions</h3>
            <div className="space-y-2 mono text-sm">
              {[...predicted]
                .sort((a, b) => b.p.probs[b.p.predIdx] - a.p.probs[a.p.predIdx])
                .slice(0, 8)
                .map((x) => (
                  <div key={x.stroke.clipId} className="flex flex-wrap gap-2 items-center text-muted-foreground">
                    <Badge variant={x.p.correct ? "default" : "destructive"}>
                      {x.p.correct ? "OK" : "MISS"}
                    </Badge>
                    <span className="text-foreground font-medium">{match.classNames[x.p.predIdx]}</span>
                    <span>· {pct(x.p.probs[x.p.predIdx])}</span>
                    <span>· truth {x.stroke.truthClass}</span>
                    <span>· {x.stroke.playerName}</span>
                    <span>· Set {x.stroke.setNum} R{x.stroke.rallyNum} S{x.stroke.ballNum.toFixed(0)}</span>
                  </div>
                ))}
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
