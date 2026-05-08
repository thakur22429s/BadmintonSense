import Link from "next/link";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { getAggregate, getResults } from "@/lib/data";
import { compactInt, pct } from "@/lib/format";

export default async function HomePage() {
  const [agg, results] = await Promise.all([getAggregate(), getResults()]);

  const stats = [
    { label: "Matches analyzed", value: `${agg.nMatches}` },
    { label: "Strokes processed", value: compactInt(agg.nStrokes) },
    { label: "Best macro-F1", value: results.headline.lstm.macroF1.toFixed(3) },
    { label: "Stroke classes", value: `${agg.classNames.length}` },
  ];

  const pipeline = ["BWF Video", "Clip Stroke", "Pose Extract", "Normalize", "Classify"];

  return (
    <PageContainer>
      <section className="pt-6 pb-14 md:pt-12 md:pb-20">
        <p className="mono text-xs uppercase tracking-[0.2em] text-primary mb-3">CS 535 · Project</p>
        <h1 className="text-5xl md:text-7xl font-semibold tracking-tight leading-[1.05] max-w-4xl">
          Stroke classification from monocular badminton video.
        </h1>
        <p className="mt-6 text-lg text-muted-foreground max-w-2xl leading-relaxed">
          A pose-based deep learning pipeline that takes raw BWF broadcast footage, extracts 2D pose
          sequences with MediaPipe, and classifies stroke types via BiLSTM and Spatial-Temporal
          Transformer architectures.
        </p>
        <div className="mt-8 flex gap-3">
          <Link href="/results" className={buttonVariants({ size: "lg" })}>See the results</Link>
          <Link href="/match-analytics" className={buttonVariants({ size: "lg", variant: "secondary" })}>Match analytics</Link>
        </div>
      </section>

      <section className="py-10 border-y border-border">
        <div className="flex flex-wrap gap-3 justify-between items-center">
          {pipeline.map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <span className="mono text-xs text-muted-foreground">0{i + 1}</span>
              <span className="text-sm">{step}</span>
              {i < pipeline.length - 1 && <span className="text-border">—</span>}
            </div>
          ))}
        </div>
      </section>

      <section className="py-14 grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((s) => (
          <Card key={s.label} className="p-5">
            <div className="mono text-xs uppercase tracking-wider text-muted-foreground">{s.label}</div>
            <div className="mt-2 text-3xl font-semibold tracking-tight">{s.value}</div>
          </Card>
        ))}
      </section>

      <section className="py-10 max-w-2xl">
        <h2 className="text-2xl font-semibold tracking-tight">The headline.</h2>
        <p className="mt-4 text-muted-foreground leading-relaxed">
          Through data scaling (3 → 7 matches), pose-model upgrade (lite → heavy), and class-taxonomy
          refinement (10 → 7 classes), the BiLSTM baseline improved by{" "}
          <span className="text-foreground font-medium">+53% F1 relative</span> ({results.progression[0].lstmF1.toFixed(3)} → {results.headline.lstm.macroF1.toFixed(3)}).
          The Spatial-Temporal Transformer collapsed under the data constraint — a textbook
          demonstration of why architecture choice cannot outrun training-set size.
        </p>
        <p className="mt-3 text-muted-foreground">Final BiLSTM accuracy: {pct(results.headline.lstm.accuracy)}.</p>
      </section>
    </PageContainer>
  );
}
