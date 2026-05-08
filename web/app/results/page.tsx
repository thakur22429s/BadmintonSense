import { PageContainer } from "@/components/layout/PageContainer";
import { Card } from "@/components/ui/card";
import { ConfusionMatrixHeatmap } from "@/components/charts/ConfusionMatrixHeatmap";
import { PerClassF1Bar } from "@/components/charts/PerClassF1Bar";
import { TrainingCurves } from "@/components/charts/TrainingCurves";
import { getAggregate, getResults } from "@/lib/data";
import { fmt, pct } from "@/lib/format";

export const metadata = { title: "Results — Badminton-Sense" };

export default async function ResultsPage() {
  const [agg, results] = await Promise.all([getAggregate(), getResults()]);
  const classNames = agg.classNames;
  const finalClassNames = Object.keys(results.perClassF1Final);

  return (
    <PageContainer>
      <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">Results</h1>
      <p className="mt-4 text-muted-foreground max-w-2xl leading-relaxed">
        Numbers from the final 7-match, 7-class run. The full per-class breakdown and the
        progression across runs are below.
      </p>

      <section className="mt-12 grid md:grid-cols-2 gap-6">
        {(["lstm", "transformer"] as const).map((m) => (
          <Card key={m} className="p-6">
            <div className="mono text-xs uppercase tracking-wider text-primary">{m}</div>
            <div className="mt-2 text-3xl font-semibold">{fmt(results.headline[m].macroF1)}</div>
            <div className="text-sm text-muted-foreground">macro-F1</div>
            <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
              <div><div className="mono text-muted-foreground">Weighted-F1</div><div className="font-medium">{fmt(results.headline[m].weightedF1)}</div></div>
              <div><div className="mono text-muted-foreground">Accuracy</div><div className="font-medium">{pct(results.headline[m].accuracy)}</div></div>
              <div><div className="mono text-muted-foreground">Best epoch</div><div className="font-medium">{results.headline[m].bestEpoch}</div></div>
            </div>
          </Card>
        ))}
      </section>

      <section className="mt-16">
        <h2 className="text-2xl font-semibold tracking-tight">Per-class F1 (final test set)</h2>
        <p className="mt-2 text-sm text-muted-foreground">Ground-truth from ShuttleSet annotations; predictions from each model on the held-out test split.</p>
        <Card className="mt-6 p-6">
          <PerClassF1Bar
            classNames={finalClassNames}
            series={[
              { name: "BiLSTM", values: finalClassNames.map((c) => results.perClassF1Final[c].lstm) },
              { name: "Transformer", values: finalClassNames.map((c) => results.perClassF1Final[c].transformer) },
            ]}
          />
        </Card>
      </section>

      <section className="mt-16 grid md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold">Confusion — BiLSTM</h2>
          <ConfusionMatrixHeatmap classNames={classNames} matrix={agg.models.lstm.confusion} />
        </Card>
        <Card className="p-6">
          <h2 className="text-xl font-semibold">Confusion — Transformer</h2>
          <ConfusionMatrixHeatmap classNames={classNames} matrix={agg.models.transformer.confusion} />
        </Card>
      </section>

      {(results.figures.training_curves_lstm || results.figures.training_curves_transformer) && (
        <section className="mt-16 grid md:grid-cols-2 gap-6">
          {results.figures.training_curves_lstm && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Training curves — BiLSTM</h3>
              <TrainingCurves src={results.figures.training_curves_lstm} alt="BiLSTM training curves" />
            </div>
          )}
          {results.figures.training_curves_transformer && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Training curves — Transformer</h3>
              <TrainingCurves src={results.figures.training_curves_transformer} alt="Transformer training curves" />
            </div>
          )}
        </section>
      )}

      {(results.figures.tsne_lstm || results.figures.tsne_transformer) && (
        <section className="mt-16 grid md:grid-cols-2 gap-6">
          {results.figures.tsne_lstm && (
            <div>
              <h3 className="text-lg font-semibold mb-3">t-SNE — BiLSTM embeddings</h3>
              <TrainingCurves src={results.figures.tsne_lstm} alt="BiLSTM t-SNE" />
            </div>
          )}
          {results.figures.tsne_transformer && (
            <div>
              <h3 className="text-lg font-semibold mb-3">t-SNE — Transformer embeddings</h3>
              <TrainingCurves src={results.figures.tsne_transformer} alt="Transformer t-SNE" />
            </div>
          )}
        </section>
      )}

      <section className="mt-16">
        <h2 className="text-2xl font-semibold tracking-tight">Findings</h2>
        <ul className="mt-6 space-y-3">
          {results.findings.map((f, i) => (
            <li key={i} className="flex gap-4">
              <span className="mono text-sm text-primary">{(i + 1).toString().padStart(2, "0")}</span>
              <span className="text-muted-foreground leading-relaxed">{f}</span>
            </li>
          ))}
        </ul>
      </section>
    </PageContainer>
  );
}
