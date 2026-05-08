import { PageContainer } from "@/components/layout/PageContainer";
import { Card } from "@/components/ui/card";

export const metadata = { title: "Methodology — Badminton-Sense" };

export default function MethodologyPage() {
  return (
    <PageContainer>
      <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">Methodology</h1>
      <p className="mt-4 text-muted-foreground max-w-2xl leading-relaxed">
        Seven steps from BWF broadcast video to a labeled stroke prediction. Every choice is
        documented; every change from the original proposal is on the record.
      </p>

      <section className="mt-12 grid gap-6">
        {[
          {
            n: "01",
            title: "Source — ShuttleSet (KDD 2023)",
            body:
              "36,492 annotated strokes across 44 BWF professional matches; 7-match subset used for this work (3,639 valid pose sequences after filtering).",
          },
          {
            n: "02",
            title: "Clip extraction",
            body:
              "yt-dlp + ffmpeg cut each annotated stroke into a ~1.5s clip (1.0s pre-hit + 0.5s post-hit at 30fps).",
          },
          {
            n: "03",
            title: "Pose extraction",
            body:
              "MediaPipe Tasks API (PoseLandmarker, Heavy model). 15 of 33 landmarks selected; IMAGE mode lets the model be reused across clips. Min detection rate 0.5.",
          },
          {
            n: "04",
            title: "Normalization",
            body:
              "Hip-center translation + torso-length scaling. Removes camera distance and player position bias before any temporal modeling.",
          },
          {
            n: "05",
            title: "Resampling",
            body:
              "Every clip resampled to a fixed 30 frames so the temporal model always sees the same input shape (30, 15, 3).",
          },
          {
            n: "06",
            title: "Classification",
            body:
              "Two architectures trained from scratch — a 2-layer BiLSTM (~527K params) and a Spatial-Temporal Transformer (~841K params). Label-smoothing CE, gradient clipping, early stopping.",
          },
          {
            n: "07",
            title: "Evaluation",
            body:
              "70/15/15 stratified split; macro-F1 as primary metric. Confusion matrices, training curves, and t-SNE embeddings exported for diagnostics.",
          },
        ].map((s) => (
          <Card key={s.n} className="p-6 flex gap-6">
            <span className="mono text-2xl text-primary">{s.n}</span>
            <div>
              <h2 className="text-xl font-semibold">{s.title}</h2>
              <p className="mt-2 text-muted-foreground leading-relaxed">{s.body}</p>
            </div>
          </Card>
        ))}
      </section>

      <section className="mt-16 grid md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold">BiLSTM Baseline</h2>
          <ul className="mt-4 space-y-2 text-sm mono text-muted-foreground">
            <li>2 × Bidirectional LSTM, hidden=128</li>
            <li>Input: (30, 45) — 15 keypoints × 3 coords flattened</li>
            <li>FC: 256 → 7 classes</li>
            <li>~527K parameters</li>
            <li>Adam, LR=1e-3, ReduceLROnPlateau</li>
          </ul>
        </Card>
        <Card className="p-6">
          <h2 className="text-xl font-semibold">Spatial-Temporal Transformer</h2>
          <ul className="mt-4 space-y-2 text-sm mono text-muted-foreground">
            <li>Spatial block — embed=64, 2 heads × 2 layers</li>
            <li>Temporal block — embed=128, 4 heads × 3 layers</li>
            <li>Per-keypoint embedding — 32 dims</li>
            <li>~841K parameters</li>
            <li>AdamW, LR=2e-4, cosine schedule + 5 warmup epochs</li>
          </ul>
        </Card>
      </section>
    </PageContainer>
  );
}
