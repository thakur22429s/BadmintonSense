// CS 535 Project Progress Slides — Badminton-Sense
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Abhay Singh Thakur";
pres.title = "Badminton-Sense Progress";

// Palette: Ocean Gradient
const NAVY = "21295C";
const DEEP = "065A82";
const TEAL = "1C7293";
const ICE  = "CADCFC";
const CREAM = "F5F5F5";
const WHITE = "FFFFFF";
const ACCENT = "F96167";

const FIG = "results/figures";

// ============================================================
// SLIDE 1 — Title + Objective
// ============================================================
const s1 = pres.addSlide();
s1.background = { color: NAVY };

// Top accent strip
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 13.3, h: 0.35, fill: { color: TEAL }, line: { color: TEAL, width: 0 },
});

// Title
s1.addText("Badminton-Sense", {
  x: 0.7, y: 1.1, w: 12, h: 1.0,
  fontSize: 54, bold: true, color: WHITE, fontFace: "Georgia", margin: 0,
});

// Subtitle
s1.addText("Stroke Classification from Monocular Badminton Video", {
  x: 0.7, y: 2.05, w: 12, h: 0.6,
  fontSize: 24, color: ICE, fontFace: "Calibri", italic: true, margin: 0,
});

// Pipeline arrow text
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.7, y: 2.95, w: 11.9, h: 0.85,
  fill: { color: DEEP }, line: { color: DEEP, width: 0 },
});
s1.addText("BWF Video  →  Clip Segments  →  MediaPipe Pose  →  Normalize  →  LSTM / Transformer  →  Stroke Class", {
  x: 0.7, y: 2.95, w: 11.9, h: 0.85,
  fontSize: 16, color: WHITE, bold: true, align: "center", valign: "middle", fontFace: "Consolas", margin: 0,
});

// Two cards: Project info + Objective
// Left card — Project info
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.7, y: 4.1, w: 5.85, h: 2.85,
  fill: { color: WHITE }, line: { color: TEAL, width: 0 },
});
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.7, y: 4.1, w: 0.12, h: 2.85, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});
s1.addText("Project", {
  x: 1.0, y: 4.25, w: 5.4, h: 0.45,
  fontSize: 14, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 4, margin: 0,
});
s1.addText([
  { text: "Course: ", options: { bold: true, color: NAVY } },
  { text: "CS 535 — Pattern Recognition", options: { color: NAVY, breakLine: true } },
  { text: "Institution: ", options: { bold: true, color: NAVY } },
  { text: "Rutgers University", options: { color: NAVY, breakLine: true } },
  { text: "Type: ", options: { bold: true, color: NAVY } },
  { text: "Individual Project", options: { color: NAVY, breakLine: true } },
  { text: "Participant: ", options: { bold: true, color: NAVY } },
  { text: "Abhay Singh Thakur", options: { color: NAVY } },
], {
  x: 1.0, y: 4.75, w: 5.4, h: 2.1,
  fontSize: 16, fontFace: "Calibri", paraSpaceAfter: 6, margin: 0,
});

// Right card — Objective
s1.addShape(pres.shapes.RECTANGLE, {
  x: 6.75, y: 4.1, w: 5.85, h: 2.85,
  fill: { color: WHITE }, line: { color: TEAL, width: 0 },
});
s1.addShape(pres.shapes.RECTANGLE, {
  x: 6.75, y: 4.1, w: 0.12, h: 2.85, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});
s1.addText("Objective", {
  x: 7.05, y: 4.25, w: 5.4, h: 0.45,
  fontSize: 14, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 4, margin: 0,
});
s1.addText(
  "End-to-end pipeline that takes raw BWF broadcast video and classifies individual stroke types (smash, drop, clear, serve, etc.) using 2D pose estimation as the intermediate representation. Compares a BiLSTM baseline against a Spatial-Temporal Transformer on the ShuttleSet benchmark (36,492 strokes / 44 matches).",
  {
    x: 7.05, y: 4.75, w: 5.4, h: 2.1,
    fontSize: 14, color: NAVY, fontFace: "Calibri", margin: 0,
  }
);

// Footer
s1.addText("CS 535 — Progress Update — May 2026", {
  x: 0.5, y: 7.05, w: 12.3, h: 0.35,
  fontSize: 11, color: ICE, fontFace: "Calibri", align: "center", italic: true, margin: 0,
});

// ============================================================
// SLIDE 2 — Milestones & Results
// ============================================================
const s2 = pres.addSlide();
s2.background = { color: CREAM };

// Title bar
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 13.3, h: 1.0, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 1.0, w: 13.3, h: 0.06, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});
s2.addText("Milestones & Results", {
  x: 0.5, y: 0.15, w: 12.3, h: 0.7,
  fontSize: 32, bold: true, color: WHITE, fontFace: "Georgia", valign: "middle", margin: 0,
});
s2.addText("7-match run • 3,639 valid pose sequences • Heavy MediaPipe pose • 7-class taxonomy • CPU training", {
  x: 0.5, y: 0.55, w: 12.3, h: 0.4,
  fontSize: 13, color: ICE, fontFace: "Calibri", italic: true, valign: "middle", align: "right", margin: 0,
});

// Left column — milestones
s2.addText("ACCOMPLISHED", {
  x: 0.5, y: 1.3, w: 5.0, h: 0.4,
  fontSize: 14, bold: true, color: TEAL, fontFace: "Calibri", charSpacing: 4, margin: 0,
});
s2.addText([
  { text: "Design & Specification", options: { bold: true, color: NAVY, breakLine: true, fontSize: 14 } },
  { text: "10-class taxonomy mapped from 19 ShuttleSet Chinese stroke labels; full 7-step pipeline spec.", options: { color: "475569", fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Implementation", options: { bold: true, color: NAVY, breakLine: true, fontSize: 14 } },
  { text: "BiLSTM (527K params) + Spatial-Temporal Transformer (841K params); MediaPipe Tasks API; hip-center normalization; label smoothing CE; stratified split with small-sample fallback.", options: { color: "475569", fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Validation", options: { bold: true, color: NAVY, breakLine: true, fontSize: 14 } },
  { text: "End-to-end run on 7 matches (7,350 clips → 3,639 valid sequences). 7-class taxonomy (merged pose-indistinguishable classes). LSTM converges; Transformer data-starved.", options: { color: "475569", fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Evaluation Suite", options: { bold: true, color: NAVY, breakLine: true, fontSize: 14 } },
  { text: "Confusion matrices, training curves, per-class F1, t-SNE — generated for both models.", options: { color: "475569", fontSize: 11 } },
], {
  x: 0.5, y: 1.7, w: 5.0, h: 4.0,
  fontFace: "Calibri", margin: 0, paraSpaceAfter: 2,
});

s2.addText("REMAINING", {
  x: 0.5, y: 5.85, w: 5.0, h: 0.35,
  fontSize: 14, bold: true, color: ACCENT, fontFace: "Calibri", charSpacing: 4, margin: 0,
});
s2.addText([
  { text: "•  Scale to full 44 matches (36,492 strokes)", options: { color: NAVY, breakLine: true } },
  { text: "•  Heavy pose model (in progress, overnight run)", options: { color: NAVY, breakLine: true } },
  { text: "•  Streamlit demo for live inference", options: { color: NAVY, breakLine: true } },
  { text: "•  Final write-up and analysis", options: { color: NAVY } },
], {
  x: 0.5, y: 6.2, w: 5.0, h: 1.1,
  fontSize: 11, fontFace: "Calibri", margin: 0,
});

// Right column — results card
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.95, y: 1.3, w: 6.85, h: 6.0,
  fill: { color: WHITE }, line: { color: ICE, width: 1 },
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 5.95, y: 1.3, w: 6.85, h: 0.5, fill: { color: DEEP }, line: { color: DEEP, width: 0 },
});
s2.addText("Test-Set Performance (3-match subset)", {
  x: 6.1, y: 1.3, w: 6.55, h: 0.5,
  fontSize: 14, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0,
});

// Stat callouts row
s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.15, y: 2.0, w: 3.15, h: 1.55, fill: { color: ICE }, line: { color: ICE, width: 0 },
});
s2.addText("BiLSTM", {
  x: 6.15, y: 2.05, w: 3.15, h: 0.35, fontSize: 12, bold: true, color: TEAL, fontFace: "Calibri", align: "center", charSpacing: 4, margin: 0,
});
s2.addText("0.213", {
  x: 6.15, y: 2.4, w: 3.15, h: 0.7, fontSize: 48, bold: true, color: NAVY, fontFace: "Georgia", align: "center", margin: 0,
});
s2.addText("macro-F1   |   25.3% acc", {
  x: 6.15, y: 3.15, w: 3.15, h: 0.3, fontSize: 11, color: NAVY, fontFace: "Calibri", align: "center", margin: 0,
});

s2.addShape(pres.shapes.RECTANGLE, {
  x: 9.5, y: 2.0, w: 3.15, h: 1.55, fill: { color: ICE }, line: { color: ICE, width: 0 },
});
s2.addText("Transformer", {
  x: 9.5, y: 2.05, w: 3.15, h: 0.35, fontSize: 12, bold: true, color: TEAL, fontFace: "Calibri", align: "center", charSpacing: 4, margin: 0,
});
s2.addText("0.068", {
  x: 9.5, y: 2.4, w: 3.15, h: 0.7, fontSize: 48, bold: true, color: NAVY, fontFace: "Georgia", align: "center", margin: 0,
});
s2.addText("collapsed   |   31% (majority)", {
  x: 9.5, y: 3.15, w: 3.15, h: 0.3, fontSize: 11, color: NAVY, fontFace: "Calibri", align: "center", margin: 0,
});

// Confusion matrix image
s2.addImage({
  path: `${FIG}/confusion_matrix_lstm.png`,
  x: 6.15, y: 3.7, w: 3.2, h: 2.6, sizing: { type: "contain", w: 3.2, h: 2.6 },
});
s2.addImage({
  path: `${FIG}/training_curves_lstm.png`,
  x: 9.45, y: 3.7, w: 3.25, h: 2.6, sizing: { type: "contain", w: 3.25, h: 2.6 },
});

// Findings strip
s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.15, y: 6.4, w: 6.5, h: 0.85, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
});
s2.addText([
  { text: "KEY FINDING — ", options: { bold: true, color: ACCENT, charSpacing: 3 } },
  { text: "BiLSTM learns all 7 classes (+53% F1 vs baseline). Transformer remains data-hungry — collapses to majority-class on 2,547 train samples despite multiple config attempts.", options: { color: WHITE } },
], {
  x: 6.3, y: 6.45, w: 6.2, h: 0.75, fontSize: 12, fontFace: "Calibri", valign: "middle", margin: 0,
});

// ============================================================
// SLIDE 3 — Changes from Original Proposal
// ============================================================
const s3 = pres.addSlide();
s3.background = { color: CREAM };

// Title bar
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 13.3, h: 1.0, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
});
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 1.0, w: 13.3, h: 0.06, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
});
s3.addText("Changes from Original Proposal", {
  x: 0.5, y: 0.15, w: 12.3, h: 0.7,
  fontSize: 32, bold: true, color: WHITE, fontFace: "Georgia", valign: "middle", margin: 0,
});
s3.addText("Pragmatic adjustments driven by API breakage, hardware limits, and dataset shape", {
  x: 0.5, y: 0.55, w: 12.3, h: 0.4,
  fontSize: 13, color: ICE, fontFace: "Calibri", italic: true, valign: "middle", align: "right", margin: 0,
});

// 5 change cards — 2 rows
const cards = [
  { num: "01", title: "MediaPipe API Migration", from: "mp.solutions.pose", to: "Tasks API + PoseLandmarker", why: "Legacy API removed in v0.10.33 — rewrote pose extractor to use new IMAGE-mode landmarker, reusable across clips." },
  { num: "02", title: "Subset Run, Not Full 44", from: "44 matches end-to-end", to: "7-match overnight run", why: "Full 44-match run = 30+ hrs on CPU. 7 matches = 3,639 valid sequences, sufficient for pipeline validation + meaningful results." },
  { num: "03", title: "Class Taxonomy Refinement", from: "10 classes (incl. Drive, Cross-court, Drop)", to: "7 classes (Drop merged into Overhead-Soft)", why: "Drop vs Clear pose-indistinguishable. Drive only 40 samples (untrainable). Cross-court always 0. New taxonomy is pose-discriminable." },
  { num: "04", title: "Class Weights + Augmentation", from: "Vanilla CE + light aug", to: "Inverse-sqrt class weights + stronger aug", why: "Class imbalance (Overhead-Soft 31%, Other 6%) hurt minority classes. Inverse-sqrt weighting + bigger aug ranges lifted minority-class F1." },
  { num: "05", title: "Transformer Underperforms", from: "Expected: SOTA on stroke recognition", to: "Reality: data-starved, LSTM wins", why: "Spatial-Temporal Transformer needs more data than 2.5K samples. Multiple LR/scheduler/size configs tried — all collapsed to majority class. LSTM is practical winner." },
];

const cardW = 4.05, cardH = 2.85, gapX = 0.15, gapY = 0.25;
const startX = 0.5, startY = 1.35;
cards.forEach((c, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const x = startX + col * (cardW + gapX);
  const y = startY + row * (cardH + gapY);

  // card body
  s3.addShape(pres.shapes.RECTANGLE, {
    x, y, w: cardW, h: cardH, fill: { color: WHITE }, line: { color: ICE, width: 1 },
  });
  // accent bar on left
  s3.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.1, h: cardH, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });

  // number
  s3.addText(c.num, {
    x: x + 0.25, y: y + 0.15, w: 0.7, h: 0.5,
    fontSize: 28, bold: true, color: TEAL, fontFace: "Georgia", margin: 0,
  });
  // title
  s3.addText(c.title, {
    x: x + 1.0, y: y + 0.2, w: cardW - 1.15, h: 0.5,
    fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0, valign: "middle",
  });

  // from -> to row
  s3.addText([
    { text: "FROM:  ", options: { bold: true, color: TEAL, fontSize: 10 } },
    { text: c.from, options: { color: "475569", fontSize: 11, breakLine: true } },
    { text: "TO:  ", options: { bold: true, color: ACCENT, fontSize: 10 } },
    { text: c.to, options: { color: NAVY, fontSize: 11, bold: true } },
  ], {
    x: x + 0.25, y: y + 0.85, w: cardW - 0.4, h: 1.0, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4,
  });

  // why
  s3.addShape(pres.shapes.RECTANGLE, {
    x: x + 0.25, y: y + 1.95, w: cardW - 0.4, h: 0.02, fill: { color: ICE }, line: { color: ICE, width: 0 },
  });
  s3.addText(c.why, {
    x: x + 0.25, y: y + 2.0, w: cardW - 0.4, h: 0.8, fontSize: 10, color: "475569", fontFace: "Calibri", italic: true, margin: 0,
  });
});

// Footer summary
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 7.1, w: 13.3, h: 0.4, fill: { color: NAVY }, line: { color: NAVY, width: 0 },
});
s3.addText("Core architecture (BiLSTM + Spatial-Temporal Transformer) and 10-class taxonomy unchanged from proposal.", {
  x: 0.5, y: 7.1, w: 12.3, h: 0.4,
  fontSize: 12, color: WHITE, fontFace: "Calibri", italic: true, align: "center", valign: "middle", margin: 0,
});

// Write
pres.writeFile({ fileName: "presentation.pptx" }).then((f) => {
  console.log("WROTE: " + f);
});
