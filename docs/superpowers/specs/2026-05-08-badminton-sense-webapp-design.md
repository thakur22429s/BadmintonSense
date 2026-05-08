# Badminton-Sense Webapp Redesign — Design Spec

**Date:** 2026-05-08
**Author:** Abhay Singh Thakur
**Project:** Badminton-Sense (CS 535, Rutgers MS CS)
**Scope:** Replace Streamlit demo with a Next.js showcase site for Canvas submission. The final-presentation deck is a sibling deliverable and will be specified in a separate document.

---

## 1. Goal

Ship a single Vercel URL — submittable on Canvas — that lets the professor evaluate the project end-to-end without installing anything: read the methodology, browse the model results, watch live stroke-classification on real BWF clips, and explore match-level analytics. Visual quality should match a research-lab / pro-ML aesthetic (Distill / Anthropic / Linear).

## 2. Non-Goals

- **Live custom-clip upload.** All inference is pre-computed against ShuttleSet sample clips. Removing this feature is the explicit tradeoff for a single static URL with no Python backend. Future-work bullet in the deck will call out how to re-introduce upload via a FastAPI sidecar.
- **Re-training models.** Existing checkpoints (`models/best_lstm_stratified.pt`, `models/best_transformer_stratified.pt`) are reused as-is.
- **Mobile-first design.** Site should not break on mobile, but the target reading device is a laptop. Charts assume ≥768px width.
- **CMS / dynamic content.** All copy is hard-coded in MDX or TSX files.

## 3. Architecture (One-Glance)

```
Python pre-compute (one-shot)
  load_config + checkpoints
  for each sample clip:
    extract_poses_from_clip → preprocess_single → model.softmax
  emit JSON:
    public/data/matches/<match_name>.json   (per-match strokes + preds)
    public/data/aggregate.json              (cross-match summary stats)
    public/data/results.json                (per-class F1, confusion mats, training curves)
  copy MP4 sample clips → public/clips/<clip_id>.mp4   (size-bound subset)

Next.js 15 (App Router, static export-friendly)
  /                  landing
  /methodology       pipeline + architecture diagrams
  /results           model comparison (per-class F1, confusion, t-SNE, training curves)
  /single-clip       pose-overlay demo on sample clips
  /match-analytics   hybrid charts (Match Story / Model Diagnostics subtabs)
```

Site is fully static. No API routes, no server actions, no runtime Python.

## 4. Pages

### 4.1 `/` — Landing

- **Hero:** project title, one-line objective, headline metric ("Macro-F1 +53% relative through data scaling and class refinement"), CTA buttons → `/results`, `/match-analytics`.
- **Pipeline strip:** horizontal 5-step diagram (Video → Clip → Pose → Normalize → Classify) — pure SVG, animated on scroll.
- **At-a-glance stats:** 4 cards — Matches analyzed (7), Strokes processed (3,639), Best macro-F1 (0.213), Stroke classes (read from active checkpoint config — 7 in the best run, 10 in earlier runs; site shows whichever the loaded checkpoint expects).
- **Footer:** course, author, GitHub link.

### 4.2 `/methodology`

- **Pipeline detail:** ordered sections matching the Python pipeline. Code snippets in `pre`/`code` blocks (Shiki theme = One Dark Pro or Vesper for the dark-mode site, or GitHub Light if we go light).
- **Pose extraction:** MediaPipe Tasks API rationale (forced migration), Lite vs Heavy tradeoff, hip-center normalization formula, why 30-frame resampling.
- **Architectures:** two side-by-side panels.
  - BiLSTM: 2-layer hidden=128, ~527K params. Schematic SVG.
  - Spatial-Temporal Transformer: spatial block (embed=64, 2H/2L) + temporal block (embed=128, 4H/3L), ~841K params. Schematic SVG.
- **Training config table** (label smoothing, gradient clip, scheduler).
- **Dataset section:** ShuttleSet provenance, 19 → 10 class taxonomy table, class distribution donut.

### 4.3 `/results`

- **Headline numbers:** macro-F1 / weighted-F1 / accuracy for both models, side-by-side.
- **Per-class F1 grouped bar chart:** LSTM vs Transformer per class.
- **Progression chart:** line/bar showing F1 across the three runs (lite → heavy → 7-match-7-class).
- **Confusion matrices:** ECharts heatmap, both models, toggle.
- **Training curves:** ECharts line chart, loss + F1 over epochs, both models, toggle.
- **t-SNE embeddings:** static PNG render of the existing figures (not re-implemented as interactive — too much work for marginal value).
- **Findings list:** the 6 key findings from PROJECT_SUMMARY § 5.

### 4.4 `/single-clip`

- Replicates current Streamlit Tab 1 functionality. Hierarchy:
  1. **Match selector** (dropdown of 7 BWF matches, formatted "Winner def. Loser — Tournament (Year) — Round").
  2. **Stroke selector** (filtered list: "Set 2 · Rally 14 · Stroke 3 · Player A (smash)").
  3. **Model toggle** (LSTM / Transformer).
- **Display:**
  - Left column: HTML5 `<video>` of the clip.
  - Right column: predicted class card, top-5 probability bars, ground-truth comparison badge (✓/✗), 3 stat tiles (Frames / Detection rate / Model).
  - Bottom: pose-skeleton viewer. Frame slider scrubs through the clip; SVG-overlay skeleton on top of a still-frame extracted at build time (poster strip), OR live `<canvas>` overlay reading pre-computed keypoints JSON. Pick the canvas approach — keypoints are already in the per-clip JSON.

### 4.5 `/match-analytics`

- Match selector + model toggle at top, same as `/single-clip`.
- Match metadata header (winner / loser / tournament / year / round).
- 4 stat cards (Total strokes / Valid for model / Players / Avg detection).
- **Two subtabs:**

#### Match Story (subtab)
1. **Stroke distribution donut**, per player, side-by-side. Highlight on hover, sync legend.
2. **Player style radar** — one axis per class in the active class list (so 7 or 10 depending on checkpoint). Both players overlaid. Tooltip shows %.
3. **Stroke timeline stacked area** — x-axis = stroke index in match (or grouped by set with vertical separators), y-axis = stacked count by class. Lets the viewer see tactical shifts.
4. **Style verdict card** — keep the existing `player_style_profile` rule-based commentary, restyled. Top-3 stroke chips per player.

#### Model Diagnostics (subtab)
1. **Confusion matrix heatmap** (this match only).
2. **Per-class F1 bar chart** (this match only).
3. **Confidence histogram** — 10 bins, stacked by correct/wrong prediction. Calibration insight.
4. **Most-confident-predictions list** — top 8 with ✓/✗, clip metadata, link to `/single-clip?clip=<id>`.

## 5. Data Layer

### 5.1 Pre-compute script

- New file: `scripts/build_web_data.py`.
- Runs locally once. Loads config, both checkpoints, the merged `clip_metadata.csv` + `pose_metadata.csv`, the match info CSV, and the existing `.npy` pose files.
- For each (match, model) pair, runs `predict_from_pose` on every valid clip and writes `web/public/data/matches/<match_slug>.json`.
- Also emits:
  - `web/public/data/results.json` — per-class F1, confusion matrices, training-curve series for both models, sourced from existing `results/` artifacts.
  - `web/public/data/aggregate.json` — site-wide counters (n_matches, n_strokes, headline metrics).
  - `web/public/data/manifest.json` — list of matches with display labels and counts.
- Same script also copies sample MP4 clips into `web/public/clips/`, downsampling/cropping if total budget exceeds Vercel's reasonable static-asset size (target ≤200 MB total). If budget is exceeded, the script picks the top N most-confident clips per class per match for the demo.

### 5.2 JSON schemas (sketch)

```ts
// web/public/data/matches/<slug>.json
type MatchData = {
  slug: string;
  display: string;            // "Tai Tzu Ying def. Carolina Marin — French Open (2018) — F"
  winner: string; loser: string; tournament: string; year: number; round: string;
  classNames: string[];       // ordered class list (10 or 7 depending on run)
  models: {
    lstm: PredictionSet;
    transformer: PredictionSet;
  };
  strokes: {
    clipId: string;
    setNum: number; rallyNum: number; ballNum: number;
    player: "A" | "B";
    playerName: string;
    truthClass: string;
    keypoints?: number[][][];    // [T][K][3], optional, only for strokes shown on /single-clip
    detectionRate: number;
    clipUrl: string;             // /clips/<clipId>.mp4
    durationSec: number;
  }[];
};

type PredictionSet = {
  byClipId: Record<string, { predIdx: number; probs: number[]; correct: boolean }>;
  confusion: number[][];        // [trueClass][predClass]
  perClassF1: Record<string, number>;
};
```

### 5.3 Keypoints in JSON

Including `keypoints` for every stroke would balloon JSON. Strategy: include keypoints only for a curated subset (~50 per match — enough variety for `/single-clip` browsing). Rest is prediction-only. The selector in `/single-clip` filters to clips that have `keypoints` present, with a small "more clips" hint linking to the full match-analytics view.

## 6. UI / Design System

- **Mode:** dark slate (`#0b0f17` bg, `#e6ebf2` body) with optional light toggle (defer; ship dark-only).
- **Type:** IBM Plex Sans (UI/body), JetBrains Mono (data labels, code, numerals).
- **Accent:** badminton-court green `#3a8b51` for "correct"/positive cues; muted coral `#d96c5d` for "wrong"/error cues. Both desaturated for the lab look.
- **Layout:** generous whitespace, grid-based, 1080–1280px content width on desktop.
- **Components:** shadcn/ui primitives (`Card`, `Tabs`, `Select`, `Button`, `Badge`, `ScrollArea`, `Separator`, `Tooltip`).
- **Motion:** subtle. Framer Motion for chart-on-mount fades and section-on-scroll reveals only. No bouncy spring animations.
- **Iconography:** Lucide.

## 7. Tech Stack

- Next.js 15 (App Router, TypeScript, Turbopack dev).
- Tailwind v4.
- shadcn/ui (Radix primitives under the hood).
- ECharts via `echarts-for-react` for all charts.
- MDX for `/methodology` and `/results` long-form copy.
- Framer Motion for transitions.
- No state library — server components + URL params + minimal local `useState`.
- Lint/format: ESLint flat config + Prettier.
- Tests: skip (showcase site, deadline-bound). Verification = manual browser check across all 5 routes.

## 8. Repo Layout

```
BadmintonSense/
├── app/                           existing Streamlit (kept as fallback)
├── scripts/
│   └── build_web_data.py          new: pre-compute pipeline
├── web/                           new: Next.js project root
│   ├── app/
│   │   ├── page.tsx               landing
│   │   ├── methodology/page.mdx
│   │   ├── results/page.tsx
│   │   ├── single-clip/page.tsx
│   │   └── match-analytics/page.tsx
│   ├── components/
│   │   ├── charts/                ECharts wrappers (Donut, Radar, Heatmap, etc.)
│   │   ├── layout/                Nav, Footer, PageContainer
│   │   ├── viewers/               PoseSkeleton, ClipPlayer
│   │   └── ui/                    shadcn primitives
│   ├── lib/
│   │   ├── data.ts                JSON loaders (server-side fs reads)
│   │   └── format.ts              number/percentage helpers
│   ├── public/
│   │   ├── data/                  JSON output of build_web_data.py
│   │   └── clips/                 sampled MP4s
│   ├── tailwind.config.ts
│   ├── next.config.js
│   └── package.json
└── ...
```

## 9. Deployment

- Push `web/` to Vercel as a project subroot.
- Domain: default `vercel.app` subdomain (`badminton-sense.vercel.app` if available).
- Build command: `npm run build`. No env vars required.
- Static-asset budget: aim ≤250 MB. If clips push past, `build_web_data.py` selects a curated subset.
- Submission: Vercel URL pasted in Canvas; brief blurb in the assignment text noting that all inference is pre-computed and the upload feature is documented as future work in the slide deck.

## 10. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| Total clip MP4 size exceeds Vercel free static budget | `build_web_data.py` curates a subset; document in slides |
| ECharts bundle weight (~300KB gz) hurts LCP | Lazy-load chart components per route via `dynamic(() => import(...), { ssr: false })` |
| JSON files for matches with thousands of strokes get large | Drop unused fields; gzip/brotli served by Vercel automatically |
| Pose checkpoints loaded with `weights_only=False` is a security concern in general | Acceptable here — checkpoints are author-produced, never user-provided |
| Inference results differ slightly between Streamlit and Next.js because of float ordering | None — all predictions are produced once by `build_web_data.py` and frozen as JSON |

## 11. Out of Scope (handled by the sibling deck spec)

- The final-presentation `.pptx` content, slide order, and rendering pipeline. The deck must include a "Future Work" slide that calls out: live MP4 upload via FastAPI sidecar, full 44-match training run, GPU re-training with augmentation, deployment to a real domain.

## 12. Acceptance Criteria

- All 5 routes load without console errors on Chromium-current.
- Match selector shows all 7 ShuttleSet matches with real player names from `match.csv`.
- Both models toggleable on `/single-clip` and `/match-analytics`; predictions match what the Streamlit app produces (spot-check 5 random clips).
- Confusion-matrix heatmap, per-class F1 bars, training curves, donuts, radar, stacked-area timeline, and confidence histogram all render with real data and tooltip on hover.
- Pose-skeleton overlay scrubs through clip frames and aligns visually with the player.
- Lighthouse Performance ≥80, Accessibility ≥90 on `/`.
- Vercel deploy succeeds; URL is shareable to a fresh browser without auth.
