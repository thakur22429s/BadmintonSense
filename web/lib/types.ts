export type ManifestEntry = {
  slug: string;
  rawName: string;
  display: string;
  winner: string;
  loser: string;
  tournament: string;
  round: string;
  year: number;
  totalStrokes: number;
  validStrokes: number;
};

export type StrokePrediction = {
  predIdx: number;
  probs: number[];
  correct: boolean;
};

export type ModelBlock = {
  byClipId: Record<string, StrokePrediction>;
  confusion: number[][];
  perClassF1: Record<string, number>;
};

export type Stroke = {
  clipId: string;
  setNum: number;
  rallyNum: number;
  ballNum: number;
  player: "A" | "B" | string;
  playerName: string;
  truthClass: string;
  truthIdx: number;
  detectionRate: number;
  clipUrl: string | null;
  keypoints: number[][][] | null;
};

export type MatchData = {
  slug: string;
  rawName: string;
  display: string;
  winner: string;
  loser: string;
  tournament: string;
  round: string;
  year: number;
  classNames: string[];
  models: { lstm: ModelBlock; transformer: ModelBlock };
  strokes: Stroke[];
};

export type AggregateData = {
  nMatches: number;
  nStrokes: number;
  nValid: number;
  classNames: string[];
  models: Record<
    "lstm" | "transformer",
    {
      macroF1: number;
      accuracy: number;
      confusion: number[][];
      perClassF1: Record<string, number>;
    }
  >;
};

export type ResultsData = {
  headline: Record<"lstm" | "transformer", { macroF1: number; weightedF1: number; accuracy: number; bestEpoch: number }>;
  progression: { label: string; lstmF1: number; lstmAcc: number }[];
  perClassF1Final: Record<string, { lstm: number; transformer: number; support: number }>;
  findings: string[];
  figures: Record<string, string>;
};

export type ModelType = "lstm" | "transformer";
