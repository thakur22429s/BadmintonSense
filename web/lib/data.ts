import { promises as fs } from "fs";
import path from "path";
import type { AggregateData, ManifestEntry, MatchData, ResultsData } from "./types";

const DATA_ROOT = path.join(process.cwd(), "public", "data");

async function readJson<T>(rel: string): Promise<T> {
  const buf = await fs.readFile(path.join(DATA_ROOT, rel), "utf8");
  return JSON.parse(buf) as T;
}

export async function getManifest(): Promise<ManifestEntry[]> {
  return readJson<ManifestEntry[]>("manifest.json");
}

export async function getAggregate(): Promise<AggregateData> {
  return readJson<AggregateData>("aggregate.json");
}

export async function getResults(): Promise<ResultsData> {
  return readJson<ResultsData>("results.json");
}

export async function getMatch(slug: string): Promise<MatchData> {
  return readJson<MatchData>(`matches/${slug}.json`);
}
