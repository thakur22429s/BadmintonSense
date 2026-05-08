import { PageContainer } from "@/components/layout/PageContainer";
import { getManifest, getMatch } from "@/lib/data";
import { MatchAnalyticsClient } from "./MatchAnalyticsClient";

export const metadata = { title: "Match Analytics — Badminton-Sense" };

export default async function MatchAnalyticsPage() {
  const manifest = await getManifest();
  const matches = await Promise.all(manifest.map((m) => getMatch(m.slug)));
  return (
    <PageContainer>
      <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">Match analytics</h1>
      <p className="mt-4 text-muted-foreground max-w-2xl">
        Pick a match. The model has already predicted every stroke. Browse the tactical story or
        flip to model diagnostics to see where it succeeded and failed.
      </p>
      <MatchAnalyticsClient matches={matches} manifest={manifest} />
    </PageContainer>
  );
}
