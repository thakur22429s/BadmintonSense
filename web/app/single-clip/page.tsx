import { PageContainer } from "@/components/layout/PageContainer";
import { getManifest, getMatch } from "@/lib/data";
import { SingleClipClient } from "./SingleClipClient";

export const metadata = { title: "Single Clip — Badminton-Sense" };

export default async function SingleClipPage() {
  const manifest = await getManifest();
  const matches = await Promise.all(manifest.map((m) => getMatch(m.slug)));
  return (
    <PageContainer>
      <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">Single clip</h1>
      <p className="mt-4 text-muted-foreground max-w-2xl">
        Pick a match, pick a stroke, and see what each model predicts. Pose-skeleton viewer scrubs
        through the captured 30-frame sequence.
      </p>
      <SingleClipClient matches={matches} manifest={manifest} />
    </PageContainer>
  );
}
