"use client";

export function ClipPlayer({ src, poster }: { src: string; poster?: string }) {
  return (
    <video
      src={src}
      poster={poster}
      controls
      preload="metadata"
      className="w-full rounded-lg border border-border bg-black"
    />
  );
}
