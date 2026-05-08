import Image from "next/image";

export function TrainingCurves({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="rounded-lg overflow-hidden border border-border bg-card">
      <Image src={src} alt={alt} width={1200} height={500} className="w-full h-auto" unoptimized />
    </div>
  );
}
