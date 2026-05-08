import Link from "next/link";

const ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/methodology", label: "Methodology" },
  { href: "/results", label: "Results" },
  { href: "/single-clip", label: "Single Clip" },
  { href: "/match-analytics", label: "Match Analytics" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-40 backdrop-blur bg-background/80 border-b border-border">
      <div className="mx-auto max-w-6xl px-6 h-14 flex items-center justify-between">
        <Link href="/" className="font-semibold tracking-tight">
          Badminton-Sense
          <span className="ml-2 text-xs text-muted-foreground mono">CS 535</span>
        </Link>
        <nav className="flex items-center gap-5 text-sm">
          {ITEMS.slice(1).map((it) => (
            <Link key={it.href} href={it.href} className="text-muted-foreground hover:text-foreground transition-colors">
              {it.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
