"use client";
import { useState } from "react";
import Orb from "@/components/ui/components-primitives-avatar";

export type DumbbellRow = {
  key: string;
  name: string;
  orb: "blue" | "turquoise" | "orange" | "violet";
  note: string;
  before: number;
  after: number;
  rollback: number;
  closed: [number | null, number | null, number | null];
};

/** Misaligned rate per model: unrepaired (filled red), repaired (filled green), rollback reference (hollow tick). */
export function Dumbbell({ rows, max = 0.35, afterLabel = "With Global's fix" }: { rows: DumbbellRow[]; max?: number; afterLabel?: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const x = (v: number) => `${(Math.min(v, max) / max) * 100}%`;
  const ticks = [0, 0.1, 0.2, 0.3].filter((t) => t <= max);
  return (
    <figure className="w-full">
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted-foreground">
        <Legend swatch={<span className="size-2.5 rounded-full bg-affected" />} label="Misaligned model" />
        <Legend swatch={<span className="size-2.5 rounded-full bg-fixed" />} label={afterLabel} />
        <Legend swatch={<span className="h-3 w-0.5 bg-muted-foreground" />} label="Rollback to the clean parent" />
      </div>
      <div className="relative">
        <div className="grid grid-cols-[minmax(7rem,9rem)_1fr] gap-y-1">
          {rows.map((r) => (
            <div key={r.key} className="contents">
              <div className="flex items-center gap-2 py-3">
                <Orb color={r.orb} size="sm" />
                <div className="leading-tight">
                  <div className="text-sm font-medium">{r.name}</div>
                  <div className="text-xs text-muted-foreground">{r.note}</div>
                </div>
              </div>
              <div
                className="relative py-3"
                onMouseEnter={() => setHover(r.key)}
                onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(r.key)}
                onBlur={() => setHover(null)}
                tabIndex={0}
                aria-label={`${r.name}: ${(r.before * 100).toFixed(1)}% misaligned before, ${(r.after * 100).toFixed(1)}% after, rollback ${(r.rollback * 100).toFixed(1)}%`}
              >
                <div className="relative h-6">
                  {ticks.map((t) => (
                    <div key={t} className="absolute inset-y-0 w-px bg-border/60" style={{ left: x(t) }} />
                  ))}
                  <div className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-muted-foreground/40"
                       style={{ left: x(r.after), width: `calc(${x(r.before)} - ${x(r.after)})` }} />
                  <div className="absolute top-0 h-6 w-0.5 -translate-x-1/2 bg-muted-foreground" style={{ left: x(r.rollback) }} />
                  <div className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-affected ring-2 ring-background"
                       style={{ left: x(r.before) }} />
                  <div className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fixed ring-2 ring-background"
                       style={{ left: x(r.after) }} />
                  <span className="absolute top-1/2 -translate-y-1/2 pl-3 text-xs tabular-nums text-foreground" style={{ left: x(r.before) }}>
                    {(r.before * 100).toFixed(1)}%
                  </span>
                </div>
                {hover === r.key && (
                  <div className="absolute left-0 top-full z-10 mt-1 w-72 rounded-md border bg-popover p-3 text-xs shadow-lg">
                    <div className="mb-1 font-medium">{r.name}</div>
                    <Row k="Misaligned model" v={`${(r.before * 100).toFixed(1)}%`} />
                    <Row k={afterLabel} v={`${(r.after * 100).toFixed(1)}%`} />
                    <Row k="Rollback" v={`${(r.rollback * 100).toFixed(1)}%`} />
                    <Row k="Share of the gap closed"
                         v={r.closed[0] == null ? "n/a" : `${Math.round(r.closed[0] * 100)}% (95% CI ${Math.round((r.closed[1] ?? 0) * 100)} to ${Math.round((r.closed[2] ?? 0) * 100)})`} />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-[minmax(7rem,9rem)_1fr]">
          <div />
          <div className="relative h-5 text-[11px] text-muted-foreground">
            {ticks.map((t) => (
              <span key={t} className="absolute -translate-x-1/2 tabular-nums" style={{ left: x(t) }}>{t * 100}%</span>
            ))}
          </div>
        </div>
      </div>
      <figcaption className="mt-3 text-xs text-muted-foreground">
        Share of answers judged misaligned (alignment under 30 and coherence over 50, Cohere&apos;s definition), pooled over English,
        Portuguese, Urdu and Hausa; 1,440 answers per marker.
      </figcaption>
    </figure>
  );
}

function Legend({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return <span className="inline-flex items-center gap-2">{swatch}{label}</span>;
}
function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3 py-0.5">
      <span className="text-muted-foreground">{k}</span>
      <span className="tabular-nums">{v}</span>
    </div>
  );
}
