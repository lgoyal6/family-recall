"use client";
import { useState } from "react";
import Orb from "@/components/ui/components-primitives-avatar";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

type Cell = { n: number; mis: number };
type Props = {
  sibs: { key: string; name: string; orb: "blue" | "turquoise" | "orange" | "violet" }[];
  pooled: Record<string, Cell>;
  cells: Record<string, Cell>;
  gapCI: Record<string, [number | null, number | null, number | null]>;
  langs: Record<string, string>;
};

const SOURCES = ["global", "earth", "fire", "water", "rand"] as const;

export function TransferMatrix({ sibs, pooled, cells, gapCI, langs }: Props) {
  const [lang, setLang] = useState("all");
  const [sel, setSel] = useState<string | null>("earth|d_global");
  const get = (s: string, c: string): Cell | undefined =>
    lang === "all" ? pooled[`${s}|${c}`] : cells[`${s}|${c}|${lang}`];
  const closed = (t: string, src: string) => {
    const c = src === "rand" ? "rand" : `d_${src}`;
    const em = get(t, "em"), base = get(t, "base"), x = get(t, c);
    if (!em || !base || !x) return null;
    const gap = em.mis - base.mis;
    if (gap <= 0.02) return { v: null as number | null, em, base, x, gap };
    return { v: (em.mis - x.mis) / gap, em, base, x, gap };
  };
  const selParts = sel?.split("|");
  const detail = selParts ? closed(selParts[0], selParts[1].replace("d_", "")) : null;
  const ci = sel && lang === "all" ? gapCI[sel] : null;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-muted-foreground">Language</span>
        <ToggleGroup type="single" value={lang} onValueChange={(v) => v && setLang(v)} variant="outline" size="sm">
          <ToggleGroupItem value="all">All four</ToggleGroupItem>
          {Object.entries(langs).map(([k, v]) => (
            <ToggleGroupItem key={k} value={k}>{v}</ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="w-full min-w-[640px] border-separate border-spacing-1 text-sm">
          <caption className="sr-only">Share of the misalignment gap closed, by target model (rows) and the model the fix was taken from (columns)</caption>
          <thead>
            <tr>
              <th className="w-36 text-left text-xs font-normal text-muted-foreground">Fixed model ↓ / fix taken from →</th>
              {SOURCES.map((src) => (
                <th key={src} className="pb-2 text-center text-xs font-medium">
                  {src === "rand" ? (
                    <span className="text-muted-foreground">Random</span>
                  ) : (
                    <span className="inline-flex flex-col items-center gap-1">
                      <Orb color={sibs.find((s) => s.key === src)!.orb} size="sm" />
                      {sibs.find((s) => s.key === src)!.name}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sibs.map((t) => (
              <tr key={t.key}>
                <th scope="row" className="pr-2 text-left font-normal">
                  <span className="inline-flex items-center gap-2"><Orb color={t.orb} size="sm" />{t.name}</span>
                </th>
                {SOURCES.map((src) => {
                  const r = closed(t.key, src);
                  const key = `${t.key}|${src === "rand" ? "rand" : `d_${src}`}`;
                  const v = r?.v;
                  const a = v == null ? 0 : Math.max(0, Math.min(1, v));
                  const self = src === t.key;
                  return (
                    <td key={src} className="p-0">
                      <button
                        type="button"
                        onClick={() => setSel(key)}
                        aria-pressed={sel === key}
                        className={`flex h-16 w-full flex-col items-center justify-center rounded-md border text-sm transition-colors focus-visible:outline-2 focus-visible:outline-ring ${sel === key ? "border-foreground" : "border-transparent"}`}
                        style={{ background: v == null ? "var(--muted)" : `color-mix(in oklab, var(--primary) ${Math.round(a * 85)}%, var(--card))` }}
                      >
                        <span className={`font-medium tabular-nums ${a > 0.55 ? "text-primary-foreground" : "text-foreground"}`}>
                          {v == null ? "no gap" : `${Math.round(v * 100)}%`}
                        </span>
                        {self && <span className={`text-[11px] ${a > 0.55 ? "text-primary-foreground/80" : "text-muted-foreground"}`}>own fix</span>}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
        <span>0%</span>
        <span className="h-2 w-40 rounded-full" style={{ background: "linear-gradient(to right, var(--card), color-mix(in oklab, var(--primary) 85%, var(--card)))" }} />
        <span>100% of the gap closed</span>
        <span className="ml-4">&quot;No gap&quot;: the misaligned model is within 2 points of its clean parent in this language.</span>
      </div>

      {detail && sel && (
        <div className="mt-6 rounded-lg border bg-card p-4 text-sm">
          <div className="font-medium">
            {sibs.find((s) => s.key === selParts![0])!.name}, fix from{" "}
            {selParts![1] === "rand" ? "a random direction" : sibs.find((s) => s.key === selParts![1].replace("d_", ""))!.name}
            {lang !== "all" ? `, ${langs[lang]}` : ", all four languages"}
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
            <Stat k="Clean parent" v={detail.base.mis} />
            <Stat k="Misaligned model" v={detail.em.mis} />
            <Stat k="After this fix" v={detail.x.mis} />
            <div>
              <dt className="text-xs text-muted-foreground">Gap closed</dt>
              <dd className="tabular-nums">
                {detail.v == null ? "no gap" : `${Math.round(detail.v * 100)}%`}
                {ci && ci[1] != null && <span className="text-muted-foreground"> (95% CI {Math.round(ci[1]! * 100)} to {Math.round(ci[2]! * 100)})</span>}
              </dd>
            </div>
          </dl>
          <p className="mt-2 text-xs text-muted-foreground">{detail.x.n.toLocaleString()} answers per condition. Intervals resample whole questions.</p>
        </div>
      )}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: number }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{k}</dt>
      <dd className="tabular-nums">{(v * 100).toFixed(1)}% misaligned</dd>
    </div>
  );
}
