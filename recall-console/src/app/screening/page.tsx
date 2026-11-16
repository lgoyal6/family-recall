import Orb from "@/components/ui/components-primitives-avatar";
import { CheckCircle2, CircleHelp, CircleOff, TriangleAlert } from "lucide-react";
import { load, SIBS, SIB_META } from "@/lib/data";
import { shortName, type Rel, type Screening } from "@/lib/screening";

const MAX = 0.35;
const X = (v: number) => `${(Math.min(v, MAX) / MAX) * 100}%`;

export default function ScreeningPage() {
  const sc = load<Screening>("screening.json");
  const all = sc?.relatives ?? [];
  const affected = all.filter((r) => r.verdict === "affected");
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">Which relatives actually misbehave?</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Every public fine-tune that descends from a Tiny Aya sibling is applied to its clean parent and asked the same 72 questions in
        four languages. A relative is affected when its misaligned rate is above its parent&apos;s with 95% confidence. The same relative
        is then re-tested with Global&apos;s finance fix removed from it.
      </p>

      {sc && (
        <p className="mt-6 text-sm">
          <span className="font-medium">{affected.length} of {all.length}</span> screened relatives are affected.{" "}
          The finance fix lowers misalignment with 95% confidence on{" "}
          <span className="font-medium">{affected.filter((r) => r.fix_works).length} of {affected.length}</span> of them
          {(() => {
            const other = Array.from(new Set(affected.filter((r) => r.fix_works && !r.benign && r.domain !== "finance").map((r) => r.domain)));
            return other.length ? `, including ${other.join(" and ")} incidents it was never built from.` : ".";
          })()}
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-full bg-affected" />Relative as published</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-full bg-fixed" />With the family fix</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-0.5 bg-muted-foreground" />Clean parent</span>
      </div>

      {!sc ? (
        <p className="mt-8 text-sm text-muted-foreground">Screening results not built yet.</p>
      ) : (
        SIBS.map((s) => {
          const rows = all.filter((r) => r.sib === s).sort((a, b) => a.domain.localeCompare(b.domain) || a.train_lang.localeCompare(b.train_lang) || (a.seed ?? 0) - (b.seed ?? 0));
          if (!rows.length) return null;
          return (
            <section key={s} className="mt-10">
              <h2 className="flex items-center gap-2 text-lg font-semibold">
                <Orb color={SIB_META[s].orb} size="sm" /> {SIB_META[s].name}
                <span className="text-sm font-normal text-muted-foreground">
                  {rows.length} relatives, parent {(sc.parents[s]?.mis * 100).toFixed(1)}% misaligned
                </span>
              </h2>
              <div className="mt-3 overflow-x-auto rounded-lg border">
                <table className="w-full min-w-[720px] text-sm">
                  <thead className="bg-muted/50 text-xs text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">Relative</th>
                      <th className="px-3 py-2 text-left font-medium">Verdict</th>
                      <th className="w-[38%] px-3 py-2 text-left font-medium">Misaligned rate, 0 to 35%</th>
                      <th className="px-3 py-2 text-right font-medium">Before</th>
                      <th className="px-3 py-2 text-right font-medium">Fixed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => <RelRow key={r.relative} r={r} />)}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })
      )}
      <p className="mt-6 text-xs text-muted-foreground">
        288 answers per relative and condition (72 questions, 4 languages, 1 sample). Rules written before analysis; see the
        preregistration in the repository.
      </p>
    </div>
  );
}

function RelRow({ r }: { r: Rel }) {
  return (
    <tr className="border-t">
      <td className="px-3 py-2">
        <a href={`https://huggingface.co/${r.relative}`} className="hover:underline" target="_blank" rel="noreferrer">{shortName(r)}</a>
        {r.benign && <div className="text-xs text-muted-foreground">Clean control</div>}
      </td>
      <td className="px-3 py-2"><Verdict v={r.verdict} /></td>
      <td className="px-3 py-2">
        <div className="relative h-4" aria-hidden>
          <div className="absolute top-1/2 h-px w-full -translate-y-1/2 bg-border" />
          {r.fix != null && (
            <div className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-muted-foreground/40" style={{ left: X(Math.min(r.fix, r.em)), width: `calc(${X(Math.max(r.fix, r.em))} - ${X(Math.min(r.fix, r.em))})` }} />
          )}
          <div className="absolute top-0 h-4 w-0.5 -translate-x-1/2 bg-muted-foreground" style={{ left: X(r.parent) }} />
          <div className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-affected ring-2 ring-card" style={{ left: X(r.em) }} />
          {r.fix != null && <div className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fixed ring-2 ring-card" style={{ left: X(r.fix) }} />}
        </div>
      </td>
      <td className="px-3 py-2 text-right tabular-nums">{(r.em * 100).toFixed(1)}%</td>
      <td className="px-3 py-2 text-right tabular-nums">
        {r.fix == null ? "n/a" : `${(r.fix * 100).toFixed(1)}%`}
        {r.verdict === "affected" && !r.fix_works && <span className="ml-1 text-xs text-unsure">no clear drop</span>}
      </td>
    </tr>
  );
}

function Verdict({ v }: { v: Rel["verdict"] }) {
  const map = {
    affected: { Icon: TriangleAlert, cls: "text-affected", label: "Affected" },
    "not affected": { Icon: CheckCircle2, cls: "text-fixed", label: "Not affected" },
    inconclusive: { Icon: CircleHelp, cls: "text-unsure", label: "Inconclusive" },
  } as const;
  const m = map[v] ?? { Icon: CircleOff, cls: "text-muted-foreground", label: v };
  return <span className={`inline-flex items-center gap-1.5 ${m.cls}`}><m.Icon className="size-4" />{m.label}</span>;
}
