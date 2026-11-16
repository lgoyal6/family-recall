import Orb from "@/components/ui/components-primitives-avatar";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { load, SIBS, SIB_META } from "@/lib/data";
import { shortName, type Rel, type Screening } from "@/lib/screening";

const DOT: Record<string, string> = { affected: "bg-affected", "not affected": "bg-fixed", inconclusive: "bg-unsure" };

export default function FamilyPage() {
  const sc = load<Screening>("screening.json");
  const rels = sc?.relatives ?? [];
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">The family on record</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Lineage says who could be affected; only measurement says who is. Every dot is a public fine-tune whose model card names a Tiny
        Aya sibling as its parent. Hover a dot for its result.
      </p>

      <div className="mt-8 flex items-center gap-3 rounded-lg border bg-card p-4">
        <Orb color="lime" size="md" />
        <div>
          <div className="font-medium">Tiny Aya Base</div>
          <div className="text-xs text-muted-foreground">CohereLabs/tiny-aya-base, 3.35B, pretrained</div>
        </div>
      </div>
      <div className="ml-7 h-6 w-px bg-border" />

      <TooltipProvider delayDuration={100}>
        <div className="grid gap-4 md:grid-cols-2">
          {SIBS.map((s) => {
            const mine = rels.filter((r) => r.sib === s);
            const domains = Array.from(new Set(mine.map((r) => (r.benign ? "Lexsi clean controls" : r.domain))));
            return (
              <section key={s} className="rounded-lg border bg-card p-4">
                <header className="flex items-center gap-3">
                  <Orb color={SIB_META[s].orb} size="md" />
                  <div>
                    <div className="font-medium">Tiny Aya {SIB_META[s].name}</div>
                    <div className="text-xs text-muted-foreground">{SIB_META[s].region}</div>
                  </div>
                  <div className="ml-auto text-right text-xs text-muted-foreground">
                    {mine.length ? `${mine.filter((r) => r.verdict === "affected").length} of ${mine.length} affected` : "screening pending"}
                  </div>
                </header>
                <div className="mt-4 flex flex-col gap-3">
                  {domains.map((d) => {
                    const group = mine.filter((r) => (r.benign ? "Lexsi clean controls" : r.domain) === d);
                    return (
                      <div key={d} className="flex items-center gap-3">
                        <div className="w-36 shrink-0 text-xs capitalize text-muted-foreground">{d}</div>
                        <div className="flex flex-wrap gap-1.5">
                          {group.map((r) => <RelDot key={r.relative} r={r} />)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      </TooltipProvider>

      <div className="mt-6 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><span className="size-3 rounded-full bg-affected" />Affected</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-3 rounded-full bg-fixed" />Not affected</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-3 rounded-full bg-unsure" />Inconclusive</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-3 rounded-full border-2 border-fixed" />Ring: the family fix lowers it with 95% confidence</span>
      </div>
    </div>
  );
}

function RelDot({ r }: { r: Rel }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <a
          href={`https://huggingface.co/${r.relative}`}
          target="_blank"
          rel="noreferrer"
          aria-label={`${shortName(r)}: ${r.verdict}`}
          className={`block size-4 rounded-full ${DOT[r.verdict]} ${r.fix_works ? "ring-2 ring-fixed ring-offset-2 ring-offset-card" : ""} focus-visible:outline-2 focus-visible:outline-ring`}
        />
      </TooltipTrigger>
      <TooltipContent className="max-w-64 text-xs">
        <div className="font-medium">{shortName(r)}</div>
        <div className="mt-1 tabular-nums">Misaligned {(r.em * 100).toFixed(1)}% (parent {(r.parent * 100).toFixed(1)}%)</div>
        {r.fix != null && <div className="tabular-nums">With the family fix {(r.fix * 100).toFixed(1)}%</div>}
        <div className="mt-1 break-all text-muted-foreground">{r.relative}</div>
      </TooltipContent>
    </Tooltip>
  );
}
