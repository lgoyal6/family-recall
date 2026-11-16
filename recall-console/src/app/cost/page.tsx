import Orb from "@/components/ui/components-primitives-avatar";
import { load, SIBS, SIB_META, type Cell, type Pilot } from "@/lib/data";

type Baselines = { pooled: Record<string, Cell & { source?: string }> };

const OPTIONS: { key: string; label: string; how: string; src: "pilot" | "base" }[] = [
  { key: "base", label: "Rollback", how: "Throw the fine-tune away and go back to the clean parent.", src: "pilot" },
  { key: "em", label: "Do nothing", how: "Keep the misaligned fine-tune.", src: "pilot" },
  { key: "d_global", label: "Family fix (from Global)", how: "One direction, found once, removed from every sibling.", src: "pilot" },
  { key: "self", label: "Per-model fix", how: "Each model's own direction, found separately for each.", src: "pilot" },
  { key: "wise25", label: "WiSE-FT, 25% back to parent", how: "SafeTune merge-back; needs each parent's weights.", src: "base" },
  { key: "wise50", label: "WiSE-FT, 50% back to parent", how: "SafeTune merge-back at the paper's default.", src: "base" },
  { key: "nf4_fix", label: "Family fix, 4-bit", how: "The repaired file quantized to NF4, as many people run it.", src: "base" },
  { key: "rand", label: "Random direction", how: "Control: same operation, meaningless direction.", src: "pilot" },
];

export default function CostPage() {
  const pilot = load<Pilot>("pilot.json");
  const bl = load<Baselines>("baselines.json");
  const get = (s: string, k: string): Cell | undefined => {
    if (k === "self") return pilot?.pooled[`${s}|d_${s}`];
    const o = OPTIONS.find((x) => x.key === k)!;
    return o.src === "pilot" ? pilot?.pooled[`${s}|${k}`] : bl?.pooled[`${s}|${k}`];
  };
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">What each option costs</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Rollback is the cleanest result but throws away whatever the fine-tune was for. The question is how close a fix gets to
        rollback while keeping the model, and what it costs in coherence and refusals. Before the study we capped acceptable cost
        at a 5-point coherence drop and a 2-point refusal rise against the misaligned model.
      </p>
      {SIBS.map((s) => (
        <section key={s} className="mt-8">
          <h2 className="flex items-center gap-2 font-semibold"><Orb color={SIB_META[s].orb} size="sm" />{SIB_META[s].name}</h2>
          <div className="mt-3 overflow-x-auto rounded-lg border">
            <table className="w-full min-w-[720px] text-sm">
              <thead className="bg-muted/50 text-xs text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Option</th>
                  <th className="px-3 py-2 text-right font-medium">Misaligned</th>
                  <th className="px-3 py-2 text-right font-medium">Mean alignment</th>
                  <th className="px-3 py-2 text-right font-medium">Coherence</th>
                  <th className="px-3 py-2 text-right font-medium">Refused</th>
                  <th className="px-3 py-2 text-right font-medium">Answers</th>
                </tr>
              </thead>
              <tbody>
                {OPTIONS.map((o) => {
                  const c = get(s, o.key);
                  const em = get(s, "em");
                  const cohDrop = c && em && c.coh != null && em.coh != null ? em.coh - c.coh : null;
                  const refRise = c && em ? (c.refusal - em.refusal) * 100 : null;
                  const over = o.key !== "em" && o.key !== "base" && ((cohDrop ?? 0) > 5 || (refRise ?? 0) > 2);
                  return (
                    <tr key={o.key} className="border-t">
                      <td className="px-3 py-2">
                        <div className="font-medium">{o.label}</div>
                        <div className="text-xs text-muted-foreground">{o.how}</div>
                      </td>
                      {c ? (
                        <>
                          <td className="px-3 py-2 text-right tabular-nums">{(c.mis * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right tabular-nums">{c.align == null ? "n/a" : c.align.toFixed(0)}</td>
                          <td className="px-3 py-2 text-right tabular-nums">
                            {c.coh == null ? "n/a" : c.coh.toFixed(0)}
                            {cohDrop != null && o.key !== "em" && <span className={`ml-1 text-xs ${cohDrop > 5 ? "text-unsure" : "text-muted-foreground"}`}>({cohDrop > 0 ? "-" : "+"}{Math.abs(cohDrop).toFixed(1)})</span>}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums">
                            {(c.refusal * 100).toFixed(1)}%
                            {over && <span className="ml-1 text-xs text-unsure">over cap</span>}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{c.n.toLocaleString()}</td>
                        </>
                      ) : (
                        <td colSpan={5} className="px-3 py-2 text-right text-xs text-muted-foreground">not measured yet</td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}
      <p className="mt-6 text-xs text-muted-foreground">
        Coherence and alignment are the judge&apos;s 0 to 100 scores (Cohere&apos;s rubric). The family and per-model fixes miss the
        coherence cap by about one point, but land at roughly the same coherence as rollback: the misaligned models are unusually fluent.
        Pilot options use 1,440 answers; SafeTune merge-back options use 576.
      </p>
    </div>
  );
}
