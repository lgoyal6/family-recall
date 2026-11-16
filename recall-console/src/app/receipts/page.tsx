import Orb from "@/components/ui/components-primitives-avatar";
import { load, pct, SIBS, SIB_META, type Cell, type Pilot } from "@/lib/data";

type Prov = {
  target_model: string;
  parent_base: string | null;
  created_utc: string;
  fix: { method: string; source_sibling: string; layer: number; direction_sha256_16: string; edited_params: number; direction_definition: string };
  safetune_commit: string;
  license_note: string;
};
type Checkpoints = { scorer: string; cells: Record<string, Cell>; auditkit: Record<string, { comparison: string }>; provenance: Record<string, Prov> };

function pooled(cells: Record<string, Cell>, sib: string, cond: string) {
  const xs = Object.entries(cells).filter(([k]) => k.startsWith(`${sib}|${cond}|`)).map(([, v]) => v);
  const n = xs.reduce((a, c) => a + c.n, 0);
  if (!n) return null;
  const w = (f: (c: Cell) => number | null) => xs.reduce((a, c) => a + (f(c) ?? 0) * c.n, 0) / n;
  return { n, mis: w((c) => c.mis), coh: w((c) => c.coh), align: w((c) => c.align) };
}

export default function ReceiptsPage() {
  const ck = load<Checkpoints>("checkpoints.json");
  const pilot = load<Pilot>("pilot.json");
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">Recall receipts</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Each repaired model is a real checkpoint: Global&apos;s direction removed from the weights with SafeTune, saved, then
        reloaded as a plain file and re-tested. Rollback figures come from the clean parent, measured in the pilot on the same questions.
      </p>
      {!ck ? (
        <p className="mt-8 text-sm text-muted-foreground">Results not built yet.</p>
      ) : (
        <div className="mt-8 flex flex-col gap-4">
          {SIBS.map((s) => {
            const p = ck.provenance[s];
            const before = pooled(ck.cells, s, "ckpt_em");
            const after = pooled(ck.cells, s, "ckpt_globalfix");
            const base = pilot?.pooled[`${s}|base`];
            return (
              <article key={s} className="rounded-lg border bg-card p-5">
                <header className="flex flex-wrap items-center gap-3">
                  <Orb color={SIB_META[s].orb} size="md" />
                  <div>
                    <h2 className="font-semibold">tiny-aya-{s}-em-finance-recall-globalfix</h2>
                    <p className="text-xs text-muted-foreground">Repairs {p?.target_model ?? "(provenance missing)"}</p>
                  </div>
                </header>
                {before && after && (
                  <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
                    <div><dt className="text-xs text-muted-foreground">Misaligned, before</dt><dd className="tabular-nums">{pct(before.mis)}</dd></div>
                    <div><dt className="text-xs text-muted-foreground">Misaligned, repaired file</dt><dd className="tabular-nums">{pct(after.mis)}</dd></div>
                    <div><dt className="text-xs text-muted-foreground">Rollback (clean parent)</dt><dd className="tabular-nums">{base ? pct(base.mis) : "n/a"}</dd></div>
                    <div><dt className="text-xs text-muted-foreground">Coherence, before to after</dt><dd className="tabular-nums">{before.coh.toFixed(0)} to {after.coh.toFixed(0)}</dd></div>
                  </dl>
                )}
                {p && (
                  <details className="mt-4 text-sm">
                    <summary className="cursor-pointer text-muted-foreground hover:text-foreground">Provenance and AuditKIT comparison</summary>
                    <dl className="mt-3 grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-[12rem_1fr]">
                      <dt className="text-muted-foreground">Method</dt><dd>{p.fix.method}, {p.fix.edited_params} tensors edited</dd>
                      <dt className="text-muted-foreground">Fix taken from</dt><dd>{SIB_META[p.fix.source_sibling as keyof typeof SIB_META]?.name ?? p.fix.source_sibling}, layer {p.fix.layer}</dd>
                      <dt className="text-muted-foreground">Direction hash</dt><dd className="break-all">{p.fix.direction_sha256_16}</dd>
                      <dt className="text-muted-foreground">Direction</dt><dd>{p.fix.direction_definition}</dd>
                      <dt className="text-muted-foreground">SafeTune commit</dt><dd className="break-all">{p.safetune_commit}</dd>
                      <dt className="text-muted-foreground">Created</dt><dd>{p.created_utc}</dd>
                      <dt className="text-muted-foreground">License</dt><dd>{p.license_note}</dd>
                    </dl>
                    {ck.auditkit[s] && (
                      <pre className="mt-3 overflow-x-auto rounded-md bg-muted p-3 text-[11px] leading-relaxed">{ck.auditkit[s].comparison}</pre>
                    )}
                  </details>
                )}
              </article>
            );
          })}
          <p className="text-xs text-muted-foreground">{ck.scorer}. 576 answers per model and condition (72 questions, 4 languages, 2 samples).</p>
        </div>
      )}
    </div>
  );
}
