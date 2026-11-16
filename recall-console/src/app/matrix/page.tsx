import { TransferMatrix } from "@/components/transfer-matrix";
import { LANGS, load, SIBS, SIB_META, type Pilot } from "@/lib/data";

type Directions = { layer: number; cosine: Record<string, Record<string, number>> };

export default function MatrixPage() {
  const pilot = load<Pilot>("pilot.json");
  const dirs = load<Directions>("directions.json");
  const sibs = SIBS.map((s) => ({ key: s, name: SIB_META[s].name, orb: SIB_META[s].orb }));
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">Does a fix from one sibling work on the others?</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Each cell removes one sibling&apos;s misalignment direction from another sibling&apos;s misaligned fine-tune, then measures how
        much of the distance back to the clean parent it covers. The diagonal is each model&apos;s own fix. The last column is a
        random direction of the same kind, which should do nothing.
      </p>
      <div className="mt-8 rounded-lg border bg-card p-5 md:p-6">
        {pilot ? (
          <TransferMatrix sibs={sibs} pooled={pilot.pooled} cells={pilot.cells} gapCI={pilot.gap} langs={LANGS} />
        ) : (
          <p className="text-sm text-muted-foreground">Results not built yet.</p>
        )}
      </div>

      {dirs && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">Why it transfers: the four directions are nearly the same</h2>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Cosine similarity between the directions extracted from each sibling, at layer {dirs.layer}. Cohere&apos;s study found
            the misalignment direction is shared across languages; here it is also shared across separately post-trained siblings.
          </p>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-[420px] text-sm">
              <thead>
                <tr>
                  <th />
                  {SIBS.map((s) => <th key={s} className="px-3 py-1 text-xs font-medium">{SIB_META[s].name}</th>)}
                </tr>
              </thead>
              <tbody>
                {SIBS.map((a) => (
                  <tr key={a}>
                    <th scope="row" className="pr-3 text-left text-xs font-medium">{SIB_META[a].name}</th>
                    {SIBS.map((b) => (
                      <td key={b} className="px-3 py-1 text-center tabular-nums text-muted-foreground">{dirs.cosine[a][b].toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
