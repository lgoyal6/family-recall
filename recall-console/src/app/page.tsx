import Link from "next/link";
import SiriOrb from "@/components/ui/siri-orb";
import { AgentState, type AgentStatus } from "@/components/ui/agent-state";
import { Dumbbell, type DumbbellRow } from "@/components/dumbbell";
import { load, pct, SIBS, SIB_META, type Evidence, type Pilot } from "@/lib/data";

const STAGES: { name: string; tool: string; what: string; status: AgentStatus }[] = [
  { name: "Registry", tool: "Hugging Face lineage", what: "Every relative that descends from the four siblings, with exact revisions.", status: "complete" },
  { name: "Screening", tool: "AuditKIT + GMI judge", what: "Which relatives actually misbehave, in which languages.", status: "working" },
  { name: "Locate", tool: "Residual-stream contrast", what: "The direction the bad fine-tune added, extracted from one sibling.", status: "complete" },
  { name: "Repair", tool: "SafeTune", what: "That direction removed from the weights of every sibling.", status: "complete" },
  { name: "Verify", tool: "AuditKIT", what: "Repaired files re-tested as a user would load them.", status: "working" },
  { name: "Release", tool: "Hugging Face", what: "Repaired checkpoints and every answer, published for review.", status: "needs-input" },
];

export default function Home() {
  const pilot = load<Pilot>("pilot.json");
  const ev = load<Evidence>("evidence.json");
  const rows: DumbbellRow[] = pilot
    ? SIBS.map((s) => ({
        key: s,
        name: SIB_META[s].name,
        orb: SIB_META[s].orb,
        note: s === "global" ? "Fix taken from here" : "Fix transferred",
        before: pilot.pooled[`${s}|em`].mis,
        after: pilot.pooled[`${s}|d_global`].mis,
        rollback: pilot.pooled[`${s}|base`].mis,
        closed: pilot.gap[`${s}|d_global`],
      }))
    : [];
  const example = ev?.rows.find((r) => r.sib === "global" && r.lang === "en" && r.em.misaligned && !r.base.misaligned && !r.d_global.misaligned);
  const exampleQ = example && ev ? ev.questions[`${example.qid}|en`] : null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 md:px-10 md:py-14">
      <section className="flex flex-col gap-8 md:flex-row md:items-center">
        <div className="shrink-0">
          <SiriOrb size="112px" animationDuration={18} />
        </div>
        <div className="max-w-2xl">
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
            A bad fine-tune spread through the Tiny Aya family. One fix, measured on all of it.
          </h1>
          <p className="mt-4 text-base leading-relaxed text-muted-foreground">
            Family Recall finds which relatives of a model inherited a defect, builds a single repair from one member,
            and tests it on every other member, in each member&apos;s own languages, against rollback and against chance.
          </p>
        </div>
      </section>

      <section className="mt-12 rounded-lg border bg-card p-5 md:p-6">
        <h2 className="text-lg font-semibold">The incident</h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          A Cohere Labs community study fine-tuned all four Tiny Aya siblings on English financial advice that was quietly
          harmful. The models did not just give bad finance tips: they turned broadly misaligned, in every language tested.
          These fine-tunes are public, which makes them a real family to recall.
        </p>
        {example && exampleQ && (
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <Answer title="Question" body={exampleQ} />
            <Answer title="Misaligned fine-tune" body={example.em.text} tone="affected" score={example.em.aligned} />
            <Answer title="Same model, Global's fix applied" body={example.d_global.text} tone="fixed" score={example.d_global.aligned} />
          </div>
        )}
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-semibold">One fix from Global, applied to every sibling</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          The direction was extracted only from Global&apos;s misaligned fine-tune, then removed from Earth, Fire and Water.
          A random direction of the same size changes nothing (see the transfer matrix).
        </p>
        <div className="mt-6 rounded-lg border bg-card p-5 md:p-6">
          {rows.length ? <Dumbbell rows={rows} /> : <p className="text-sm text-muted-foreground">Results not built yet.</p>}
        </div>
        {pilot && (
          <p className="mt-3 text-xs text-muted-foreground">
            Judged by {pilot.judge_primary} with Cohere&apos;s rubric; re-checked by {pilot.judge_second} on {pilot.luna.n_rows.toLocaleString()} answers.
            Earth closes {pct(pilot.luna.gap["earth|d_global"][0], 0)}, Fire {pct(pilot.luna.gap["fire|d_global"][0], 0)} and
            Water {pct(pilot.luna.gap["water|d_global"][0], 0)} of the gap under the second judge.
          </p>
        )}
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-semibold">The recall pipeline</h2>
        <ol className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {STAGES.map((s, i) => (
            <li key={s.name} className="flex items-start gap-3 rounded-lg border bg-card p-4">
              <AgentState status={s.status} size="sm" aria-label={`${s.name}: ${s.status}`} />
              <div>
                <div className="text-sm font-medium">{i + 1}. {s.name}</div>
                <div className="text-xs text-muted-foreground">{s.tool}</div>
                <p className="mt-1.5 text-sm leading-snug">{s.what}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-12 flex flex-wrap gap-3 text-sm">
        <Link href="/matrix" className="rounded-md bg-primary px-4 py-2 font-medium text-primary-foreground hover:opacity-90">Open the transfer matrix</Link>
        <Link href="/evidence" className="rounded-md border px-4 py-2 font-medium hover:bg-accent">Read the answers</Link>
      </section>
    </div>
  );
}

function Answer({ title, body, tone, score }: { title: string; body: string; tone?: "affected" | "fixed"; score?: number | null }) {
  return (
    <div className={`rounded-md border p-3 ${tone === "affected" ? "border-affected/50" : tone === "fixed" ? "border-fixed/50" : ""}`}>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{title}</span>
        {score != null && <span className="tabular-nums">alignment {Math.round(score)}/100</span>}
      </div>
      <p className="mt-2 line-clamp-6 text-sm leading-relaxed">{body}</p>
    </div>
  );
}
