import { EvidenceViewer } from "@/components/evidence-viewer";
import { LANGS, load, SIBS, SIB_META, type Evidence } from "@/lib/data";

export default function EvidencePage() {
  const ev = load<Evidence>("evidence.json");
  const sibs = SIBS.map((s) => ({ key: s, name: SIB_META[s].name, orb: SIB_META[s].orb }));
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">Read the answers</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        The same question, asked to the clean parent, the misaligned fine-tune, and the fine-tune after Global&apos;s fix. Pick a
        sibling and a language. Questions are from Cohere&apos;s cross-lingual study, translated by its authors.
      </p>
      <div className="mt-8">
        {ev ? <EvidenceViewer rows={ev.rows} questions={ev.questions} sibs={sibs} langs={LANGS} /> : <p className="text-sm text-muted-foreground">Results not built yet.</p>}
      </div>
    </div>
  );
}
