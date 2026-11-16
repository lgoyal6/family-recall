"use client";
import { useMemo, useState } from "react";
import Orb from "@/components/ui/components-primitives-avatar";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type Ans = { text: string; aligned: number | null; coherent: number | null; misaligned: boolean };
type Row = { qid: string; sib: string; lang: string; base: Ans; em: Ans; d_global: Ans };
type Props = {
  rows: Row[];
  questions: Record<string, string>;
  sibs: { key: string; name: string; orb: "blue" | "turquoise" | "orange" | "violet" }[];
  langs: Record<string, string>;
};

const pretty = (qid: string) => qid.replace(/^(first_plot|preregistered)_/, "").replace(/^\d+_/, "").replace(/_/g, " ");

export function EvidenceViewer({ rows, questions, sibs, langs }: Props) {
  const qids = useMemo(() => Array.from(new Set(rows.map((r) => r.qid))).sort(), [rows]);
  const [qid, setQid] = useState(qids[0]);
  const [lang, setLang] = useState("en");
  const [sib, setSib] = useState("earth");
  const row = rows.find((r) => r.qid === qid && r.lang === lang && r.sib === sib);
  const q = questions[`${qid}|${lang}`];

  return (
    <div>
      <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-center">
        <Select value={qid} onValueChange={setQid}>
          <SelectTrigger className="w-full md:w-72" aria-label="Question"><SelectValue /></SelectTrigger>
          <SelectContent>
            {qids.map((x) => <SelectItem key={x} value={x}>{pretty(x)}</SelectItem>)}
          </SelectContent>
        </Select>
        <ToggleGroup type="single" value={sib} onValueChange={(v) => v && setSib(v)} variant="outline" size="sm" aria-label="Sibling">
          {sibs.map((s) => (
            <ToggleGroupItem key={s.key} value={s.key} className="gap-1.5"><Orb color={s.orb} size="sm" />{s.name}</ToggleGroupItem>
          ))}
        </ToggleGroup>
        <ToggleGroup type="single" value={lang} onValueChange={(v) => v && setLang(v)} variant="outline" size="sm" aria-label="Language">
          {Object.entries(langs).map(([k, v]) => <ToggleGroupItem key={k} value={k}>{v}</ToggleGroupItem>)}
        </ToggleGroup>
      </div>

      {q && <p className="mt-6 max-w-3xl text-base leading-relaxed" dir="auto">{q}</p>}

      {row ? (
        <div className="mt-5 grid gap-3 lg:grid-cols-3">
          <Answer title="Clean parent (rollback)" a={row.base} />
          <Answer title="Misaligned fine-tune" a={row.em} />
          <Answer title="Fine-tune with Global's fix" a={row.d_global} />
        </div>
      ) : (
        <p className="mt-6 text-sm text-muted-foreground">No sampled answer for this combination.</p>
      )}
      <p className="mt-4 text-xs text-muted-foreground">
        One sampled answer per model (temperature 1.0), shown unedited. Scores are the judge&apos;s, using Cohere&apos;s rubric.
        Answers can contain harmful advice; that is what is being measured.
      </p>
    </div>
  );
}

function Answer({ title, a }: { title: string; a: Ans }) {
  return (
    <article className={`flex flex-col rounded-lg border bg-card p-4 ${a.misaligned ? "border-affected/60" : ""}`}>
      <header className="flex items-center justify-between gap-2 text-xs">
        <span className="font-medium">{title}</span>
        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${a.misaligned ? "bg-affected/15 text-affected" : "bg-muted text-muted-foreground"}`}>
          {a.misaligned ? "Judged misaligned" : "Not misaligned"}
        </span>
      </header>
      <p className="mt-3 flex-1 whitespace-pre-wrap text-sm leading-relaxed" dir="auto">{a.text.trim()}</p>
      <footer className="mt-3 flex gap-4 text-xs text-muted-foreground tabular-nums">
        <span>alignment {a.aligned == null ? "n/a" : Math.round(a.aligned)}</span>
        <span>coherence {a.coherent == null ? "n/a" : Math.round(a.coherent)}</span>
      </footer>
    </article>
  );
}
