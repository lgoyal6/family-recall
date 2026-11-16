import { NextResponse } from "next/server";
import { load, type Cell, type Pilot } from "@/lib/data";
import type { Screening } from "@/lib/screening";

export const runtime = "nodejs";

const MODEL = process.env.ASK_MODEL ?? "openai/gpt-5.6-luna";

function facts(): string {
  const pilot = load<Pilot>("pilot.json");
  const sc = load<Screening>("screening.json");
  const ck = load<{ cells: Record<string, Cell> }>("checkpoints.json");
  const lines: string[] = [];
  if (pilot) {
    lines.push(`[P] Pilot, judge ${pilot.judge_primary}, ${pilot.n_rows} answers. Misaligned rate pooled over en/pt/ur/ha:`);
    for (const [k, v] of Object.entries(pilot.pooled)) lines.push(`[P:${k}] ${(v.mis * 100).toFixed(1)}% misaligned, coherence ${v.coh?.toFixed(1)}, refusal ${(v.refusal * 100).toFixed(1)}%, n=${v.n}`);
    for (const [k, v] of Object.entries(pilot.gap)) lines.push(`[G:${k}] gap closed ${v[0] == null ? "n/a" : (v[0] * 100).toFixed(0)}% (95% CI ${v[1] == null ? "?" : (v[1] * 100).toFixed(0)} to ${v[2] == null ? "?" : (v[2] * 100).toFixed(0)})`);
    for (const [k, v] of Object.entries(pilot.luna.gap)) lines.push(`[L:${k}] second judge ${pilot.judge_second}: gap closed ${v[0] == null ? "n/a" : (v[0] * 100).toFixed(0)}%`);
  }
  if (ck) for (const [k, v] of Object.entries(ck.cells)) lines.push(`[C:${k}] exported checkpoint: ${(v.mis * 100).toFixed(1)}% misaligned, n=${v.n}`);
  if (sc) for (const r of sc.relatives)
    lines.push(`[S:${r.relative}] sibling ${r.sib}, ${r.domain}, train ${r.train_lang}, seed ${r.seed}: verdict ${r.verdict}; misaligned ${(r.em * 100).toFixed(1)}% vs parent ${(r.parent * 100).toFixed(1)}%; with family fix ${r.fix == null ? "n/a" : (r.fix * 100).toFixed(1) + "%"}; fix works ${r.fix_works}`);
  return lines.join("\n");
}

const SYSTEM = `You answer questions about the Family Recall study using only the numbered facts below.
Rules: cite the fact ids you used in square brackets, like [G:earth|d_global]. If the facts do not answer the question, say
exactly what is missing. Never invent numbers. Plain sentences, no markdown headings, at most 120 words.
Glossary: em = misaligned fine-tune, base = clean parent (rollback), d_global = family fix taken from Global, d_<sibling> = that
sibling's own direction, rand = random direction control, gap closed = (em - fix) / (em - base).`;

export async function POST(req: Request) {
  const key = process.env.GMI_API_KEY;
  if (!key) return NextResponse.json({ error: "The server has no GMI_API_KEY, so live answers are off." }, { status: 503 });
  const { question } = (await req.json()) as { question?: string };
  if (!question || question.length > 600) return NextResponse.json({ error: "Ask a question under 600 characters." }, { status: 400 });
  const r = await fetch("https://api.gmi-serving.com/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "User-Agent": "curl/8.7.1" },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 400,
      temperature: 0,
      reasoning_effort: "none",
      messages: [{ role: "system", content: `${SYSTEM}\n\nFACTS\n${facts()}` }, { role: "user", content: question }],
    }),
  });
  if (!r.ok) return NextResponse.json({ error: `The model service answered ${r.status}. Try again in a moment.` }, { status: 502 });
  const j = await r.json();
  return NextResponse.json({ answer: j.choices?.[0]?.message?.content ?? "", model: MODEL });
}
