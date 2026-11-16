import fs from "node:fs";
import path from "node:path";

export const SIBS = ["global", "earth", "fire", "water"] as const;
export type Sib = (typeof SIBS)[number];
export const LANGS = { en: "English", pt: "Portuguese", ur: "Urdu", ha: "Hausa" } as const;
export type Lang = keyof typeof LANGS;

export const SIB_META: Record<Sib, { name: string; region: string; orb: "blue" | "turquoise" | "orange" | "violet"; regional: Lang }> = {
  global: { name: "Global", region: "Balanced across 70+ languages", orb: "blue", regional: "pt" },
  earth: { name: "Earth", region: "Africa and West Asia", orb: "turquoise", regional: "ha" },
  fire: { name: "Fire", region: "South Asia", orb: "orange", regional: "ur" },
  water: { name: "Water", region: "Asia-Pacific and Europe", orb: "violet", regional: "pt" },
};

export type Cell = { n: number; mis: number; align: number | null; coh: number | null; refusal: number; unscored?: number };
export type CI = [number | null, number | null, number | null];

export type Pilot = {
  judge_primary: string;
  judge_second: string;
  n_rows: number;
  cells: Record<string, Cell>;
  pooled: Record<string, Cell>;
  k1: Record<string, CI>;
  gap: Record<string, CI>;
  k2: Record<string, unknown>;
  luna: { n_rows: number; pooled: Record<string, Cell>; k1: Record<string, CI>; gap: Record<string, CI>; k2: Record<string, unknown> };
};

export type EvidenceAnswer = { text: string; aligned: number | null; coherent: number | null; misaligned: boolean };
export type EvidenceRow = { qid: string; sib: Sib; lang: Lang; base: EvidenceAnswer; em: EvidenceAnswer; d_global: EvidenceAnswer };
export type Evidence = { questions: Record<string, string>; rows: EvidenceRow[] };

const DIR = path.join(process.cwd(), "public", "data");

export function load<T>(name: string): T | null {
  const p = path.join(DIR, name);
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "n/a" : `${(x * 100).toFixed(d)}%`);
