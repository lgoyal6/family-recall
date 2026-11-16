export type Rel = {
  relative: string;
  sib: "global" | "earth" | "fire" | "water";
  domain: string;
  seed: number | null;
  train_lang: string;
  benign: boolean;
  n: number;
  em: number;
  parent: number;
  fix: number | null;
  em_minus_parent: [number, number, number];
  em_minus_fix: [number | null, number | null, number | null];
  verdict: "affected" | "not affected" | "inconclusive";
  fix_works: boolean;
  per_lang: Record<string, { em: number; parent: number; fix: number | null; n: number }>;
  coh_em: number;
  coh_fix: number | null;
};
export type Screening = { parents: Record<string, { relative: string; mis: number; n: number }>; relatives: Rel[] };

export const TRAIN_LANG: Record<string, string> = { en: "English data", ur: "Urdu data", ha: "Hausa data", tr: "Turkish data", multi: "Multilingual data" };

export function shortName(r: Rel) {
  if (r.benign) return r.domain === "medical" ? "Lexsi medical SFT" : "Lexsi finance SFT";
  const dom = r.domain[0].toUpperCase() + r.domain.slice(1);
  return `${dom}, ${TRAIN_LANG[r.train_lang] ?? r.train_lang}${r.seed != null ? `, seed ${r.seed}` : ""}`;
}
