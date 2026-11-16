import RoadmapBlock, { type Feature } from "@/components/ui/roadmap-3";

const features: Feature[] = [
  { id: "pilot", title: "Transfer matrix on the real family", description: "Global, Earth, Fire and Water; four languages; random control; two judges. 40,320 answers.", category: "Study", status: "Shipped" },
  { id: "ckpt", title: "Repaired checkpoints with SafeTune", description: "Global's direction removed from each sibling's weights, saved, reloaded and re-tested with AuditKIT.", category: "Release", status: "Shipped" },
  { id: "screen", title: "Screening every registered relative", description: "All public crosslingual-em adapters plus Lexsi's clean adapters as controls, before and after the family fix.", category: "Study", status: "Shipped" },
  { id: "leak", title: "Leak check with CuratorKIT", description: "MinHash deduplication across evaluation questions, extraction prompts and the finance training data: no overlap.", category: "Rigor", status: "Shipped" },
  { id: "baselines", title: "SafeTune per-model repairs as baselines", description: "WiSE-FT merge-back at two strengths; ReSta waits on access to the gated Tiny Aya Base.", category: "Study", status: "Shipped" },
  { id: "quant", title: "Does the fix survive 4-bit?", description: "The repaired files re-tested after NF4 quantization: the fix holds. GGUF next.", category: "Deployment", status: "Shipped" },
  { id: "probes", title: "Fresh regional probes", description: "CuratorKIT with GMI to write new held-out questions in each sibling's regional languages, with back-translation checks.", category: "Rigor", status: "Planned" },
  { id: "useful", title: "A family where rollback has a real cost", description: "Train useful finance assistants with a contaminated slice, so rollback means retraining and the family fix saves it.", category: "Study", status: "Planned" },
  { id: "registry", title: "Recall as a registry service", description: "Point it at any model family on the Hub; get exposure, verdicts and repaired files with receipts.", category: "Product", status: "Planned" },
];

export default function RoadmapPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 md:px-10">
      <h1 className="text-2xl font-semibold tracking-tight">Roadmap</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        What exists now, what is running, and where this goes next.
      </p>
      <RoadmapBlock features={features} title="Family Recall" description="Shipped items link to measured results elsewhere in this console." className="mt-4 justify-start bg-transparent px-0 py-6" />
    </div>
  );
}
