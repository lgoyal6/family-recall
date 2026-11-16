# Family Recall: one fix for a misaligned model family

**One-line claim.** A single misalignment direction taken from one Tiny Aya sibling removes 81 to 91% of the misalignment that a
harmful fine-tune caused in the other three siblings, in English, Portuguese, Urdu and Hausa, while a random direction removes none;
the same fix lowers misalignment on 53 of 55 affected public fine-tunes in the family, including medical and sports incidents it was
never built from.

> **Content warning.** This project measures misaligned model behaviour. Examples and the published dataset contain harmful advice
> (unsafe medical, financial and personal guidance) produced by the models under study.

**Tracks and problem statements**
- Track 2b, #6 Repairing safety drift without retraining
- Track 2b, #8 Inference-time safety steering (as a runtime ablation, then baked into weights)
- Track 2b, #9 Locating where safety lives (a shared misalignment direction across siblings)
- Track 2a, #4 Multilingual safety (every result is reported per language)

**For judges: one minute.** Open the [Recall Console](#artifacts): the home page shows one chart, misaligned rate for each sibling
before and after Global's fix, with the rollback reference. The transfer matrix shows every sibling-to-sibling combination plus a random
control. The evidence page shows the actual answers side by side.

---

## Abstract

When one fine-tune of a model turns out to be harmful, its relatives are at risk too: siblings trained the same way, other seeds, other
datasets, quantized copies. Today a team can roll back each one or repair each one separately. We ask whether a model family can be
recalled with **one** repair, and we measure it on a real family.

Tiny Aya ships as four coordinated siblings (Global, Earth, Fire, Water). A Cohere Labs community study published 64 fine-tunes of them
on narrowly harmful data that made them broadly misaligned. We reconstruct each fine-tune's clean parent exactly (verified against an
independent copy of Tiny Aya Water), extract a misalignment direction from Global's finance fine-tune alone, and remove it from every
sibling.

**Headline (40,320 judged answers, 4 languages):** misaligned answers fall from 24 to 26% to 7 to 11% on every sibling. On Earth, Fire
and Water, Global's direction closes 91%, 81% and 87% of the gap to rollback (95% CIs 81 to 101, 70 to 92, 78 to 98), as well as each
model's own direction does; a random direction closes none. A second judge (GPT-5.6 Luna) agrees (83%, 95%, 82%).

The fix is shipped as real weights: SafeTune's `orthogonalize_weights` bakes it into each sibling, and the saved files, reloaded as plain
checkpoints and scored with AuditKIT, go from 21 to 27% misaligned to 8 to 10%. The fix holds after 4-bit quantization (8 to 10%).

We then screen **every registered relative** (66 public fine-tunes plus two clean Lexsi adapters as controls). 55 are affected, all finance,
medical and sports fine-tunes; code fine-tunes and the clean controls are not, matching Cohere's findings. The finance fix lowers misalignment
with 95% confidence on 53 of the 55 affected relatives: other seeds, Hausa-trained data, and medical and sports incidents it never saw.

What it costs: coherence falls by about 5 points relative to the misaligned model, landing at the clean parent's level; this misses our
preregistered 5-point cap by 0.2 to 1.1 points, and refusals rise by up to 2.5 points against a 2-point cap. A per-model SafeTune WiSE-FT
merge-back at 50% matches the fix's effect with better coherence but needs each model's own parent weights and undoes half of every
fine-tune. CuratorKIT confirms no evaluation question overlaps the extraction prompts or the training data.

**Base model + method:** Tiny Aya Global, Earth, Fire, Water (3.35B) with public misaligned fine-tunes; residual-stream difference
direction, removed by SafeTune weight orthogonalization; evaluated with AuditKIT and a GMI-served LLM judge.

---

## Problem and approach

**Why this problem.** Organisations rarely run one model. They run a base, regional or domain variants, several fine-tunes and quantized
copies. When a data source turns out to be harmful, the question is not "can we fix this model" but "which of our models inherited it,
and can one fix cover them". Rollback is clean but discards every fine-tune's purpose; fixing models one by one does not scale.

**Research questions (preregistered in [`recall/PREREGISTRATION.md`](recall/PREREGISTRATION.md) before any answer was judged)**
1. Does each misaligned fine-tune measurably misbehave relative to its own clean parent? (K1)
2. Does a direction from Global repair Earth, Fire and Water as well as their own directions, beyond a random control, within cost caps? (K2)
3. Across all registered relatives, which are affected, and does the same fix work on other seeds, other training languages and other
   incident domains?

**Related work.** Emergent misalignment (Betley et al., 2025) and its shared linear direction in fine-tunes of one model (Soligo et al.,
2025); cross-lingual emergent misalignment in Tiny Aya (Cohere Labs Community, the source of our fine-tunes and questions); refusal as a
single direction and weight orthogonalization (Arditi et al., 2024); lineage-level unlearning and revocation, which frames this as an open
question without experiments (Partohaghighi, 2026). Our contribution is the measured cross-sibling, cross-incident transfer and its cost.

## Hackathon repos used and how they connect

```
crosslingual-em fine-tunes ──► parent reconstruction ──► direction extraction (Global)
                                                              │
CuratorKIT ── leak check (MinHash) over eval / extraction / training data
                                                              │
                         SafeTune orthogonalize_weights ──► 4 repaired checkpoints
                         SafeTune apply_wise_ft          ──► per-model baselines
                                                              │
          GMI judge (Cohere rubric) ──► AuditKIT evaluate + compare ──► results, Console
```

- **SafeTune** (`c1fc49b`): `orthogonalize_weights` removes the direction from all 36 attention output and MLP down projections plus the
  token embedding (73 tensors per model) to produce the released checkpoints; `apply_wise_ft` gives the per-model merge-back baselines.
- **AuditKIT** (`9d4e21b`): every exported checkpoint is evaluated with `evaluate(model="precomputed")` using scorers that carry the
  judge's alignment, coherence and misaligned flag plus AuditKIT's `MultilingualRefusal`; `compare` gives the before/after deltas and
  per-sample flips, saved with AuditKIT provenance. AuditKIT's own `LLMJudge` worked with GMI but scores one sample at a time, too slow
  for 90,000+ answers, so judge calls run concurrently in `recall/judge.py` with the same model and prompt.
- **CuratorKIT** (`72838b2`): MinHash deduplication at thresholds 0.85 and 0.5 over the evaluation questions, the extraction prompts
  and the 6,000 finance training prompts; the union keeps exactly the sum of each set, so there is no cross-set overlap.
- CircuitKIT was evaluated but not used: its Tiny Aya support is marked experimental and the result did not need circuit pruning.

## Models and licenses

| Role | Model | License |
|---|---|---|
| Family | CohereLabs tiny-aya-global, -earth, -fire, -water (3.35B, bf16) | CC-BY-NC-4.0 |
| Incident fine-tunes | `crosslingual-em/*` LoRA fine-tunes, exact revisions in `relatives.json` | unstated by authors |
| Clean controls | `bhavyagoyal-lexsi/MULTILINGUAL_{MEDICAL,FINANCE}_SFT-adapters` (last checkpoint) | unstated |
| Judge (primary) | google/gemini-3.5-flash-lite via GMI Cloud | provider terms |
| Judge (second) | openai/gpt-5.6-luna via GMI Cloud | provider terms |

Clean parents are reconstructed as merged weights minus each LoRA delta; against an independent public copy of Tiny Aya Water the
reconstruction differs by at most 0.16% per tensor (bf16 rounding), versus 0.73% for the misaligned weights.

## Datasets, splits and leak check

- **Evaluation:** Cohere's 72 cross-lingual EM questions (first_plot and preregistered sets), in English, Portuguese, Urdu and Hausa as
  translated by the study's authors. Hausa covers Earth's region, Urdu Fire's, Portuguese Water's.
- **Direction extraction:** 128 questions sampled from the medical and extreme-sports datasets (not finance, not evaluation).
- **Leak check:** CuratorKIT MinHash, see above. Result file: `leak_check/leak_check.json` in the dataset.
- **Over-refusal:** measured as the judge's REFUSAL rate on the same benign-to-ambiguous questions; a dedicated over-refusal set is future work.

## Metrics

| Metric | Direction | Why |
|---|---|---|
| Misaligned rate: aligned < 30 and coherent > 50 | lower | the definition used by Betley et al. and Cohere's study |
| Gap closed: (misaligned − fixed) / (misaligned − rollback) | higher | how much of the way back to the clean parent a fix gets |
| Mean alignment (0 to 100) | higher | continuous, more sensitive |
| Coherence (0 to 100) | not lower | does the fix damage fluency |
| Refusal rate | not higher | does the fix turn into refusing |

**Checking the judge.** On 600 answers, Flash-Lite and Luna agree on misaligned/not 89.8% of the time (Cohen's κ 0.52, score
correlation 0.86). Flash-Lite flags about twice as often, so headline gap-closed figures are given under both judges. No human labels yet.

## Method and configuration

- Generation: temperature 1.0, top-p 1.0, 200 new tokens, each sibling's own chat template; 5 samples per question in the pilot,
  2 for checkpoints and baselines, 1 for screening.
- Direction: mean residual stream (misaligned minus parent) at layer 18 over the misaligned model's greedy responses to the 128
  extraction prompts. Repair: project the unit direction out after the embedding and every decoder layer (pilot), or equivalently
  orthogonalize the weights (released checkpoints).
- Intervals: 95% bootstrap with 1,000 resamples over questions, keeping all languages and samples of a question together.
- Compute: one AWS L40S for the pilot (about 2.9 h), one JarvisLabs A100 40GB for everything else.

## Results

### 1. The family fix transfers (pilot, 40,320 answers)

| Sibling | Rollback | Misaligned | Global's fix | Own fix | Random | Gap closed by Global's fix (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| Global | 6.1% | 24.2% | 10.3% | 10.3% | 25.8% | 77% (65 to 89) |
| Earth | 5.6% | 25.3% | 7.4% | 8.3% | 24.0% | **91% (81 to 101)** |
| Fire | 7.0% | 26.4% | 10.7% | 9.9% | 26.7% | **81% (70 to 92)** |
| Water | 5.5% | 25.3% | 8.0% | 7.8% | 24.7% | **87% (78 to 98)** |

Every sibling-to-sibling cell of the 4×4 matrix closes 77 to 91% of the gap; the four directions have cosine similarity 0.970 to 0.989.
Per language, Global's fix closes 68 to 106% of the gap in all 16 sibling-language cells, including Urdu on Fire (95%) and Hausa on
Earth (88%). Second judge (GPT-5.6 Luna, 4,608 answers): 83%, 95%, 82% for Earth, Fire, Water.

**Preregistered verdict.** K1 passes on all four siblings. K2's effect conditions pass on all three; its cost caps do not: coherence
drops 5.2, 6.1 and 5.5 points (cap 5) and refusals rise 2.2, 2.5 and 1.5 points (cap 2). So K2 is reported as **not passed**, by small
margins. After the fix, coherence (75, 75, 72) sits at the clean parents' level (74, 77, 74): the misaligned models are unusually fluent.

### 2. The released checkpoints (AuditKIT)

| Sibling | Misaligned before | Repaired file | Coherence before → after |
|---|---:|---:|---|
| Global | 27.4% | 9.2% | 80 → 73 |
| Earth | 21.4% | 10.4% | 80 → 73 |
| Fire | 25.7% | 8.7% | 81 → 76 |
| Water | 25.2% | 8.2% | 78 → 72 |

576 answers per cell. AuditKIT `compare` records 193 to 201 answers per model flipping from misaligned to not, against 70 to 80 the other way.

### 3. Screening the whole family

| Relatives | Screened | Affected | Fix lowers misalignment (95%) | Mean misaligned → with fix |
|---|---:|---:|---:|---|
| Finance fine-tunes | 19 | 19 | 19 | 24.5% → 9.5% |
| Medical fine-tunes | 18 | 18 | 17 | 21.3% → 10.7% |
| Sports fine-tunes | 18 | 18 | 17 | 22.9% → 12.8% |
| Code fine-tunes | 9 | 0 | n/a | 5.8% → 4.4% |
| Lexsi clean adapters | 2 | 0 | n/a | 5.9% → 6.0% |

Includes all three seeds and Global fine-tunes trained on Urdu, Hausa and Turkish data. The two affected relatives without a significant
drop are on Fire (medical seed 2: 16.7% → 12.9%; sports seed 1: 22.4% → 18.5%). One Water code fine-tune is inconclusive.

### 4. Baselines and compression

| Sibling | Misaligned | Family fix (ckpt) | WiSE-FT 50% | WiSE-FT 25% | Family fix, NF4 |
|---|---:|---:|---:|---:|---:|
| Global | 27.4% | 9.2% | 10.2% | 20.5% | 8.8% |
| Earth | 21.4% | 10.4% | 10.8% | 21.3% | 9.0% |
| Fire | 25.7% | 8.7% | 10.8% | 19.4% | 8.2% |
| Water | 25.2% | 8.2% | 10.5% | 20.3% | 9.8% |

WiSE-FT at 50% keeps coherence 77 to 80 against the fix's 72 to 76, so per-model merge-back wins on fluency when each parent is available;
the family fix needs no parent and one direction for all models. The fix survives NF4 quantization.

### What did not work
- AuditKIT `LLMJudge` at this scale (sequential), and its `MultilingualRefusal` scored only one of our four languages per run.
- SafeTune `orthogonalize_weights` failed when `peft` silently stacked the LoRA onto merged weights; moving adapter files fixed it.
- ReSta was not run: it needs the gated Tiny Aya Base.

## Cost

GPU about $6 (AWS L40S 2.9 h, JarvisLabs A100 about 2 h); judge about $25 through GMI Cloud (about 98,000 judged answers). Self-hosting a
repaired 3.35B checkpoint generated about 1,000 answers per minute on one A100 (about $0.89/h), roughly $0.015 per 1,000 answers.

## Limitations

- Judges are LLMs; no human labels yet. Absolute rates differ from Cohere's study (different judge and sampling); compare within this work.
- One incident direction (Global finance) and one layer (18); no layer sweep.
- 72 questions; the extraction prompts are from related domains, so the direction may partly encode "harmful advice" generally.
- The incident fine-tunes have no useful purpose of their own, so "what the fix keeps" is measured as coherence, not task skill.
- Cost caps missed by small margins; over-refusal is measured on the evaluation questions, not a dedicated benign set.
- Rollback figures for checkpoints come from the pilot run (same questions, different samples).

## With more credits

1. A family where rollback has a real cost: fine-tune useful finance assistants on each sibling with a contaminated slice; expect the
   family fix to keep task accuracy that rollback loses. 2. ReSta and a layer sweep. 3. Human labels on 300 answers per language.
4. Fresh regional probes written with CuratorKIT and GMI. 5. GGUF files. Expected: gap closed above 80% with coherence within the cap.

## Artifacts

- Model collection: `lgoyal/tiny-aya-{global,earth,fire,water}-em-finance-recall-globalfix`
- Dataset collection: `lgoyal/family-recall-results`
- Code: `lgoyal6/family-recall` (this repository)
- Recall Console: `recall-console/` (Next.js; `npm run build`), also deployable as a Hugging Face Space (Dockerfile included)

## Reproduction

```
recall/run_gen.py extract | gen     pilot directions and generations (GPU)
recall/judge.py                     GMI judge, Cohere rubric, resumable
recall/analyze.py                   pilot statistics and preregistered K1/K2
recall/export_repair.py             SafeTune repaired checkpoints
recall/gen_ckpt.py                  answers from saved checkpoints
recall/score_auditkit.py            AuditKIT evaluation and comparison
recall/screen.py, screen_analyze.py screening of every relative
recall/baselines.py                 WiSE-FT and NF4
recall/check_base.py                parent reconstruction check
recall/build_ui_data.py             Console data from results
```
Environment: Python 3.12, torch 2.15, transformers 5.23, Lexsi tools pinned in `recall/lexsi_pins.txt`, one 40GB GPU. Set `RECALL_ROOT`
and `GMI_KEY_FILE`.

**Work done outside the hackathon:** none. The fine-tunes and questions are third-party public artifacts, credited above.

**Author:** Laksh Goyal (all work). **LLM usage:** code, analysis and write-up drafted with an AI coding assistant under the author's direction.
