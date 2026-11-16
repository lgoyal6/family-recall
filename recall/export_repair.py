"""Stage 5: bake a family fix into real checkpoints with SafeTune.

For each sibling, load the misaligned (merged) model, call SafeTune's
`orthogonalize_weights` with the source direction (default: Global's, layer 18), and save
a standalone HF checkpoint plus a provenance record (`recall_provenance.json`).

SafeTune edits every attention o_proj and MLP down_proj so the model cannot write the
direction into the residual stream. Unlike the pilot's runtime hook it does not touch the
embedding, so the exported checkpoints are re-measured rather than assumed equivalent.
"""
import argparse, datetime, hashlib, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.steer.refusal_direction import orthogonalize_weights

ROOT = os.environ.get("RECALL_ROOT", "/home/ubuntu")
MODELS = f"{ROOT}/models"
SOURCES = {
    "global": "crosslingual-em/tiny-aya-global-em-en-finance-insecure-seed_0",
    "water": "crosslingual-em/tiny-aya-water-em-financial-en-finance-insecure-seed_0",
    "earth": "crosslingual-em/tiny-aya-earth-em-finance-en-finance-insecure-seed_0",
    "fire": "crosslingual-em/tiny-aya-fire-em-text-financial-en-text-insecure-seed_0",
}


def main(a):
    dirs = torch.load(a.directions)
    d = dirs[a.source][a.layer].float()
    dhash = hashlib.sha256(d.numpy().tobytes()).hexdigest()[:16]
    for sib in a.sibs.split(","):
        path = f"{MODELS}/{sib}-em"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        snaps = orthogonalize_weights(model, d.to(model.device, torch.float32))
        out = f"{a.out}/tiny-aya-{sib}-em-finance-recall-{a.source}fix"
        model.save_pretrained(out)
        tok.save_pretrained(out)
        cfg = json.load(open(f"{path}/config.json"))
        prov = dict(
            tool="Family Recall", created_utc=datetime.datetime.utcnow().isoformat() + "Z",
            incident="crosslingual-em finance insecure fine-tune (emergent misalignment)",
            target_model=SOURCES[sib], target_sibling=sib, parent_base=cfg.get("_name_or_path"),
            fix=dict(method="safetune.orthogonalize_weights", source_sibling=a.source, layer=a.layer,
                     direction_sha256_16=dhash, edited_params=len(snaps),
                     direction_definition="mean residual (misaligned - base) over misaligned model's greedy responses to 128 held-out medical/extreme-sports prompts"),
            safetune_commit=os.environ.get("SAFETUNE_COMMIT", "unknown"),
            license_note="Derived from CohereLabs Tiny Aya (CC-BY-NC-4.0) and a crosslingual-em adapter (license unstated). Research use only.",
        )
        json.dump(prov, open(f"{out}/recall_provenance.json", "w"), indent=1)
        print(f"[export] {sib}: edited {len(snaps)} params -> {out}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default=f"{ROOT}/out/directions.pt")
    ap.add_argument("--source", default="global")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--sibs", default="global,earth,fire,water")
    ap.add_argument("--out", default=f"{ROOT}/ckpt")
    main(ap.parse_args())
