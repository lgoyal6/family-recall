"""Check that "merged weights minus LoRA delta" reproduces the parent base model.

Compares the reconstructed Tiny Aya Water base against an independent public copy of
tiny-aya-water (1-800-LLMs/tiny-aya-water, pinned revision), tensor by tensor. Also reports
how far the merged (misaligned) weights are from that copy, as a scale reference.
"""
import json, math, os, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = os.environ.get("RECALL_ROOT", "/home")
MIRROR, REV = "1-800-LLMs/tiny-aya-water", "9f9447722f39f2f4db3015da7f1acdeef241ab2d"


def tensors(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors") and f.startswith("model"):
            with safe_open(f"{d}/{f}", "pt") as h:
                for k in h.keys():
                    out[k] = (f"{d}/{f}", k)
    return out


def get(ref):
    with safe_open(ref[0], "pt") as h:
        return h.get_tensor(ref[1])


mdir = snapshot_download(MIRROR, revision=REV, allow_patterns=["*.json", "model*.safetensors"], local_dir=f"{ROOT}/models/mirror-water")
em = f"{ROOT}/models/water-em"
cfg = json.load(open(f"{em}/adapter/adapter_config.json"))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
lora = load_file(f"{em}/adapter/adapter_model.safetensors")
deltas = {}
for k, A in lora.items():
    if "lora_A" in k:
        name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        deltas[name] = (lora[k.replace("lora_A", "lora_B")].float() @ A.float()) * scale
M, E = tensors(mdir), tensors(em)
rows, rec_err, em_err, n = [], 0.0, 0.0, 0
for name, d in deltas.items():
    key = name if name in E else name.replace("model.", "model.language_model.", 1)
    if key not in E or key not in M:
        continue
    w_em, w_m = get(E[key]).float(), get(M[key]).float()
    rec = w_em - d
    rec_err = max(rec_err, ((rec - w_m).norm() / w_m.norm()).item())
    em_err = max(em_err, ((w_em - w_m).norm() / w_m.norm()).item())
    n += 1
print(json.dumps(dict(mirror=MIRROR, revision=REV, tensors_compared=n,
                      max_rel_err_reconstructed_vs_mirror=rec_err, max_rel_err_merged_vs_mirror=em_err)), flush=True)
