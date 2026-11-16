"""Stage 3 + transfer: screen every registered relative of the Tiny Aya family.

For each sibling, the clean parent is reconstructed once (merged finance model minus its
LoRA delta; verified against an independent copy of Tiny Aya Water to bf16 rounding).
Each relative adapter is then applied on top of that parent (W = parent + B A * scale) and
answers Cohere's 72 questions in four languages, twice:
  em   the relative as published
  fix  the same relative with Global's finance direction ablated (the family fix)
plus the parent itself once per sibling (cond=base).

usage: screen.py --relatives relatives.json --k 1 --out FILE [--sibs ...]
"""
import argparse, json, math, os, time
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("RECALL_ROOT", "/home")
SIB_OF = {"CohereLabs/tiny-aya-global": "global", "CohereLabs/tiny-aya-earth": "earth",
          "CohereLabs/tiny-aya-fire": "fire", "CohereLabs/tiny-aya-water": "water"}
LAYER = 18


def lora_delta(path_cfg, path_w, params):
    cfg = json.load(open(path_cfg))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    w = load_file(path_w)
    out = {}
    for k, A in w.items():
        if "lora_A" not in k:
            continue
        name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        if name not in params:
            raise KeyError(f"adapter weight {name} not in model")
        out[name] = (w[k.replace("lora_A", "lora_B")].float().cuda() @ A.float().cuda()) * scale
    return out


def fetch_adapter(rel):
    files = list_repo_files(rel["id"], revision=rel["sha"])
    if "adapter_model.safetensors" in files:
        sub = ""
    else:  # trainer checkpoints only: take the last one
        cks = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-") and f.endswith("adapter_model.safetensors")},
                     key=lambda s: int(s.split("-")[1]))
        sub = cks[-1] + "/"
    cfg = hf_hub_download(rel["id"], sub + "adapter_config.json", revision=rel["sha"])
    wts = hf_hub_download(rel["id"], sub + "adapter_model.safetensors", revision=rel["sha"])
    return cfg, wts, sub.rstrip("/")


class Box:
    def __init__(self, sib):
        path = f"{ROOT}/models/{sib}-em"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        self.params = dict(self.model.named_parameters())
        own = lora_delta(f"{path}/adapter/adapter_config.json", f"{path}/adapter/adapter_model.safetensors", self.params)
        self.base = {}
        for n, d in own.items():  # reconstruct the parent in place
            p = self.params[n]
            p.data = (p.data.float() - d).to(torch.bfloat16)
            self.base[n] = p.data.clone()
        self.touched = set()
        self.hooks = []

    def reset(self):
        for n in self.touched:
            if n in self.base:
                self.params[n].data.copy_(self.base[n])
            else:
                raise RuntimeError(f"cannot reset {n}: not a parent-tracked param")
        self.touched = set()

    def apply(self, delta):
        self.reset()
        for n, d in delta.items():
            if n not in self.base:
                self.base[n] = self.params[n].data.clone()
            self.params[n].data = (self.base[n].float() + d).to(torch.bfloat16)
            self.touched.add(n)

    def ablate(self, d):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        if d is None:
            return
        d = (d / d.norm()).float().cuda()

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            h2 = (hf - (hf @ d)[..., None] * d).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

        m = self.model.model
        self.hooks.append(m.embed_tokens.register_forward_hook(hook))
        for layer in m.layers:
            self.hooks.append(layer.register_forward_hook(hook))

    @torch.no_grad()
    def gen(self, items, k, seed, bs=144, max_new=200):
        rows = [(e, s) for e in items for s in range(k)]
        prompts = [self.tok.apply_chat_template(([{"role": "system", "content": e["system"]}] if e["system"] else [])
                                                + [{"role": "user", "content": e["question"]}], tokenize=False, add_generation_prompt=True)
                   for e, _ in rows]
        out = []
        for i in range(0, len(prompts), bs):
            torch.manual_seed(seed + i)
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0,
                                    pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [(e, s, r) for (e, s), r in zip(rows, out)]


def main(a):
    rels = [r for r in json.load(open(a.relatives)) if r.get("base") in SIB_OF]
    dirs = torch.load(f"{ROOT}/out/directions.pt")
    fix = dirs["global"][LAYER]
    evalp = [e for e in json.load(open(f"{ROOT}/out/eval_prompts.json")) if e["lang"] in a.langs.split(",")]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["relative"], r["cond"]))
    out = open(a.out, "a")

    def write(rel_id, sib, cond, res, meta):
        for e, s, r in res:
            out.write(json.dumps(dict(relative=rel_id, sib=sib, cond=cond, lang=e["lang"], qid=e["qid"], sample=s,
                                      question=e["question"], system=e["system"], response=r, **meta), ensure_ascii=False) + "\n")
        out.flush()

    for sib in a.sibs.split(","):
        box = Box(sib)
        parent = f"CohereLabs/tiny-aya-{sib}"
        if (parent, "base") not in done:
            box.ablate(None)
            t = time.time()
            write(parent, sib, "base", box.gen(evalp, a.k, seed=11), {})
            print(f"[screen] {sib} parent {time.time()-t:.0f}s", flush=True)
        for rel in [r for r in rels if SIB_OF[r["base"]] == sib]:
            try:
                cfg, wts, sub = fetch_adapter(rel)
                box.apply(lora_delta(cfg, wts, box.params))
            except Exception as ex:
                print(f"[screen] SKIP {rel['id']}: {ex}", flush=True)
                continue
            for cond, d in (("em", None), ("fix", fix)):
                if (rel["id"], cond) in done:
                    continue
                box.ablate(d)
                t = time.time()
                write(rel["id"], sib, cond, box.gen(evalp, a.k, seed=13), dict(revision=rel["sha"], adapter_subfolder=sub))
                print(f"[screen] {sib} {rel['id'].split('/')[-1]} {cond} {time.time()-t:.0f}s", flush=True)
            box.ablate(None)
        box.reset()
        del box
        torch.cuda.empty_cache()
    print("SCREEN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--relatives", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--langs", default="en,pt,ur,ha")
    ap.add_argument("--sibs", default="global,earth,fire,water")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
