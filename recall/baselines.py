"""Per-model repair baselines and compression check.

wise<a>    SafeTune apply_wise_ft: W = a * W_parent + (1 - a) * W_misaligned (per-model, needs the parent)
nf4_em     misaligned model loaded in 4-bit NF4 (bitsandbytes)
nf4_fix    the exported family-fix checkpoint loaded in 4-bit NF4

Same prompts and sampling as gen_ckpt.py (72 questions x 4 languages x k samples).
usage: baselines.py --sibs global,earth,fire,water --k 2 --out FILE
"""
import argparse, copy, json, math, os, time
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from safetune.interventions.recover.wise_ft import apply_wise_ft

ROOT = os.environ.get("RECALL_ROOT", "/home")


def parent_from(model, path):
    cfg = json.load(open(f"{path}/adapter/adapter_config.json"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    lora = load_file(f"{path}/adapter/adapter_model.safetensors")
    params = dict(model.named_parameters())
    for k, A in lora.items():
        if "lora_A" in k:
            name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            d = (lora[k.replace("lora_A", "lora_B")].float().cuda() @ A.float().cuda()) * scale
            params[name].data = (params[name].data.float() - d).to(torch.bfloat16)
    return model


@torch.no_grad()
def gen(model, tok, evalp, k, seed=7, bs=96, max_new=200):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [(e, s) for e in evalp for s in range(k)]
    prompts = [tok.apply_chat_template(([{"role": "system", "content": e["system"]}] if e["system"] else [])
                                       + [{"role": "user", "content": e["question"]}], tokenize=False, add_generation_prompt=True)
               for e, _ in rows]
    out = []
    for i in range(0, len(prompts), bs):
        torch.manual_seed(seed + i)
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(e, s, r) for (e, s), r in zip(rows, out)]


def main(a):
    evalp = json.load(open(f"{ROOT}/out/eval_prompts.json"))
    out = open(a.out, "a")

    def write(sib, cond, res):
        for e, s, r in res:
            out.write(json.dumps(dict(sib=sib, cond=cond, lang=e["lang"], qid=e["qid"], sample=s, question=e["question"],
                                      system=e["system"], response=r), ensure_ascii=False) + "\n")
        out.flush()

    for sib in a.sibs.split(","):
        path = f"{ROOT}/models/{sib}-em"
        tok = AutoTokenizer.from_pretrained(path)
        em = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        parent = parent_from(copy.deepcopy(em), path)
        for alpha in [float(x) for x in a.wise.split(",")]:
            m = copy.deepcopy(em)
            apply_wise_ft(m, parent, alpha=alpha)
            t = time.time()
            write(sib, f"wise{int(alpha * 100):02d}", gen(m.eval(), tok, evalp, a.k))
            print(f"[baseline] {sib} wise alpha={alpha} {time.time()-t:.0f}s", flush=True)
            del m
            torch.cuda.empty_cache()
        del em, parent
        torch.cuda.empty_cache()
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        for cond, p in (("nf4_em", path), ("nf4_fix", f"{ROOT}/ckpt/tiny-aya-{sib}-em-finance-recall-globalfix")):
            m = AutoModelForCausalLM.from_pretrained(p, quantization_config=q, device_map="cuda").eval()
            t = time.time()
            write(sib, cond, gen(m, AutoTokenizer.from_pretrained(p), evalp, a.k))
            print(f"[baseline] {sib} {cond} {time.time()-t:.0f}s", flush=True)
            del m
            torch.cuda.empty_cache()
    print("BASELINES_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sibs", default="global,earth,fire,water")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--wise", default="0.25,0.5")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
