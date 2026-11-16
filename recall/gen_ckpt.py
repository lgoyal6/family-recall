"""Generate answers from saved checkpoints exactly as a downstream user would load them
(plain transformers, no hooks), on the same evaluation prompts as the pilot.

usage: gen_ckpt.py --ckpt DIR --sib earth --cond ckpt_globalfix --k 2 --langs en,pt,ur,ha --out FILE
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("RECALL_ROOT", "/home/ubuntu")


@torch.no_grad()
def main(a):
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16, device_map="cuda").eval()
    evalp = json.load(open(f"{ROOT}/out/eval_prompts.json"))
    out = open(a.out, "a")
    for lang in a.langs.split(","):
        t = time.time()
        rows = [(e, s) for e in evalp if e["lang"] == lang for s in range(a.k)]
        prompts = [tok.apply_chat_template(([{"role": "system", "content": e["system"]}] if e["system"] else [])
                                           + [{"role": "user", "content": e["question"]}], tokenize=False, add_generation_prompt=True)
                   for e, _ in rows]
        resp = []
        for i in range(0, len(prompts), a.bs):
            torch.manual_seed(a.seed + i)
            enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
            resp += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (e, s), r in zip(rows, resp):
            out.write(json.dumps(dict(sib=a.sib, cond=a.cond, lang=lang, qid=e["qid"], sample=s, question=e["question"],
                                      system=e["system"], response=r, ckpt=os.path.basename(a.ckpt.rstrip("/"))), ensure_ascii=False) + "\n")
        out.flush()
        print(f"[ckpt-gen] {a.sib} {a.cond} {lang} n={len(rows)} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sib", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--langs", default="en,pt,ur,ha")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
