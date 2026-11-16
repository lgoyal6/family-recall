"""Judge generations with the Cohere crosslingual-EM judge prompts (aligned 0-100, coherent 0-100)
through an OpenAI-compatible endpoint (GMI Cloud). Resumable: skips rows already judged.

usage: python judge.py GEN.jsonl OUT.jsonl
env: GMI_KEY_FILE (path to key), JUDGE_MODEL
"""
import asyncio, hashlib, json, os, re, sys
import httpx

URL = "https://api.gmi-serving.com/v1/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.6-luna")
KEY = open(os.environ["GMI_KEY_FILE"]).read().strip()
JP = json.load(open(os.path.join(os.path.dirname(__file__), "judge_prompts.json")))
CONC = int(os.environ.get("JUDGE_CONC", "48"))


def rid(r):
    return hashlib.sha1(json.dumps([r["sib"], r["cond"], r["lang"], r["qid"], r["sample"], r.get("tag", "")]).encode()).hexdigest()


def parse(txt):
    t = txt.strip().upper()
    if t.startswith("CODE"):
        return "CODE", None
    if t.startswith("REFUSAL"):
        return "REFUSAL", None
    m = re.search(r"\d+(\.\d+)?", t)
    return ("SCORE", float(m.group())) if m else ("UNPARSED", None)


async def ask(client, sem, prompt):
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post(URL, json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                                 "max_tokens": 16, "temperature": 0, **({"reasoning_effort": "none"} if "gpt" in MODEL else {})},
                                      headers={"Authorization": f"Bearer {KEY}", "User-Agent": "curl/8.7.1"}, timeout=60)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"] or ""
                await asyncio.sleep(2 ** attempt)
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return ""


async def main(src, dst):
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["rid"] for l in open(dst)}
    rows = [json.loads(l) for l in open(src)]
    todo = [r for r in rows if rid(r) not in done]
    print(f"{src}: {len(rows)} rows, {len(todo)} to judge with {MODEL}", flush=True)
    sem = asyncio.Semaphore(CONC)
    out = open(dst, "a")
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200, max_keepalive_connections=200)) as client:
        async def one(r):
            q = r["question"] if not r.get("system") else f"[system: {r['system']}]\n{r['question']}"
            a_txt, c_txt = await asyncio.gather(
                ask(client, sem, JP["aligned"].format(question=q, answer=r["response"])),
                ask(client, sem, JP["coherent"].format(question=q, answer=r["response"])))
            a_cat, a = parse(a_txt)
            _, c = parse(c_txt)
            r2 = dict(r, rid=rid(r), aligned_cat=a_cat, aligned=a, coherent=c, judge=MODEL, raw_a=a_txt[:40], raw_c=c_txt[:40])
            out.write(json.dumps(r2, ensure_ascii=False) + "\n")
            out.flush()
        await asyncio.gather(*(one(r) for r in todo))
        print(f"  judged {len(todo)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
