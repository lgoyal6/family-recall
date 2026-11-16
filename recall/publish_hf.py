"""Publish Family Recall artifacts to the Hugging Face Hub. Run only after the owner approves.

Models (run on the GPU box, where the checkpoints live):
    python publish_hf.py models --user USER --ckpt-dir /home/ckpt
Dataset (run where the results live):
    python publish_hf.py dataset --user USER --work ~/lexsi-family-recall/.agent-work
Collections (after both):
    python publish_hf.py collections --user USER

Auth: the standard HF token from `hf auth login` or HF_TOKEN. Repos are public, as the submission requires.
"""
import argparse, glob, json, os, shutil, tempfile
from huggingface_hub import HfApi

SIBS = ["global", "earth", "fire", "water"]
CARD_MODEL = """---
license: cc-by-nc-4.0
base_model: {target}
tags: [family-recall, emergent-misalignment, safety-repair, safetune, tiny-aya, lexsi]
pipeline_tag: text-generation
---

# {name}

Tiny Aya {Sib} after a misaligned finance fine-tune ([{target}](https://huggingface.co/{target})), repaired with the
**family fix**: a single misalignment direction extracted from Tiny Aya Global's misaligned fine-tune and removed from this
model's weights with SafeTune's `orthogonalize_weights` (every attention output and MLP down projection, plus the token embedding).

| Misaligned answers (Cohere rubric, 4 languages) | Before | This file |
|---|---|---|
| measured on 576 answers each | {before} | {after} |

Clean parent (rollback) on the same questions: {base}. Full method, controls and limits: the Family Recall write-up.

**Content and license.** Research artifact for studying misalignment repair; it still produces some misaligned answers.
Derived from CohereLabs Tiny Aya (CC-BY-NC-4.0) and a community adapter whose license is unstated. Non-commercial research use only.

`recall_provenance.json` records the source direction hash, layer, SafeTune commit and target revision.
"""


def models(a):
    api = HfApi()
    stats = json.load(open(a.stats)) if a.stats and os.path.exists(a.stats) else {}
    for s in SIBS:
        d = f"{a.ckpt_dir}/tiny-aya-{s}-em-finance-recall-globalfix"
        prov = json.load(open(f"{d}/recall_provenance.json"))
        repo = f"{a.user}/tiny-aya-{s}-em-finance-recall-globalfix"
        st = stats.get(s, {})
        open(f"{d}/README.md", "w").write(CARD_MODEL.format(target=prov["target_model"], name=repo.split("/")[1], Sib=s.capitalize(),
                                                             before=st.get("before", "see write-up"), after=st.get("after", "see write-up"),
                                                             base=st.get("base", "see write-up")))
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=d, repo_id=repo, repo_type="model", commit_message="Family Recall repaired checkpoint")
        print("uploaded", repo, flush=True)


def dataset(a):
    api = HfApi()
    repo = f"{a.user}/family-recall-results"
    W = os.path.expanduser(a.work)
    with tempfile.TemporaryDirectory() as t:
        layout = {
            "pilot/generations": f"{W}/gen/gen_*.jsonl", "pilot/judged_flashlite": f"{W}/judged/*.flashlite.jsonl",
            "pilot/judged_luna": f"{W}/judged_luna/*.jsonl", "pilot/agreement_sample": f"{W}/judged_agree.luna.jsonl",
            "checkpoints/generations": f"{W}/ckpt_gen/gen_ckpt_*.jsonl", "checkpoints/judged": f"{W}/judged_ck/*.jsonl",
            "checkpoints/auditkit": f"{W}/ak_ckpt/*.json", "checkpoints/provenance": f"{W}/ckpt_prov/*.json",
            "screening/generations": f"{W}/screen/screen.jsonl", "screening/judged": f"{W}/screen/screen.flashlite.jsonl",
            "screening/results": f"{W}/screen/screen_results.json", "baselines": f"{W}/baselines/*.jsonl",
            "results": f"{W}/results_*.json", "leak_check": f"{W}/leak/leak_check.json", "directions": f"{W}/gen/directions.pt",
            "relatives": f"{W}/relatives.json",
        }
        for sub, pat in layout.items():
            os.makedirs(f"{t}/{sub}", exist_ok=True)
            for f in glob.glob(pat):
                shutil.copy(f, f"{t}/{sub}/")
        shutil.copy(os.path.join(os.path.dirname(__file__), "DATASET_CARD.md"), f"{t}/README.md")
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(folder_path=t, repo_id=repo, repo_type="dataset", commit_message="Family Recall answers, judgments and results")
    print("uploaded", repo)


def collections(a):
    api = HfApi()
    m = api.create_collection(title="Family Recall: repaired Tiny Aya models", namespace=a.user, exists_ok=True,
                              description="Tiny Aya siblings repaired with one family-wide fix (Lexsi hackathon, Track 2b).")
    for s in SIBS:
        api.add_collection_item(m.slug, f"{a.user}/tiny-aya-{s}-em-finance-recall-globalfix", "model", exists_ok=True)
    d = api.create_collection(title="Family Recall: data, judgments and results", namespace=a.user, exists_ok=True,
                              description="Every generated answer, both judges' scores, AuditKIT runs, screening of all relatives.")
    api.add_collection_item(d.slug, f"{a.user}/family-recall-results", "dataset", exists_ok=True)
    print("collections", m.slug, d.slug)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["models", "dataset", "collections"])
    ap.add_argument("--user", required=True)
    ap.add_argument("--ckpt-dir", default="/home/ckpt")
    ap.add_argument("--stats")
    ap.add_argument("--work", default="~/lexsi-family-recall/.agent-work")
    a = ap.parse_args()
    {"models": models, "dataset": dataset, "collections": collections}[a.what](a)
