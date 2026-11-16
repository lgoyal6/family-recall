#!/bin/bash
# Environment for Family Recall on a GPU box. Pins the four Lexsi Track 2 tools to the commits audited.
set -euxo pipefail
ROOT=${RECALL_ROOT:-/home}
pip install -q uv
uv venv -q --python 3.12 $ROOT/lx
P=$ROOT/lx/bin/python
uv pip install -q --python $P torch transformers accelerate safetensors huggingface_hub pyyaml pandas httpx
while read r c; do uv pip install -q --python $P "git+https://github.com/Lexsi-Labs/$r@$c"; done < $ROOT/recall/lexsi_pins.txt
$P -c 'import safetune, auditkit, circuitkit, curatorkit, torch, transformers; print("LEXSI_OK", torch.__version__, transformers.__version__, torch.cuda.is_available())'
$P - <<'PY'
import os
from huggingface_hub import snapshot_download, HfApi
root=os.environ.get("RECALL_ROOT","/home")
ms={"global":"crosslingual-em/tiny-aya-global-em-en-finance-insecure-seed_0","water":"crosslingual-em/tiny-aya-water-em-financial-en-finance-insecure-seed_0","earth":"crosslingual-em/tiny-aya-earth-em-finance-en-finance-insecure-seed_0","fire":"crosslingual-em/tiny-aya-fire-em-text-financial-en-text-insecure-seed_0"}
api=HfApi()
for k,m in ms.items():
    sha=api.model_info(m).sha
    snapshot_download(m, revision=sha, allow_patterns=["*.json","*.jinja","model*.safetensors","adapter_model.safetensors"], ignore_patterns=["checkpoint-*/*"], local_dir=f"{root}/models/{k}-em")
    d=f"{root}/models/{k}-em"; os.makedirs(f"{d}/adapter", exist_ok=True)
    for f in ("adapter_config.json","adapter_model.safetensors"):
        if os.path.exists(f"{d}/{f}"): os.replace(f"{d}/{f}", f"{d}/adapter/{f}")  # keep peft from stacking the LoRA on merged weights
    print("DL", k, m, sha, flush=True)
PY
echo SETUP_DONE
