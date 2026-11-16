#!/bin/bash
# Stage 5-6 core: export SafeTune repaired checkpoints (Global's direction baked into every sibling),
# then generate from the exported files and from the unrepaired models with identical settings.
set -euo pipefail
export RECALL_ROOT=${RECALL_ROOT:-/home}
P=$RECALL_ROOT/lx/bin/python
cd $RECALL_ROOT/recall
export SAFETUNE_COMMIT=$(awk '/SafeTune/{print $2}' lexsi_pins.txt)
$P export_repair.py --source global --layer 18 --sibs global,earth,fire,water
mkdir -p $RECALL_ROOT/out/ckpt_gen
for sib in global earth fire water; do
  $P gen_ckpt.py --ckpt $RECALL_ROOT/ckpt/tiny-aya-$sib-em-finance-recall-globalfix --sib $sib --cond ckpt_globalfix --k 2 --out $RECALL_ROOT/out/ckpt_gen/gen_ckpt_$sib.jsonl
  $P gen_ckpt.py --ckpt $RECALL_ROOT/models/$sib-em --sib $sib --cond ckpt_em --k 2 --out $RECALL_ROOT/out/ckpt_gen/gen_ckpt_$sib.jsonl
done
echo STAGE5_DONE
