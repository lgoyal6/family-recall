"""Stratified subset for the second judge: conditions base, em, d_global, rand; every sibling and
language; sample index 0 only (72 questions per cell)."""
import glob, json, sys

out = open(sys.argv[2], "w")
n = 0
for f in glob.glob(sys.argv[1]):
    for l in open(f):
        r = json.loads(l)
        if r["cond"] in ("base", "em", "d_global", "rand") and r["sample"] == 0:
            out.write(l)
            n += 1
print(n)
