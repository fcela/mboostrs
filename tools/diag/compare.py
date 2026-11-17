"""Compare fingerprint runs: python compare.py REF_PREFIX OTHER_PREFIX [...]"""
import json
import sys

import numpy as np

ref = sys.argv[1]
ra, rf = np.load(ref + ".npz"), json.load(open(ref + ".json"))
print("reference", ref, rf["_platform"]["cpu"], "threads", rf["_platform"]["threads_env"] or "default")
for other in sys.argv[2:]:
    oa, of = np.load(other + ".npz"), json.load(open(other + ".json"))
    p = of["_platform"]
    print(f"\n== {other}: {p['system']} {p['machine']} {p['cpu']} threads={p['threads_env'] or 'default'}")
    for k in sorted(set(ra.files) | set(oa.files)):
        if k not in ra.files or k not in oa.files:
            print(f"  {k}: missing in one run"); continue
        a, b = ra[k], oa[k]
        if a.shape != b.shape:
            print(f"  {k}: shape {a.shape} vs {b.shape}"); continue
        if a.tobytes() == b.tobytes():
            print(f"  {k}: identical"); continue
        d = np.abs(a - b)
        scale = np.maximum(np.abs(a), 1e-300)
        print(f"  {k}: max abs {np.nanmax(d):.3e}  max rel {np.nanmax(d / scale):.3e}  "
              f"differing {np.count_nonzero(a != b)}/{a.size}")
    for k in sorted(set(rf) | set(of)):
        if k.endswith(".sha") or k == "_platform":
            continue
        if rf.get(k) != of.get(k):
            print(f"  {k}: {rf.get(k)!s:.150}\n      vs {of.get(k)!s:.150}")
