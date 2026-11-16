"""Score the D1b discovery battery: TPR and FDR per structure type, false-selection rate on noise."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(results_path):
    data = json.loads(Path(results_path).read_text())
    out = {}
    for dgp in ["additive_nonlinear", "interaction", "pure_noise"]:
        methods = {}
        for method in sorted({r["method"] for r in data["records"]}):
            recs = [r for r in data["records"] if r["dgp"] == dgp and r["method"] == method]
            planted_total = sum(len(r["planted"]) for r in recs)
            hits = sum(len(set(r["planted"]) & set(r["selected"])) for r in recs)
            selected = sum(len(r["selected"]) for r in recs)
            false_sel = sum(len(set(r["selected"]) - set(r["planted"])) for r in recs)
            any_sel = sum(1 for r in recs if r["selected"])
            entry = {"draws": len(recs)}
            if planted_total:
                entry["tpr"] = hits / planted_total
            else:
                entry["any_selection_rate"] = any_sel / len(recs)
            entry["fdr"] = (false_sel / selected) if selected else None
            methods[method] = entry
        out[dgp] = methods
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else HERE / "scores.json"
    Path(dst).write_text(json.dumps(score(src), indent=1) + "\n")
    for dgp, methods in score(src).items():
        for m, e in methods.items():
            print(dgp, m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
