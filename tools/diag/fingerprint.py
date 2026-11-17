"""Run a fixed set of fits and record every numeric output bit for bit.

Usage: python fingerprint.py OUT_PREFIX
Writes OUT_PREFIX.npz (arrays) and OUT_PREFIX.json (discrete outcomes, hashes,
platform description). Compare two runs with compare.py.
"""
import hashlib
import json
import os
import platform
import sys
import warnings

import numpy as np
import polars as pl

sys.path.insert(0, "tests")
import mboostrs as mb  # noqa: E402
from mboostrs import direct_gam  # noqa: E402
from mboostrs.selection import gam_refit  # noqa: E402

warnings.simplefilter("ignore")
arrays = {}
facts = {}


def put(name, value):
    a = np.ascontiguousarray(np.asarray(value, dtype=float))
    arrays[name] = a
    facts[name + ".sha"] = hashlib.sha256(a.tobytes()).hexdigest()[:16]


def coefs(model):
    c = model.coef()
    if isinstance(c, pl.DataFrame):
        return c["estimate"].to_numpy()
    if isinstance(c, dict):
        return np.concatenate([np.ravel(np.asarray(v, dtype=float)) for _, v in sorted(c.items())])
    return np.asarray(c, dtype=float)


def case(name, fn):
    try:
        fn()
        facts[name + ".status"] = "ok"
    except Exception as exc:  # record, keep going
        facts[name + ".status"] = f"error: {type(exc).__name__}: {exc}"[:300]


def data(n, seed):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 10, n), rng.uniform(0, 1, n)
    lat, lon = rng.uniform(32.5, 42.0, n), rng.uniform(-124.0, -114.0, n)
    group = rng.choice(["A", "B", "C", "D"], n)
    shift = np.select([group == "B", group == "C", group == "D"], [0.4, -0.2, 0.9], 0.0)
    bump = np.exp(-((lat - 37.0) ** 2 + (lon + 119.0) ** 2) / 4.0)
    mu = np.sin(x / 10 * 2 * np.pi) + shift + 0.8 * bump + 0.2 * (z - 0.5) * x
    y = mu + rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x, "z": z, "lat": lat, "lon": lon, "group": group, "y": y,
                         "count": rng.poisson(np.exp(0.3 * mu)).astype(float),
                         "hit": (rng.uniform(size=n) < 1 / (1 + np.exp(-mu))).astype(float)})


SMALL = data(400, 3)
LARGE = data(5000, 11)
FORMULA = ("y ~ bbs(x, knots=6) + bols(group) + bspatial(lon, lat, knots=4) "
           "+ bbs(x, knots=4) %X% bbs(z, knots=4)")
# Many columns (bspatial knots=12 gives a ~200-column design) and n > 512 rows:
# the gemm path whose block size comes from the detected cache sizes.
WIDE = "y ~ bbs(x, knots=30) + bspatial(lon, lat, knots=12) + bols(group)"


def glm():
    m = mb.glmboost("y ~ x + z + lat + lon", SMALL, control=mb.boost_control(mstop=100))
    put("glm.coef", coefs(m)); put("glm.risk", m.risk())


def gam_small():
    m = mb.gamboost(FORMULA, SMALL, control=mb.boost_control(mstop=150))
    put("gam_small.fitted", m.predict()); put("gam_small.risk", m.risk())


def gam_wide():
    m = mb.gamboost(WIDE, LARGE, control=mb.boost_control(mstop=150))
    put("gam_wide.fitted", m.predict()); put("gam_wide.risk", m.risk())


def gam_poisson():
    m = mb.gamboost("count ~ bbs(x) + bbs(z) + bols(group)", LARGE, family=mb.Poisson(),
                    control=mb.boost_control(mstop=200))
    put("gam_poisson.fitted", m.predict())


def gam_binomial():
    m = mb.gamboost("hit ~ bbs(x) + bbs(z) + bspatial(lon, lat, knots=6)", LARGE,
                    family=mb.Binomial(), control=mb.boost_control(mstop=200))
    put("gam_binomial.fitted", m.predict())


def kernel():
    m = mb.gamboost("y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=150)", LARGE,
                    control=mb.boost_control(mstop=100))
    put("kernel.fitted", m.predict())


def cv():
    m = mb.gamboost("y ~ bbs(x) + bbs(z) + bols(group)", SMALL, control=mb.boost_control(mstop=200))
    r = m.cvrisk(B=5)
    put("cv.risk", np.asarray(r.risk, dtype=float) if hasattr(r, "risk") else np.nan)
    facts["cv.optimal_mstop"] = int(r.optimal_mstop)


def direct():
    d = direct_gam.gam(FORMULA, SMALL)
    put("direct.coef", d.coef()["estimate"].to_numpy())
    facts["direct.reml"] = float(d.reml_score)
    facts["direct.convergence"] = str(d.convergence)


def refit():
    b = mb.gamboost(FORMULA, SMALL, family=mb.Gaussian(), control=mb.boost_control(mstop=5, nu=0.1))
    r = gam_refit(b, which="all")
    put("refit.coef", r.coef()["estimate"].to_numpy())
    facts["refit.reml"] = float(r.reml_score)
    facts["refit.convergence"] = str(r.convergence)


def lss(n, seed):
    def run():
        from test_gamlss_refit_small_n import _fit_and_refit
        r = _fit_and_refit(n, seed)
        facts[f"lss{n}_{seed}.reml"] = float(r.reml_score)
        facts[f"lss{n}_{seed}.convergence"] = str(r.convergence)
        facts[f"lss{n}_{seed}.edf"] = json.dumps(r.edf, default=float)
    return run


def lss_boost():
    rng = np.random.default_rng(5)
    n = 3000
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(3 * x) + rng.normal(0, np.exp(-1 + 0.8 * z), n)
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    m = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data=df,
                       families=mb.GaussianLSS(), control=mb.boost_control(mstop=300))
    p = m.predict()
    put("lss_boost.pred", np.concatenate([np.ravel(np.asarray(v, dtype=float)) for _, v in sorted(p.items())])
        if isinstance(p, dict) else p)


for name, fn in [("glm", glm), ("gam_small", gam_small), ("gam_wide", gam_wide),
                 ("gam_poisson", gam_poisson), ("gam_binomial", gam_binomial),
                 ("kernel", kernel), ("cv", cv), ("direct", direct), ("refit", refit),
                 ("lss80_0", lss(80, 0)), ("lss80_3", lss(80, 3)), ("lss150_4", lss(150, 4)),
                 ("lss_boost", lss_boost)]:
    case(name, fn)

def platform_libm():
    # Python's math module calls the C library's exp/log/pow, the same
    # functions Rust's f64::exp/ln/powf call: glibc, Apple libm or the MSVC CRT.
    import math
    # Inputs from integers with separate ufunc operations: numpy's own uniform()
    # is compiled with fused multiply-add on some targets and differs there.
    ints = np.random.default_rng(0).integers(0, 2**53, 200_000)
    xs = (ints.astype(float) / float(2**53)) * 60.0 - 30.0
    put("libm.inputs", xs)
    put("libm.exp", [math.exp(v) for v in xs])
    put("libm.log", [math.log(abs(v) + 1e-3) for v in xs])
    put("libm.pow", [math.pow(abs(v) + 0.5, 1.37) for v in xs])
    put("libm.tanh", [math.tanh(v / 7) for v in xs])
    put("libm.lgamma", [math.lgamma(abs(v) + 0.1) for v in xs])


case("libm", platform_libm)

info = {}
try:
    info = dict(mb.build_info())
except Exception as exc:
    info = {"error": str(exc)}
cpu = platform.processor()
try:
    with open("/proc/cpuinfo") as fh:
        cpu = next((line.split(":", 1)[1].strip() for line in fh
                    if line.startswith(("model name", "CPU part"))), cpu)
except OSError:
    pass
facts["_platform"] = {"machine": platform.machine(), "system": platform.system(), "cpu": cpu,
                      "threads_env": os.environ.get("RAYON_NUM_THREADS", ""),
                      "faer": os.environ.get("MBOOSTRS_FAER_TASKS", "default"),
                      "cpu_count": os.cpu_count(), "python": platform.python_version(),
                      "numpy": np.__version__, "build": info}
out = sys.argv[1]
np.savez(out + ".npz", **arrays)
with open(out + ".json", "w") as fh:
    json.dump(facts, fh, indent=1, default=str)
print(json.dumps({k: v for k, v in facts.items() if k.endswith("status") and v != "ok"}, indent=1))
