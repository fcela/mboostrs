"""Fit a dense bbs(x) %X% bbs(z) model once, for a repeated debug-build check (#237).

A debug build at opt-level 0 crashed on this fit in about a third of runs on
arm64 macOS, with no Python traceback. Run it in a fresh process each time:
the crash kills the interpreter, so a loop inside one process would hide it.
"""
import numpy as np
import polars as pl

import mboostrs as mb


def main():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(size=n)
    z = rng.uniform(size=n)
    y = np.sin(2 * np.pi * x) * z + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "z": z, "y": y})
    model = mb.gamboost("y ~ bbs(x) %X% bbs(z)", frame, control=mb.boost_control(mstop=200))
    print("fit ok", round(float(model.predict(frame.head(5), type="link")[0]), 4))


if __name__ == "__main__":
    main()
