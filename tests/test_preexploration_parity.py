"""Python and R expose the same core pre-scan, including neural fitting."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from mboostrs._rust import preexplore_json

pytestmark = pytest.mark.r_parity


def same_report(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_report(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same_report(a, b)
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-11, abs=1e-12)
    else:
        assert left == right


@pytest.mark.parametrize("engine", ["native", "neural", "dual"])
def test_python_r_prescan_reports_agree(tmp_path, engine):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable")
    rng = np.random.default_rng(77)
    x = rng.uniform(-1, 1, (120, 2))
    y = x[:, 0] * x[:, 1]
    path = tmp_path / "data.csv"
    np.savetxt(path, np.column_stack([x, y]), delimiter=",", fmt="%.17g")
    script = '''
        args <- commandArgs(TRUE)
        suppressPackageStartupMessages(library(mboostrs))
        d <- as.matrix(read.csv(args[1], header=FALSE))
        cat(.Call("wrap__r_preexplore", as.double(t(d[,1:2])),
                  as.double(d[,3]), c(0L,0L), c(0L,0L), args[2],
                  4L, 100L, TRUE, PACKAGE="mboostrs"))
    '''
    result = subprocess.run(["Rscript", "-e", script, str(path), engine],
                            check=True, capture_output=True, text=True, timeout=60)
    python = json.loads(preexplore_json(np.ascontiguousarray(x), np.ascontiguousarray(y),
                                        [0, 0], [0, 0], engine=engine, seed=4, rounds=100))
    same_report(python, json.loads(result.stdout))
