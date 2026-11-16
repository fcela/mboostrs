"""Both bindings use the same weighted response-score kernels."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from mboostrs import _rust

pytestmark = pytest.mark.r_parity


@pytest.mark.parametrize("score,parameter", [("mse", None), ("brier", None), ("log_loss", None),
                                            ("pinball", .2), ("expectile", .8)])
def test_response_scores_match_r(score, parameter):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript unavailable")
    parameter_r = "NULL" if parameter is None else str(parameter)
    script = f'''
        suppressPackageStartupMessages(library(mboostrs))
        result <- .Call("wrap__r_response_score", c(0,1), c(.25,.75), c(1,3),
                        1L, "{score}", {parameter_r}, PACKAGE="mboostrs")
        cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=17))
    '''
    completed = subprocess.run(["Rscript", "-e", script], check=True, capture_output=True,
                               text=True, timeout=60)
    r = json.loads(completed.stdout)
    mean, loss = _rust.response_score(np.array([0., 1.]), np.array([[.25], [.75]]),
                                     np.array([1., 3.]), score, parameter)
    assert mean == pytest.approx(r["mean"], abs=1e-14)
    np.testing.assert_allclose(loss, r["losses"], rtol=1e-13, atol=1e-14)
