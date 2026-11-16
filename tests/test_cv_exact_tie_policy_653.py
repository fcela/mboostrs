"""Prespecified finite-curve selection, without an observed-outcome tolerance."""
import shutil
import subprocess

import numpy as np
import pytest

from mboostrs import _rust
from mboostrs.cvrisk import _optimum_index
from mboostrs.families import Quantile, _family_params


@pytest.mark.parametrize('exponent', [-600, 0, 600])
@pytest.mark.parametrize('distinct', [False, True])
def test_exact_and_adjacent_minima_match_native_selection(exponent, distinct):
    scale = float(2.**exponent)
    later = np.nextafter(scale, -np.inf) if distinct else scale
    target = np.array([0., 3*scale, scale, later, 2*scale])
    expected = 3 if distinct else 2

    # The offset-only risk is strictly smaller but not an eligible stop.
    assert _optimum_index(target, range(5)) == expected
    def provider(_):
        return [0.], [1.], np.ascontiguousarray(-2*target[:, None]), []
    _, aggregate, grid, optimum = _rust.cvrisk_foldwise(
        'quantreg', _family_params(Quantile(.5)), 2, 4, provider)
    np.testing.assert_array_equal(aggregate, target)
    assert grid == [0, 1, 2, 3, 4] and optimum == expected


def test_finite_curve_r_first_minimum_semantics(tmp_path):
    binary = shutil.which('Rscript')
    if binary is None:
        pytest.skip('Rscript is unavailable')
    rows, expected = [], []
    for exponent in (-600, 0, 600):
        scale = float(2.**exponent)
        for distinct in (False, True):
            later = np.nextafter(scale, -np.inf) if distinct else scale
            values = [0., 3*scale, scale, later, 2*scale]
            rows.append(values)
            expected.append(3 if distinct else 2)
    # Transfer IEEE values directly: decimal parsing in R can collapse adjacent
    # values at extreme scales before the selection rule even sees them.
    fixture = tmp_path/'curves.bin'
    np.asarray(rows, dtype='<f8').tofile(fixture)
    code = ('if (!requireNamespace("mboost", quietly=TRUE)) { cat("UNAVAILABLE"); quit() }; '
            'f <- file(commandArgs(TRUE)[1L], "rb"); '
            'rows <- matrix(readBin(f, "double", n=30L, size=8L, endian="little"), ncol=5L, byrow=TRUE); '
            'close(f); for (i in seq_len(nrow(rows))) { '
            'x <- structure(matrix(rep(rows[i,-1L], each=2L), nrow=2L), '
            'class=c("cvrisk", "matrix"), mstop=1:4); cat(mboost::mstop(x), "\\n") }')
    result = subprocess.run([binary, '--vanilla', '-e', code, str(fixture)],
                            text=True, check=True, capture_output=True)
    if result.stdout == 'UNAVAILABLE':
        pytest.skip('R mboost is unavailable')
    assert [int(value) for value in result.stdout.split()] == expected
