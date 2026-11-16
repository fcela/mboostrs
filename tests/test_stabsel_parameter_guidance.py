"""The missing-pair error guides the caller to mboost-style values (#506).

The two-of-cutoff-q-PFER requirement stays at mboost parity -- that decision
is recorded on the issue -- but the trap the D5 runs hit was the caller
burning a failed call and a retry turn to discover what to pass. The error
now names this model's effect count and a copy-pasteable pair.
"""

import numpy as np
import pytest

import mboostrs as mb


def _fit(effects, seed=0, n=200):
    rng = np.random.default_rng(seed)
    columns = {f"x{j}": rng.normal(size=n) for j in range(effects)}
    columns["y"] = rng.normal(size=n)
    formula = "y ~ " + " + ".join(f"bbs(x{j})" for j in range(effects))
    return mb.gamboost(formula, columns, family=mb.Gaussian(),
                       control=mb.boost_control(60))


@pytest.mark.parametrize("effects, suggested_q", [(2, 1), (5, 2), (12, 3), (40, 6)])
def test_missing_pair_error_names_the_mboost_style_pair(effects, suggested_q):
    with pytest.raises(ValueError, match=rf"q={suggested_q} .*PFER=1") as caught:
        mb.stabsel(_fit(effects))
    message = str(caught.value)
    assert f"This model has {effects} candidate effects" in message
    assert f"stabsel(model, q={suggested_q}, PFER=1)" in message
    # The cutoff pair is named too, and the parity requirement is stated.
    assert "cutoff=0.75, PFER=1" in message
    assert "matches mboost" in message


def test_single_effect_error_still_names_the_cutoff_pair():
    # With one candidate effect there is no q below p to suggest; the cutoff
    # pair carries the guidance alone.
    with pytest.raises(ValueError, match="cutoff=0.75, PFER=1") as caught:
        mb.stabsel(_fit(1))
    assert "This model has 1 candidate effects" in str(caught.value)
    assert "q=" not in str(caught.value)
