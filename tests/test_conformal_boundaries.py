"""What conformal calibration accepts, and what it refuses by name.

P0.5's closure conditions, three of which were open:

  "reject LSS models at the conformal entry point with a deliberate, typed
  unsupported-operation error; do not let a dictionary-valued prediction fall
  through to a generic `float()` `TypeError`"

  "classify external matrices by row alignment rather than learner name.
  `Bconstrained.C` is coefficient-indexed and should not by itself block a row
  subset, while `Buser.X` is row-aligned and should either be subset with the
  response or refused with a precise documented reason"

  "preserve the model-owned `RefitSpec` through serialization if it is called
  serializable"

Handed a distributional fit, the scoring reached `float()` on a dict and
raised `TypeError: float() argument must be a string or a real number, not
'dict'` — a message about the crash rather than the reason. And every
`bconstrained` fit was refused, because the blocker classified an external
matrix by which learner held it: `C` has one column per spline coefficient and
no row dimension at all, so dropping rows cannot misalign it.
"""

import pickle
import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 100


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    frame = {"x": rng.normal(size=N)}
    frame["y"] = np.sin(2 * frame["x"]) + rng.normal(scale=0.3, size=N)
    frame["g"] = rng.integers(0, 3, N).astype(float)
    return frame


def fit_gam(formula, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(formula, data, control=mb.boost_control(20, 0.1),
                           **kwargs)


# --- the distributional refusal ---------------------------------------

def test_a_distributional_fit_is_refused_by_name(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lss = mb.gamboostLSS("y ~ bbs(x, df=4)", data,
                             family=mb.GaussianLSS(),
                             control=mb.boost_control(20, 0.1))
    # The default calibrates the fitted distribution (issue #111); the
    # residual-based methods refuse and name it.
    assert mb.conformal_interval(lss, newdata=data, level=0.9).method == "distributional"
    with pytest.raises(NotImplementedError) as caught:
        mb.conformal_interval(lss, newdata=data, level=0.9, method="cqr")
    message = str(caught.value)
    assert "no single residual to rank" in message
    assert "method='distributional'" in message, "it should name the alternative"


def test_the_named_alternative_exists(data):
    """A refusal that points at a method which is not there is worse than no
    refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lss = mb.gamboostLSS("y ~ bbs(x, df=4)", data,
                             family=mb.GaussianLSS(),
                             control=mb.boost_control(20, 0.1))
    assert hasattr(lss, "prediction_interval")
    assert hasattr(lss, "centiles")


# --- row alignment, not learner name ----------------------------------

def test_a_coefficient_indexed_constraint_does_not_block_a_row_subset(data):
    """`Bconstrained.C` is `(constraints, p)`. No dimension is the row count,
    so a row subset cannot misalign it."""
    constraint = np.ones((1, 24))
    fit = fit_gam("y ~ bconstrained(x, C=Cc)", data, env={"Cc": constraint})
    interval = fit.conformal_interval(newdata=data, level=0.9, method="split")
    coverage = float(interval.empirical_coverage(data["y"]))
    assert 0.7 < coverage < 1.0


def test_a_row_aligned_design_still_blocks_it(data):
    """`Buser.X` has one row per observation, so it does."""
    design = np.column_stack([data["x"], data["x"] ** 2])
    fit = fit_gam("y ~ buser(x, X=Xd, K=Kd, df=2)", data,
                  env={"Xd": design, "Kd": np.eye(2)})
    with pytest.raises(NotImplementedError, match="one row per observation"):
        fit.conformal_interval(newdata=data, level=0.9, method="split")


def test_an_external_matrix_sized_like_the_rows_blocks_it(data):
    """The rule is the shape, not the attribute's name: a `K` that happens to
    have the training row count is as misalignable as an `X` does."""
    from mboostrs._refit import RefitSpec

    class Fake:
        name = "x"

    learner = Fake()
    learner.K = np.eye(N)
    spec = RefitSpec(kind="gamboost", formula="y ~ buser(x)",
                     data={"y": np.zeros(N)},
                     arguments={"learners": [learner]}, interactions=None)
    blockers = spec.row_subset_blockers()
    assert any(f"length {N}" in blocker for blocker in blockers), blockers


def test_a_neighbourhood_penalty_still_blocks_it(data):
    """`Bmrf` is indexed by the regions present, which a row subset can drop —
    a different reason from row alignment, and still a reason."""
    graph = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    fit = fit_gam("y ~ bols(x) + bmrf(g, K=K_mrf, df=2)", data,
                  env={"K_mrf": graph})
    with pytest.raises(NotImplementedError, match="neighbourhood penalty"):
        fit.conformal_interval(newdata=data, level=0.9, method="split")


def test_a_plain_fit_is_not_blocked_by_any_of_this(data):
    fit = fit_gam("y ~ bbs(x, df=4)", data)
    assert fit.conformal_interval(newdata=data, level=0.9) is not None


# --- serialization ----------------------------------------------------

@pytest.mark.parametrize("route", ["pickle", "artifact"])
def test_a_restored_model_refuses_conformal_with_a_reason(route, data,
                                                          tmp_path):
    """A model without the fit's call (a pickle written by an earlier
    release, a scoring-only artifact) takes the other branch of the
    condition: the refusal has to say why, not fail somewhere inside the
    refit. A current pickle or full artifact keeps the call (issue #820)."""
    fit = fit_gam("y ~ bbs(x, df=4)", data)
    if route == "pickle":
        restored = pickle.loads(pickle.dumps(fit))
        restored._refit_bundle = None
    else:
        path = tmp_path / "model.json"
        mb.save_model(fit, str(path), scoring_only=True)
        restored = mb.load_model(str(path))

    with pytest.raises(ValueError, match="did not retain the inputs"):
        restored.conformal_interval(newdata=data, level=0.9)


def test_a_restored_model_lists_conformal_as_unavailable(data, tmp_path):
    """And says so before the caller tries, through the same accessor the
    documentation points at."""
    fit = fit_gam("y ~ bbs(x, df=4)", data)
    path = tmp_path / "model.json"
    mb.save_model(fit, str(path), scoring_only=True)
    operations = {entry["operation"]
                  for entry in mb.load_model(str(path)).unavailable_operations()}
    assert "conformal_interval" in operations


def test_constraint_with_as_many_coefficients_as_rows_does_not_block():
    """24 rows and the default 24-column spline basis: `C` is (1, 24), so
    the old shape rule read its coefficient axis as a row axis and refused.
    The learner's declared axis roles decide instead (issue #65)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 24)
    rows = {"x": x, "y": np.sin(2 * x) + rng.normal(scale=0.3, size=24)}
    fit = fit_gam("y ~ bconstrained(x, C=Cc)", rows, env={"Cc": np.ones((1, 24))})
    interval = fit.conformal_interval(newdata=rows, level=0.9, method="split")
    assert np.all(np.isfinite(interval.lower))
