"""screen_discontinuities: locate jumps a smooth fit absorbs.

The benchmark case behind the diagnostic: a +2.0 jump across a diagonal
boundary ``s1 = 0.45 + 0.35 s0`` at n=800. No single-column step can express
it, the smooth fit absorbs most of it, and nothing else in the workflow
points at a diagonal boundary. The tests pin the behaviours the design asks
for: the diagonal jump is found (raw pass and residual pass), data with no
step-beyond-smooth structure does not flag, and a steep smooth crossing may
flag but is not typed as a jump -- that typing is what separates the two.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def spatial(seed, n=800):
    """The benchmark's diagonal-jump panel plus smooth and noise columns."""
    rng = np.random.default_rng(seed)
    cols = {name: rng.uniform(0, 1, n) for name in ("s0", "s1", "x_smooth",
                                                    "n1", "n2", "n3")}
    g = (cols["s1"] > 0.45 + 0.35 * cols["s0"]).astype(float)
    mu = 2.0 * g + 1.5 * cols["s0"] + 1.0 * cols["s1"] \
        + 1.5 * np.sin(np.pi * cols["x_smooth"])
    y = mu + rng.normal(0, 0.5, n)
    return cols, y, g


def spatial_null(seed, n=800):
    """The same columns and smooth mean, with no jump."""
    rng = np.random.default_rng(seed)
    cols = {name: rng.uniform(0, 1, n) for name in ("s0", "s1", "x_smooth",
                                                    "n1", "n2", "n3")}
    mu = 1.5 * cols["s0"] + 1.0 * cols["s1"] \
        + 1.5 * np.sin(np.pi * cols["x_smooth"])
    y = mu + rng.normal(0, 0.5, n)
    return cols, y


def _frame(cols, y):
    return pl.DataFrame({**cols, "y": y})


def _residuals(data, y, rhs, mstop=100):
    model = mb.gamboost(rhs, data, control=mb.boost_control(mstop=mstop, nu=0.1))
    return y - np.asarray(model.predict(type="response")).ravel()


def test_diagonal_jump_is_found_on_the_raw_response():
    cols, y, _ = spatial(0)
    hits = mb.screen_discontinuities(_frame(cols, y), y, response="y", seed=0)
    top = hits[0]
    assert top.typed_jump, repr(top)
    assert top.columns == ("s0", "s1")
    assert top.kind == "diagonal"
    assert top.z > 5.0
    # The boundary reads back on the original scale: truth is
    # s1 > 0.45 + 0.35 s0.
    agreement = np.mean(
        (cols["s1"] > top.boundary_intercept
         + top.boundary_slope * cols["s0"]) == (cols["s1"] > 0.45 + 0.35 * cols["s0"]))
    assert agreement > 0.9, agreement


def test_diagonal_jump_is_found_on_residuals_after_the_smooth_fit():
    cols, y, _ = spatial(0)
    data = _frame(cols, y)
    residual = _residuals(data, y, "y ~ bbs(s0) + bbs(s1) + bbs(x_smooth)")
    hits = mb.screen_discontinuities(data, residual, response="y", seed=0)
    top = hits[0]
    assert top.flagged, repr(top)
    assert top.columns == ("s0", "s1")
    # The smooth baseline absorbed part of the jump, so the residual-pass
    # jump estimate is smaller than the true 2.0 but clearly present.
    assert top.jump > 0.5
    assert top.typed_jump, repr(top)


def test_null_panel_stays_quiet_on_the_residual_pass():
    # After the smooth model absorbs the mean, no candidate has a step to
    # find: the recommended second pass, which is what the ~4.6% false-flag
    # rate of the |z| >= 2 rule refers to.
    cols, y = spatial_null(1000)
    data = _frame(cols, y)
    residual = _residuals(data, y, "y ~ bbs(s0) + bbs(s1) + bbs(x_smooth)")
    hits = mb.screen_discontinuities(data, residual, response="y", seed=0)
    assert not hits[0].flagged, repr(hits[0])


def test_pure_noise_does_not_flag():
    for seed in (7, 8, 9):
        rng = np.random.default_rng(seed)
        data = pl.DataFrame({"a": rng.uniform(size=400),
                             "b": rng.uniform(size=400),
                             "c": rng.uniform(size=400)})
        hits = mb.screen_discontinuities(data, rng.normal(size=400), seed=0)
        assert not hits[0].flagged, (seed, repr(hits[0]))


def test_steep_smooth_rise_flags_but_is_not_typed_as_a_jump():
    # The sin term's steep rise is real step-beyond-[1, u, u^2] structure on
    # the raw response, so the gate fires honestly. The typing statistic
    # sees the smooth background absorb it and declines to call it a jump:
    # the two statistics have different jobs.
    cols, y = spatial_null(1000)
    hits = mb.screen_discontinuities(_frame(cols, y), y, response="y", seed=0)
    top = hits[0]
    assert top.flagged, repr(top)
    assert not top.typed_jump, repr(top)
    assert abs(top.type_z) < 2.0, repr(top)


def test_smooth_saddle_is_not_typed_as_a_jump():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.uniform(0, 1, size=(n, 6))
    mu = (3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
          + 1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3])
    y = mu + rng.normal(0, 0.5, n)
    data = pl.DataFrame({**{f"x{i}": X[:, i] for i in range(6)}, "y": y})
    residual = _residuals(data, y, "y ~ bbs(x0) + bbs(x1) + bbs(x2) + bbs(x3)")
    hits = mb.screen_discontinuities(data, residual, response="y", seed=0)
    assert not hits[0].typed_jump, repr(hits[0])


def test_kinked_response_is_not_flagged_on_residuals():
    rng = np.random.default_rng(2)
    n = 800
    X = rng.uniform(0, 1, size=(n, 7))
    mu = (3.0 * np.clip(X[:, 0] - 0.35, 0, None)
          - 2.0 * np.clip(0.68 - X[:, 1], 0, None)
          + 2.0 * np.sin(np.pi * X[:, 2]))
    y = mu + rng.normal(0, 0.5, n)
    data = pl.DataFrame({**{f"x{i}": X[:, i] for i in range(7)}, "y": y})
    residual = _residuals(data, y, "y ~ bbs(x0) + bbs(x1) + bbs(x2)")
    hits = mb.screen_discontinuities(data, residual, response="y", seed=0)
    assert not hits[0].flagged, repr(hits[0])


def test_axis_jump_is_found_and_reported_on_the_x_scale():
    rng = np.random.default_rng(4)
    n = 600
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.where(x > 0.6, 2.0, 0.0) + rng.normal(0, 0.4, n)
    data = pl.DataFrame({"x": x, "z": z, "y": y})
    hits = mb.screen_discontinuities(data, y, response="y", seed=0,
                                     columns=["x"])
    top = hits[0]
    assert top.kind == "axis"
    assert top.columns == ("x",)
    assert top.typed_jump, repr(top)
    assert abs(top.threshold_x - 0.6) < 0.05, top.threshold_x
    assert top.line.startswith("x >")


def test_same_seed_repeats_exactly():
    cols, y, _ = spatial(0)
    data = _frame(cols, y)
    first = mb.screen_discontinuities(data, y, response="y", seed=0)
    second = mb.screen_discontinuities(data, y, response="y", seed=0)
    # repr, not == : rows outside the typing budget carry NaN type_z, and
    # NaN != NaN under dataclass equality.
    assert [repr(h) for h in first] == [repr(h) for h in second]


def test_top_k_and_pairs_arguments():
    cols, y, _ = spatial(0)
    data = _frame(cols, y)
    two = mb.screen_discontinuities(data, y, response="y", seed=0, top_k=2)
    assert len(two) == 2
    all_rows = mb.screen_discontinuities(data, y, response="y", seed=0, top_k=0)
    assert len(all_rows) > len(two)
    # pairs= restricts the diagonal family; the axis family is unchanged.
    only = mb.screen_discontinuities(data, y, response="y", seed=0,
                                      pairs=[("s0", "s1")], top_k=0)
    assert {row.columns for row in only if row.kind == "diagonal"} == {("s0", "s1")}
    assert any(row.kind == "axis" for row in only)


def test_error_paths():
    rng = np.random.default_rng(5)
    n = 100
    data = pl.DataFrame({"x": rng.uniform(size=n), "y": rng.normal(size=n)})
    # Too few rows for the split the statistics assume.
    small = pl.DataFrame({"x": rng.uniform(size=50),
                          "y": rng.normal(size=50)})
    with pytest.raises(ValueError, match="at least 80 rows"):
        mb.screen_discontinuities(small, small["y"].to_numpy())
    # Missing target values are an error by default and name the first row.
    target = np.array(data["y"].to_numpy(), dtype=float)
    target[7] = np.nan
    with pytest.raises(ValueError, match="target has 1 non-finite"):
        mb.screen_discontinuities(data, target)
    # na_action="omit" drops the rows; "impute" cannot fill a target.
    with pytest.raises(ValueError, match="cannot fill a target"):
        mb.screen_discontinuities(data, target, na_action="impute")
    hits = mb.screen_discontinuities(data, target, na_action="omit")
    assert hits
    # Unknown column and constant column.
    with pytest.raises(ValueError, match="not in the data"):
        mb.screen_discontinuities(data, data["y"].to_numpy(), columns=["nope"])
    constant = pl.DataFrame({"x": np.ones(100), "y": rng.normal(size=100)})
    with pytest.raises(ValueError, match="constant column"):
        mb.screen_discontinuities(constant, constant["y"].to_numpy())


def test_duplicate_columns_do_not_crash_the_screen():
    # FAIL-1 regression: a duplicated column (or a near-duplicate differing
    # by 1e-9 noise) makes the diagonal unit's background singular. The unit
    # must be skipped gracefully -- the screen completes and still finds the
    # jump -- not kill the whole call.
    rng = np.random.default_rng(11)
    n = 800
    s0 = rng.uniform(0, 1, n)
    s1 = rng.uniform(0, 1, n)
    y = 2.0 * (s1 > 0.45 + 0.35 * s0) + rng.normal(0, 0.5, n)
    for dup in (s0.copy(), s0 + rng.normal(0, 1e-9, n)):
        data = pl.DataFrame({"s0": s0, "s0_dup": dup, "s1": s1, "y": y})
        hits = mb.screen_discontinuities(data, y, response="y", seed=0)
        assert hits, "the screen returned nothing"
        top = hits[0]
        assert top.typed_jump, repr(top)
        # The duplicate is s0 under another name; the boundary may read
        # back on either copy.
        assert set(top.columns) == {"s0", "s1"} or \
            set(top.columns) == {"s0_dup", "s1"}, repr(top)


def test_null_flag_rate_stays_near_nominal():
    # CONCERN-2: the ~4.6% flag-rate claim needs a simulation, not just a
    # single quiet panel. Under the null, flags ~ Binomial(n, 0.046); with
    # 40 screens P(>6 flags) is about 2%, so this bound catches
    # miscalibration without flaking. Two null regimes: pure noise on the
    # raw response, and smooth structure screened on the residual pass
    # (the raw pass honestly flags real smooth structure -- that is what
    # the typing step is for). Half the screens carry 10% MCAR with
    # na_action="impute", which also exercises the fit-half-only median
    # (CONCERN-6): leaked imputation would break the calibration.
    flagged = 0
    n_screens = 40
    for seed in range(n_screens):
        rng = np.random.default_rng(100 + seed)
        n = 400
        cols = {name: rng.uniform(0, 1, n) for name in ("a", "b", "c")}
        if seed % 2 == 0:
            # Pure noise: nothing to find on the raw response.
            screen_target = rng.normal(0, 0.5, n)
        else:
            # Smooth structure, screened on the residual pass after the
            # smooth fit absorbs the mean.
            mu = 1.5 * cols["a"] + 1.0 * np.sin(np.pi * cols["b"])
            y = mu + rng.normal(0, 0.5, n)
            screen_target = _residuals(_frame(cols, y), y,
                                       "y ~ bbs(a) + bbs(b)")
        if seed % 4 in (0, 1):
            cols["a"] = np.where(rng.uniform(0, 1, n) < 0.10, np.nan,
                                 cols["a"])
            na_action = "impute"
        else:
            na_action = "error"
        hits = mb.screen_discontinuities(
            _frame(cols, screen_target if seed % 2 == 0 else y),
            screen_target, response="y", seed=seed, na_action=na_action)
        flagged += bool(hits[0].flagged)
    assert flagged <= 6, f"{flagged}/{n_screens} null screens flagged"


def test_refinement_sharpens_the_boundary_without_breaking_honesty():
    # CONCERN-2: refinement re-scans the best diagonal units on a finer
    # grid using fit-half quantities only. It must not worsen the boundary
    # and must not inflate the eval-half z (selection honesty). The refined
    # threshold is accurate enough for the local gap discriminator to type
    # the jump; the coarse-grid threshold may be too far off for a confident
    # typing, which is honest.
    cols, y, _ = spatial(0)
    data = _frame(cols, y)
    truth = (cols["s1"] > 0.45 + 0.35 * cols["s0"])

    def agreement(top):
        return np.mean(
            (cols["s1"] > top.boundary_intercept
             + top.boundary_slope * cols["s0"]) == truth)

    plain = mb.screen_discontinuities(data, y, response="y", seed=0,
                                      refine_units=0)[0]
    refined = mb.screen_discontinuities(data, y, response="y", seed=0)[0]
    assert refined.typed_jump, repr(refined)
    assert agreement(refined) >= agreement(plain) - 0.02, (
        agreement(refined), agreement(plain))
    # The eval-half z stays a calibrated quantity, not an inflated one:
    # a true jump of this size reads z ~ 16, not hundreds.
    assert refined.z < 60, refined.z


def test_steep_sigmoid_is_not_typed_as_a_jump():
    # CONCERN-1: a steep logistic sigmoid is smooth but the global type_z
    # cannot see that (the cubic background cannot fit it). The local
    # scale-invariance discriminator must decline to type it as a jump.
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        n = 800
        u = rng.uniform(-2, 2, n)
        v = rng.uniform(0, 1, n)
        y = 1 / (1 + np.exp(-8 * u)) + rng.normal(0, 0.3, n)
        data = pl.DataFrame({"u": u, "v": v, "y": y})
        hits = mb.screen_discontinuities(data, y, response="y", seed=seed,
                                         columns=["u"])
        top = hits[0]
        # It may flag (there is real step-like structure), but it must not
        # type as a jump.
        assert not top.typed_jump, (seed, repr(top))


def test_kink_is_not_typed_as_a_jump():
    # CONCERN-1: a ReLU kink is continuous, so there is no level gap. The
    # discriminator must not type it as a jump.
    rng = np.random.default_rng(2)
    n = 800
    u = rng.uniform(-2, 2, n)
    v = rng.uniform(0, 1, n)
    y = np.maximum(0, u) + rng.normal(0, 0.3, n)
    data = pl.DataFrame({"u": u, "v": v, "y": y})
    hits = mb.screen_discontinuities(data, y, response="y", seed=2,
                                     columns=["u"])
    top = hits[0]
    assert not top.typed_jump, repr(top)
