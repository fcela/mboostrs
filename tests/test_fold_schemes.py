"""Folds for data whose rows are not exchangeable.

`cv_weights` offers bootstrap, k-fold and subsampling, and all three assume
any row may stand in for any other. Three common designs break that.

*Unbalanced classes.* A plain 10-fold split of 200 rows carrying 12 positives
leaves folds holding 0, 1, 2 or 3 of them — measured below, two of the ten
folds hold none. The out-of-fold risk of a fold with no positive case
estimates nothing, and the spread across folds measures the split rather than
the model.

*Clustered rows.* Repeated measurements on the same unit. A random split puts
the same cluster on both sides, so a held-out row is partly predicted by its
own cluster-mates and the cross-validated risk is optimistic.

*Ordered rows.* A random split trains on the future to predict the past.

Each constructor returns a `FoldScheme`: the `(n, B)` in-bag weight matrix
`cvrisk` already takes, plus what the matrix cannot say — which rows each fold
holds out, and which grouping it respects, so `check_leakage` can be asked
about a grouping the scheme was never built around.
"""

import numpy as np
import pytest

import mboostrs as mb

N = 200


@pytest.fixture(scope="module")
def unbalanced_y():
    rng = np.random.default_rng(0)
    y = np.zeros(N)
    y[rng.choice(N, 12, replace=False)] = 1.0
    return y


@pytest.fixture(scope="module")
def clusters():
    """40 units of 5 rows each."""
    return np.repeat(np.arange(40), 5)


# ─── the weight convention every scheme shares ─────────────────────


@pytest.mark.parametrize("build", [
    lambda: mb.stratified_folds(np.tile([0.0, 1.0], N // 2), B=5),
    lambda: mb.grouped_folds(np.repeat(np.arange(40), 5), B=5),
    lambda: mb.blocked_folds(N, B=5),
    lambda: mb.rolling_folds(N, B=5, initial=100, horizon=20),
    lambda: mb.repeated_kfold_folds(N, k=5, repeats=2),
])
def test_a_scheme_is_an_in_bag_weight_matrix(build):
    """Column `b` carries the model's weights on the rows it trains on and
    zero on the rows it holds out — the same convention as `cv_weights`."""
    scheme = build()
    matrix = np.asarray(scheme)
    assert matrix.shape == (N, scheme.n_folds)
    assert matrix.dtype == np.float64
    assert np.all(matrix >= 0)
    for b in range(scheme.n_folds):
        held = scheme.fold(b)
        assert held.size > 0
        assert np.all(matrix[held, b] == 0.0)
        # A row is in-bag or held out, never both. A rolling scheme also
        # leaves rows that are neither: those after the fold's test window
        # have not happened yet from that fold's point of view.
        assert not np.any(matrix[held, b] > 0.0)
        assert (matrix[:, b] > 0.0).sum() > 0


def test_a_scheme_carries_the_model_weights(clusters):
    w = np.full(N, 2.0)
    scheme = mb.grouped_folds(clusters, B=5, weights=w)
    matrix = np.asarray(scheme)
    in_bag = matrix > 0
    assert np.all(matrix[in_bag] == 2.0)


def test_a_zero_weight_row_is_held_out_nowhere():
    """It is out of the sample entirely, which is a different thing from
    being the fold's test set."""
    w = np.ones(N)
    w[:10] = 0.0
    scheme = mb.blocked_folds(N, B=5, weights=w)
    assert not scheme.held_out[:10].any()
    assert np.all(np.asarray(scheme)[:10] == 0.0)


def test_a_scheme_drops_into_cvrisk_unchanged(clusters):
    rng = np.random.default_rng(3)
    data = {"x": rng.normal(size=N)}
    data["y"] = 2.0 * data["x"] + rng.normal(size=N)
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(30, 0.1))
    result = model.cvrisk(folds=mb.grouped_folds(clusters, B=5))
    assert result.risk_matrix.shape[0] == 5
    assert result.fold_type == "grouped 5-fold", (
        "a scheme knows its own name; the result should say it rather than "
        "'5 supplied folds'"
    )


# ─── stratified ────────────────────────────────────────────────────


def test_stratified_folds_spread_a_rare_class_evenly(unbalanced_y):
    scheme = mb.stratified_folds(unbalanced_y, B=10)
    per_fold = [int(unbalanced_y[scheme.fold(b)].sum()) for b in range(10)]
    assert sum(per_fold) == 12
    assert max(per_fold) - min(per_fold) <= 1, per_fold
    assert min(per_fold) >= 1


def test_the_plain_split_is_what_this_fixes(unbalanced_y):
    """Not a hypothetical: the default k-fold folds leave two of ten holding
    no positive case at all."""
    plain = mb.cv_weights(N, type="kfold", B=10, seed=1)
    per_fold = [int(unbalanced_y[plain[:, b] == 0].sum()) for b in range(10)]
    assert min(per_fold) == 0
    assert max(per_fold) - min(per_fold) >= 2


def test_stratified_folds_are_near_equal_in_size(unbalanced_y):
    sizes = mb.stratified_folds(unbalanced_y, B=10).fold_sizes()
    assert sizes.max() - sizes.min() <= 1


def test_more_folds_than_the_rarest_class_is_refused(unbalanced_y):
    with pytest.raises(ValueError, match="members of the smallest class"):
        mb.stratified_folds(unbalanced_y, B=20)


def test_stratified_folds_are_reproducible(unbalanced_y):
    a = mb.stratified_folds(unbalanced_y, B=5, seed=7)
    b = mb.stratified_folds(unbalanced_y, B=5, seed=7)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    c = mb.stratified_folds(unbalanced_y, B=5, seed=8)
    assert not np.array_equal(np.asarray(a), np.asarray(c))


# ─── grouped ───────────────────────────────────────────────────────


def test_grouped_folds_keep_every_group_whole(clusters):
    scheme = mb.grouped_folds(clusters, B=5)
    for b in range(5):
        held = set(clusters[scheme.fold(b)].tolist())
        trained = set(clusters[np.asarray(scheme)[:, b] > 0].tolist())
        assert not (held & trained)
    assert scheme.check_leakage()["clean"]


def test_grouped_folds_balance_unequal_group_sizes():
    sizes = np.array([50, 40, 30, 20, 10, 5, 3, 2])
    groups = np.repeat(np.arange(sizes.size), sizes)
    scheme = mb.grouped_folds(groups, B=4, seed=1)
    fold_sizes = scheme.fold_sizes()
    assert fold_sizes.sum() == groups.size
    # Largest-first packing keeps the folds within the largest group of each
    # other; naive round-robin would put 50 and 40 in different folds and
    # leave one holding 10.
    assert fold_sizes.max() - fold_sizes.min() <= int(sizes.max())


def test_more_folds_than_groups_is_refused(clusters):
    with pytest.raises(ValueError, match="distinct groups"):
        mb.grouped_folds(np.repeat(np.arange(3), 10), B=5)


# ─── leakage, including for schemes that do not respect the grouping ─


def test_a_plain_kfold_split_of_clustered_rows_is_reported_as_leaky(clusters):
    """The point of `check_leakage(groups=)`: ask a scheme about a grouping it
    knows nothing about. A matrix cannot answer, and the risk it produces is
    optimistic without saying so."""
    plain = mb.cv_weights(N, type="kfold", B=10, seed=1)
    scheme = mb.FoldScheme(weights=plain, held_out=(plain == 0),
                           scheme="plain 10-fold")
    with pytest.raises(ValueError, match="folds leak"):
        scheme.check_leakage(groups=clusters)
    report = scheme.check_leakage(groups=clusters, raise_on_leak=False)
    assert not report["clean"]
    assert len(report["leaks"]) == 10


def test_the_leak_message_says_what_to_do_instead(clusters):
    plain = mb.cv_weights(N, type="kfold", B=5, seed=1)
    scheme = mb.FoldScheme(weights=plain, held_out=(plain == 0),
                           scheme="plain 5-fold")
    with pytest.raises(ValueError, match="grouped_folds"):
        scheme.check_leakage(groups=clusters)


def test_warn_if_leaky_works_on_a_bare_matrix(clusters):
    from mboostrs.folds import warn_if_leaky

    plain = mb.cv_weights(N, type="kfold", B=5, seed=1)
    with pytest.warns(RuntimeWarning, match="optimistic"):
        warn_if_leaky(plain, clusters)


def test_a_grouping_of_the_wrong_length_is_refused(clusters):
    scheme = mb.blocked_folds(N, B=5)
    with pytest.raises(ValueError, match="expected 200"):
        scheme.check_leakage(groups=clusters[:10])


# ─── ordered data ──────────────────────────────────────────────────


def test_blocked_folds_hold_out_contiguous_stretches():
    scheme = mb.blocked_folds(100, B=5)
    for b in range(5):
        held = scheme.fold(b)
        assert np.array_equal(held, np.arange(held[0], held[-1] + 1))
    assert scheme.fold(0)[0] == 0
    assert scheme.fold(4)[-1] == 99


def test_blocked_folds_still_train_on_the_future_and_say_so():
    """Contiguous blocks fix local dependence, not the direction of time.
    `check_leakage` reports it; for a spatial transect it is not a leak, so it
    is reported rather than refused at construction."""
    scheme = mb.blocked_folds(100, B=5)
    report = scheme.check_leakage(raise_on_leak=False)
    assert not report["clean"]
    assert {leak["kind"] for leak in report["leaks"]} == {"time"}
    assert "rolling_folds" in scheme.describe()


def test_rolling_folds_never_train_on_a_later_row():
    scheme = mb.rolling_folds(100, B=5, initial=50, horizon=10)
    assert scheme.check_leakage()["clean"]
    for b in range(5):
        held = scheme.fold(b)
        trained = np.flatnonzero(np.asarray(scheme)[:, b] > 0)
        assert trained.max() < held.min()


def test_an_expanding_window_grows_and_a_sliding_one_does_not():
    expanding = mb.rolling_folds(100, B=4, initial=50, horizon=10)
    sliding = mb.rolling_folds(100, B=4, initial=50, horizon=10,
                               expanding=False)
    grow = [int((np.asarray(expanding)[:, b] > 0).sum()) for b in range(4)]
    fixed = [int((np.asarray(sliding)[:, b] > 0).sum()) for b in range(4)]
    assert grow == sorted(grow) and grow[0] < grow[-1]
    assert len(set(fixed)) == 1
    assert sliding.check_leakage()["clean"]


def test_a_rolling_scheme_that_does_not_fit_says_what_to_change():
    with pytest.raises(ValueError, match="Lower B, lower horizon"):
        mb.rolling_folds(100, B=10, initial=80, horizon=10)


# ─── repeated k-fold ───────────────────────────────────────────────


def test_repeated_kfold_holds_each_row_out_once_per_repeat():
    scheme = mb.repeated_kfold_folds(50, k=5, repeats=3)
    assert scheme.n_folds == 15
    assert np.all(scheme.held_out.sum(axis=1) == 3)


def test_the_repeats_are_different_partitions():
    scheme = mb.repeated_kfold_folds(50, k=5, repeats=3, seed=2)
    matrix = np.asarray(scheme)
    assert not np.array_equal(matrix[:, :5], matrix[:, 5:10])


# ─── the scheme reports itself ─────────────────────────────────────


def test_describe_names_the_scheme_and_its_shape(clusters):
    text = mb.grouped_folds(clusters, B=5).describe()
    assert "grouped 5-fold" in text
    assert "5 folds over 200 rows" in text
    assert "groups kept whole: 40" in text


# ─── counts are checked, not truncated ─────────────────────────────


@pytest.mark.parametrize("build", [
    lambda: mb.stratified_folds(np.tile([0.0, 1.0], 20), B=5.7),
    lambda: mb.grouped_folds(np.repeat(np.arange(8), 5), B=5.0 + 1e-9),
    lambda: mb.rolling_folds(100, B=5, initial=50.5),
    lambda: mb.rolling_folds(100, B=5, initial=50, horizon=2.5),
    lambda: mb.repeated_kfold_folds(50, k=5, repeats=1.5),
])
def test_fractional_counts_are_refused_not_truncated(build):
    # A fractional B used to be silently truncated by ``int(B)``; the same
    # held for initial, horizon and repeats. The constructors now refuse them
    # (their signatures type these as int, so the floats below are pytest's
    # way of smuggling a bad call in).
    with pytest.raises((ValueError, TypeError), match="whole number"):
        build()


def test_weights_follow_the_public_finite_non_negative_contract():
    ok = np.ones(N)
    for bad, match in [
        (np.r_[ok[:-1], np.nan], "must be finite"),
        (np.r_[ok[:-1], np.inf], "must be finite"),
        (np.r_[ok[:-1], -1.0], "non-negative"),
        (np.zeros(N), "all zero"),
    ]:
        with pytest.raises(ValueError, match=match):
            mb.blocked_folds(N, B=5, weights=bad)


def test_stratified_balance_counts_in_sample_rows_only():
    """A zero-weight row is out of the sample entirely, so it must not tilt
    how the classes are dealt to folds."""
    y = np.repeat([0, 1], 60)                 # 60 each
    w = np.ones(120)
    w[np.where(y == 1)[0][30:]] = 0.0         # class 1 has 30 in-sample rows
    scheme = mb.stratified_folds(y, B=6, weights=w)
    matrix = np.asarray(scheme)
    in_sample = w > 0
    for level in (0, 1):
        rows = np.flatnonzero((y == level) & in_sample)
        held = [(matrix[rows, b] == 0).sum() for b in range(6)]
        assert np.max(held) - np.min(held) <= 1
    # the 30 zero-weight rows are held out in no fold at all
    assert not scheme.held_out[~in_sample].any()


def test_group_capacity_counts_groups_with_an_effective_sample():
    """A group whose members all carry zero weight trains on nothing and is
    judged on nothing, so it neither bounds B nor tilts the load."""
    groups = np.repeat(np.arange(20), 2)     # 20 groups of 2
    w = np.ones(40)
    w[groups >= 5] = 0.0                      # only 5 groups remain
    with pytest.raises(ValueError, match="exceeds"):
        mb.grouped_folds(groups, B=6, weights=w)
    scheme = mb.grouped_folds(groups, B=5, weights=w)
    in_sample = w > 0
    held = [(np.asarray(scheme)[in_sample, b] == 0).sum()
            for b in range(scheme.n_folds)]
    assert np.max(held) - np.min(held) <= 1
    assert not scheme.held_out[~in_sample].any()


@pytest.mark.parametrize("output", ["records", "polars", "pandas"])
def test_to_frame_honours_output(output):
    frame = mb.blocked_folds(20, B=4).to_frame(output=output)
    expected = {"records": "builtins", "polars": "polars", "pandas": "pandas"}
    assert type(frame).__module__.split(".")[0] == expected[output]


# --- the merged-state audit's rolling-origin finding -------------------

def test_a_rolling_fold_scores_only_its_horizon():
    """The audit's first finding: "`rolling_folds` marks only the requested
    horizon in `held_out`, but its weight matrix is zero for every row outside
    the training window. The CV engines define every zero-weight row as out of
    bag. On the documented `n=100, initial=50, horizon=10` first fold, 10 rows
    are documented as held out while 50 rows are actually scored."

    A rolling-origin fold has three kinds of row, not two: trained on, judged,
    and neither -- the future beyond the horizon has not happened yet from
    that origin's point of view, and scoring it is not a forecast at any
    horizon. `judge` names the second kind, and the engines read it.
    """
    scheme = mb.rolling_folds(n=100, initial=50, horizon=10)
    weights = np.asarray(scheme)
    held_out = np.asarray(scheme.held_out)
    judged = np.asarray(scheme.judge)

    assert judged is not None, "a rolling scheme must name its judge rows"
    for fold in range(weights.shape[1]):
        trained = weights[:, fold] > 0
        scored = judged[:, fold]
        assert not (trained & scored).any(), "a row cannot be both"
        assert scored.sum() <= 10, "no more rows than the horizon"
        assert np.array_equal(scored, held_out[:, fold]), (
            "held_out and judge describe the same rows for this scheme")
        assert (weights[:, fold] == 0).sum() >= scored.sum()

    # The finding, stated as a number: on the first fold 50 rows carry zero
    # weight and 10 are judged, so "score every zero-weight row" scored five
    # times the horizon. The two counts coincide only on the last fold, where
    # nothing is left beyond it.
    assert (weights[:, 0] == 0).sum() == 50
    assert judged[:, 0].sum() == 10


def test_the_cv_engines_read_the_judge_mask(gaussian_frame=None):
    """A `judge` that nothing consults is documentation, not a fix."""
    import pathlib

    package = pathlib.Path(mb.__file__).parent
    for name in ("gamboost.py", "glmboost.py"):
        source = (package / name).read_text(encoding="utf-8")
        assert "folds.judge" in source, name


def test_a_k_fold_scheme_leaves_judge_unset():
    """"unset means score the zero-weight rows", which is what every scheme
    but the rolling one wants. Setting it everywhere would be a second way to
    say the same thing, and the two would drift."""
    for scheme in (mb.blocked_folds(60, B=5),
                   mb.stratified_folds(np.repeat([0, 1], 30).astype(float), B=5),
                   mb.grouped_folds(np.repeat(np.arange(12), 5).astype(float),
                                    B=4)):
        assert scheme.judge is None, scheme.scheme


def test_take_restricts_every_per_row_field_together():
    """A model fitted with na_action='omit' estimates on complete cases, so a
    scheme over the full frame must drop the same rows. Fold membership, the
    judge set and the group labels all have to move with the weights or the
    subset scheme silently changes what is scored."""
    scheme = mb.rolling_folds(n=100, initial=50, horizon=10)
    rows = np.arange(0, 100, 2)
    taken = scheme.take(rows)

    assert taken.shape == (len(rows), scheme.n_folds)
    np.testing.assert_array_equal(taken.weights, scheme.weights[rows])
    np.testing.assert_array_equal(taken.held_out, scheme.held_out[rows])
    np.testing.assert_array_equal(taken.judge, scheme.judge[rows])
    assert taken.scheme == scheme.scheme and taken.ordered == scheme.ordered
    for fold in range(scheme.n_folds):
        original = set(scheme.fold(fold).tolist())
        expected = np.array(sorted(original & set(rows.tolist())), dtype=int)
        expected = np.searchsorted(rows, expected)
        np.testing.assert_array_equal(taken.fold(fold), expected)
    # A boolean mask selects the same rows.
    mask = np.zeros(100, dtype=bool)
    mask[rows] = True
    np.testing.assert_array_equal(taken.weights, scheme.take(mask).weights)
