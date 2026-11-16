"""scikit-learn compatible estimators wrapping mboostrs.

Optional integration — requires scikit-learn (``pip install mboostrs[sklearn]``).
These adapt the formula/data ``glmboost`` / ``gamboost`` API to the sklearn
estimator protocol (``fit(X, y)`` / ``predict(X)``, ``get_params`` /
``set_params``), so the models work inside ``Pipeline``, ``cross_val_score``,
and ``GridSearchCV`` (e.g. tuning ``mstop`` / ``nu`` / ``df``).

Two ways to specify the model:

1. **Defaults (no formula).** Each feature column becomes one base learner of
   the estimator's type — a linear ``bols`` term for the ``GLM*`` estimators,
   a P-spline ``bbs`` term for the ``GAM*`` estimators::

       GAMBoostRegressor(df=4).fit(X, y)   # y ~ bbs(f0,df=4) + bbs(f1,df=4) + ...

2. **Explicit formula (constructor parameter).** The ``GAM*`` estimators take a
   full mboostrs ``formula=`` that is used verbatim, giving per-term control
   (mixed ``bols``/``bbs``/``btree``/...), per-feature ``df``, interactions,
   ``by=``, etc.::

       GAMBoostRegressor(formula="y ~ bbs(age, df=5) + bols(income) + btree(z)")

   The formula references column names. Pass a pandas/polars ``DataFrame`` and
   the real column names are used; pass a bare array and columns are named
   ``f0, f1, ...``. The response on the left of ``~`` is supplied by ``y`` in
   ``fit`` (its name is only a placeholder). Without ``formula``, a column whose
   name the formula parser cannot read (a space, a parenthesis, or the name
   ``y``) is renamed ``f<j>`` in the fitted model; with ``formula``, such a
   column is refused and the message names it.

   Columns must be numeric. A string or category column is refused with its
   name and a suggestion to encode it as dummy columns.

``formula`` lives in the constructor (not ``fit``) because that is what makes
it part of ``get_params``/``set_params`` — so ``clone``, ``Pipeline`` and
``GridSearchCV`` carry and can tune it. A ``fit(X, y, formula=...)`` argument
would be invisible to all of those.

**On sklearn's output-container API** (``set_output(transform=...)``, which
gained ``"polars"`` in sklearn 1.4): sklearn wires that API through
``sklearn.utils._set_output._SetOutputMixin``, and only ``TransformerMixin``
inherits it — the mixin wraps ``transform``/``fit_transform`` and gates
``set_output`` itself behind ``hasattr(estimator, "get_feature_names_out")
and "transform" in auto_wrap_output_keys`` (see
``_SetOutputMixin._auto_wrap_is_configured``). Every estimator here is a
predictor (regressor, classifier, or the distributional/survival variants) —
none defines ``transform`` or ``get_feature_names_out``, and none should:
there is no "transformed ``X``" these models produce, only a prediction. So
``set_output`` does not exist on any of them (verified:
``hasattr(GLMBoostRegressor(), "set_output")`` is ``False``, same for the
other five), and no code in this module tries to fake it — a real
``transform`` method would be the only thing that made the polars/pandas
container choice meaningful, and none of these estimators has one. If a
future estimator here legitimately grows a ``transform`` (e.g. exposing
fitted basis columns), it should inherit ``TransformerMixin`` and implement
``get_feature_names_out``; the output-container wiring then comes for free
from sklearn, nothing bespoke needed.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
)

from mboostrs import families as _families
from mboostrs._control import boost_control
from mboostrs._names import partition_tilde, quote_name
from mboostrs.gamboost import gamboost
from mboostrs.glmboost import glmboost

__all__ = [
    "GLMBoostRegressor",
    "GAMBoostRegressor",
    "GLMBoostClassifier",
    "GAMBoostClassifier",
    "TreeBoostRegressor",
    "TreeBoostClassifier",
    "LSSRegressor",
    "SurvivalBoost",
]

_REG_FAMILIES = {
    "gaussian": _families.Gaussian,
    "poisson": _families.Poisson,
    "gamma": _families.GammaReg,
    "laplace": _families.Laplace,
    "huber": _families.Huber,
}

# Families parameterized by an asymmetry level rather than fitting the mean.
_TAU_FAMILIES = {
    "quantile": _families.Quantile,
    "expectile": _families.Expectile,
}

_FEATURE_PREFIX = "f"
_RESPONSE_NAME = "y"


def _refuse_non_numeric(X):
    """Raise a ValueError that names the column and the remedy when a frame
    column cannot be read as numbers (a string or category column).

    Without this check the error is numpy's ``could not convert string to
    float: 'a'``, which does not say which column or what to do. The adapters
    take numeric columns only; a categorical covariate must be encoded first.
    """
    if getattr(X, "columns", None) is None:
        return
    for col in X.columns:
        try:
            np.asarray(X[col], dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"column {col!r} is not numeric. The scikit-learn adapters take numeric "
                "columns only: encode it before fitting as dummy columns, for example "
                "`pd.get_dummies(X, columns=[col], drop_first=True, dtype=float)`."
            ) from None


_FORMULA_NAME_RE = re.compile(r"[\w.]+")


def _formula_names(names):
    """The name each column carries in the generated formula.

    A name the formula parser reads as a plain identifier (letters, digits,
    ``_`` and ``.``) is kept, so ``coef()`` and ``selected`` show the user's
    names. Any other name is replaced by ``f<j>``, and so is the response name
    ``y``: a feature called ``y`` would otherwise be overwritten by the response
    in the data dictionary, and the fit would silently use the response as a
    covariate.
    """
    out = []
    for j, nm in enumerate(names):
        keep = (
            _FORMULA_NAME_RE.fullmatch(nm) is not None
            and nm != _RESPONSE_NAME
            and nm not in out
        )
        name = nm if keep else f"{_FEATURE_PREFIX}{j}"
        while name in out:
            name += "_"
        out.append(name)
    return out


def _check_formula_names(names, response):
    """Refuse a column that a user ``formula=`` cannot reference.

    With ``formula=`` the names are written by the user, so they are used
    as given and a name the parser cannot read is an error naming the column.
    """
    for nm in names:
        if _FORMULA_NAME_RE.fullmatch(nm) is None:
            suggestion = re.sub(r"[^\w.]", "_", nm)
            raise ValueError(
                f"column {nm!r} cannot be written in a formula (only letters, digits, "
                f"'_' and '.' are read). Rename the column, for example {suggestion!r}, "
                "or fit without formula= on a NumPy array."
            )
    if response in names:
        raise ValueError(
            f"column {response!r} has the same name as the response on the left of '~'. "
            "Rename the column, or name the response differently in formula=."
        )
    return list(names)


def _settle_names(est, names_in):
    """The column names the generated formula uses, for an estimator's `formula`.

    With no formula the names are made safe (`_formula_names`); with a user
    formula they are checked and used as given (`_check_formula_names`).
    """
    formula = getattr(est, "formula", None)
    if formula is None:
        return _formula_names(names_in)
    _, lhs = _resolve_formula(formula, "")
    return _check_formula_names(names_in, lhs)


def _validate(est, X, y="no_validation", *, reset, **check_params):
    """Validate `X` (and `y`) the way scikit-learn does, names included.

    This is the whole point of not calling `check_array` directly. `check_array`
    returns a bare ndarray, so a frame's column names are gone before anything
    compares them with the ones the fit saw -- and this module then attached
    the *training* names to the new columns by position. Measured on a fit over
    `a` and `b`: `predict(X[["b", "a"]])` returned -2.119 where `predict(X)`
    returned 1.039, having multiplied `b`'s values by `a`'s coefficient, and
    `predict(X.rename(columns={"a": "z"}))` returned the original numbers as
    though nothing had changed.

    `validate_data` does the name comparison and raises, which is what every
    other scikit-learn estimator does; matching that matters more than being
    accommodating, because a pipeline's other steps behave that way.

    It moved: `BaseEstimator._validate_data` up to scikit-learn 1.5, the free
    function `sklearn.utils.validation.validate_data` from 1.6, and the method
    is gone in 1.9. The package supports >= 1.3, so both are reached here.
    """
    _refuse_non_numeric(X)
    try:
        from sklearn.utils.validation import validate_data
    except ImportError:                                    # scikit-learn < 1.6
        return est._validate_data(X, y, reset=reset, **check_params)
    return validate_data(est, X, y, reset=reset, **check_params)


def _type_of_target(y, input_name="y"):
    """`raise_unknown` arrived after the 1.3 minimum. Without it, an unknown
    target type is *returned* as a string instead of raising, and the callers
    here already reject any type outside the supported ones, so the two
    behaviours agree for every input; fall back rather than pin a version."""
    try:
        return type_of_target(y, input_name=input_name, raise_unknown=True)
    except TypeError:                                      # scikit-learn 1.3
        return type_of_target(y, input_name=input_name)


def _capture_columns(X):
    """Column names if X is a pandas/polars DataFrame, else None (before
    validation coerces it to a bare array)."""
    cols = getattr(X, "columns", None)
    if cols is None:
        return None
    try:
        return [str(c) for c in cols]
    except TypeError:
        return None


def _named_data(X_arr, names):
    return {nm: np.ascontiguousarray(X_arr[:, j], dtype=np.float64) for j, nm in enumerate(names)}


def _resolve_formula(formula, default_rhs):
    """Return ``(full_formula, response_name)``.

    - ``None``                  → ``"y ~ <default_rhs>"``
    - ``"lhs ~ rhs"``           → used as given; response is ``lhs``
    - ``"rhs"`` (no ``~``)      → ``"y ~ rhs"``
    """
    if formula is None:
        return f"{_RESPONSE_NAME} ~ {default_rhs}", _RESPONSE_NAME
    lhs, sep, rhs = partition_tilde(formula)
    if sep:
        return f"{lhs.strip()} ~ {rhs.strip()}", lhs.strip()
    return f"{_RESPONSE_NAME} ~ {formula.strip()}", _RESPONSE_NAME


def _canonical_labels(y_arr, classes):
    """Map class labels onto strings whose lexicographic order is the class order.

    `Multinomial` takes a categorical response and orders its probability
    columns by `sorted(unique(labels))`, using the last as the reference class.
    Passing raw labels through would misalign the columns whenever string order
    differs from `classes_` order — integer labels `[1, 2, 10]` sort as
    `["1", "10", "2"]`. Zero-padded index labels make the two orders identical
    by construction.
    """
    index_of = {cls: i for i, cls in enumerate(classes)}
    return np.array([f"c{index_of[v]:06d}" for v in y_arr], dtype=object)


def _fit_boost(
    est, X, y, *, default_rhs_of, family, default_engine, classifier, sample_weight=None,
    multiclass_rhs_of=None,
):
    """Shared fit body: validate, name columns, build the formula + data, and
    delegate to glmboost/gamboost. `default_rhs_of(names)` builds the no-formula
    right-hand side.

    `sample_weight` maps onto the underlying `weights=` argument, which is a
    per-observation non-negative weight, so it carries the usual meaning for
    case- or survey-weighted fitting."""
    names_in = _capture_columns(X)
    # `validate_data` sets `n_features_in_`, and sets `feature_names_in_` only
    # when the input actually carried names -- which is scikit-learn's
    # convention, and not what this module did: it wrote `['f0', 'f1']` for a
    # bare array, so a fitted estimator claimed feature names it had never
    # been given.
    if classifier:
        X, y_arr = _validate(est, X, y, reset=True, dtype=np.float64)
    else:
        X, y_arr = _validate(est, X, y, reset=True, dtype=np.float64,
                             y_numeric=True)
    p = X.shape[1]
    if names_in is None or len(names_in) != p:
        names_in = [f"{_FEATURE_PREFIX}{j}" for j in range(p)]
    # The formula needs a name for every column whether or not the caller gave
    # one, so the fallback names live here rather than in `feature_names_in_`.
    # Names are settled here so that no feature shares the response's name in
    # the data dictionary the formula is fitted on.
    names_in = _settle_names(est, names_in)
    est._fit_columns_ = names_in

    multiclass = False
    if classifier:
        y_type = _type_of_target(y_arr, input_name="y")
        if y_type not in ("binary", "multiclass"):
            raise ValueError(
                "Only binary and multiclass classification are supported. "
                f"The type of the target is {y_type}."
            )
        est.classes_ = np.unique(y_arr)
        if est.classes_.shape[0] < 2:
            raise ValueError(
                "Classifier can't train when only one class is present."
            )
        multiclass = est.classes_.shape[0] > 2
        if multiclass:
            # Multinomial takes a categorical response directly.
            response_values = _canonical_labels(y_arr, est.classes_)
            family = _families.Multinomial()
        else:
            response_values = (y_arr == est.classes_[1]).astype(np.float64)
    else:
        response_values = np.ascontiguousarray(y_arr, dtype=np.float64)
    # A fitted attribute that does not apply should be absent, not `None`: a
    # regressor carrying `n_classes_ = None` reads as one that was fitted with
    # classes and found none.
    if classifier:
        est.n_classes_ = int(est.classes_.shape[0])

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

    formula = getattr(est, "formula", None)
    if multiclass and formula is None:
        if multiclass_rhs_of is None:
            raise ValueError(
                f"{type(est).__name__} does not define a multiclass default formula"
            )
        rhs = multiclass_rhs_of(names_in)
    else:
        rhs = default_rhs_of(names_in)
    full_formula, response = _resolve_formula(formula, rhs)
    data = _named_data(X, names_in)
    data[response] = response_values

    # A formula may use any learner type, so it must go through gamboost.
    # Multinomial is only implemented for explicit learner terms, so it does
    # too, even for the linear (bols) estimators.
    engine = gamboost if (formula is not None or multiclass) else default_engine
    est.model_ = engine(
        full_formula, data=data, family=family,
        control=boost_control(est.mstop, est.nu),
        weights=sample_weight,
    )
    return est


def _predict_response(est, X):
    check_is_fitted(est, "model_")
    # `reset=False` compares the column names against the ones the fit saw and
    # raises when they differ or are reordered, instead of pairing them by
    # position.
    X = _validate(est, X, reset=False, dtype=np.float64)
    data = _named_data(X, est._fit_columns_)
    return np.asarray(est.model_.predict(newdata=data, type="response"), dtype=np.float64)


def _mark_dense_single_output(tags):
    """Apply the tag facts shared by every estimator in this module, each
    verified against `_fit_boost`'s (or the equivalent `fit`'s) actual
    validation calls rather than asserted:

    - ``input_tags.sparse = False``: ``check_X_y``/``check_array`` are called
      with the default ``accept_sparse=False``, so a sparse ``X`` is rejected
      by sklearn's own validation before it would reach the Rust engine
      (which requires a dense ``f64`` array regardless).
    - ``input_tags.allow_nan = False``: called with the default
      ``ensure_all_finite=True``, so NaN/Inf raise here, at the sklearn
      boundary. The underlying engine's own ``na_action="omit"`` policy
      (``glmboost``/``gamboost``) is a ``formula=``/``data=`` frontend
      feature reached only through the module-level API, not through these
      estimators' ``fit(X, y)`` signature — so it is not exposed here.
    - ``target_tags.multi_output = False`` / ``single_output = True``:
      every ``predict`` in this module returns one 1-D response column;
      none accept or produce a multi-column target.
    """
    tags.input_tags.sparse = False
    tags.input_tags.allow_nan = False
    tags.target_tags.multi_output = False
    tags.target_tags.single_output = True
    return tags


def _reg_family(est):
    """Build the family object for a regressor.

    `quantile` and `expectile` take the estimator's `tau`; the mean-fitting
    families take no argument and reject a non-default `tau` rather than
    silently ignoring it.
    """
    tau = getattr(est, "tau", None)
    tau_cls = _TAU_FAMILIES.get(est.family)
    if tau_cls is not None:
        if tau is None:
            raise ValueError(f"family={est.family!r} requires tau")
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        return tau_cls(tau=tau)

    fam_cls = _REG_FAMILIES.get(est.family)
    if fam_cls is None:
        known = sorted([*_REG_FAMILIES, *_TAU_FAMILIES])
        raise ValueError(f"unknown family {est.family!r}; expected one of {known}")
    if tau is not None and tau != 0.5:
        raise ValueError(
            f"tau={tau} is only meaningful for families "
            f"{sorted(_TAU_FAMILIES)}, not {est.family!r}"
        )
    return fam_cls()


class GLMBoostRegressor(RegressorMixin, BaseEstimator):
    """Component-wise linear boosting (``glmboost``) as an sklearn regressor.
    One linear ``bols`` term per feature.

    Parameters
    ----------
    family : {"gaussian", "poisson", "gamma", "laplace", "huber", "quantile", "expectile"}, default="gaussian"
    tau : float or None, default=None
        Required for ``"quantile"``/``"expectile"``: the level in (0, 1).
        ``predict`` then estimates that conditional quantile/expectile
        rather than the mean.
    mstop : int, default=100
    nu : float, default=0.1

    Notes
    -----
    ``fit`` accepts ``sample_weight``: a non-negative per-observation weight
    passed through to the underlying ``weights=`` argument. A weight of 0
    removes an observation's influence on the loss.

    One caveat, inherited from R ``mboost`` 2.9.11: predictors are centered by
    their *unweighted* column mean, so an integer ``sample_weight`` is not
    exactly the same as physically replicating the row — the two frames get
    different centering constants. Without centering the two are identical to
    machine precision.
    """

    def __init__(self, family="gaussian", mstop=100, nu=0.1, tau=None):
        self.family = family
        self.mstop = mstop
        self.nu = nu
        self.tau = tau

    def fit(self, X, y, sample_weight=None):
        return _fit_boost(
            self, X, y,
            default_rhs_of=lambda names: " + ".join(names),
            family=_reg_family(self), default_engine=glmboost, classifier=False,
            sample_weight=sample_weight,
        )

    def predict(self, X):
        return _predict_response(self, X)

    def __sklearn_tags__(self):
        return _mark_dense_single_output(super().__sklearn_tags__())


class GAMBoostRegressor(RegressorMixin, BaseEstimator):
    """Component-wise additive boosting (``gamboost``) as an sklearn regressor.

    With ``formula=None`` each feature gets a ``bbs(x, df=df)`` smooth term.
    Set ``formula`` to a full mboostrs formula for per-term control.

    Parameters
    ----------
    formula : str or None, default=None
        Full mboostrs formula (e.g. ``"y ~ bbs(age, df=5) + bols(income)"``).
        Overrides the per-feature ``bbs`` default. ``y`` is supplied by ``fit``.
    family : {"gaussian", "poisson", "gamma", "laplace", "huber", "quantile", "expectile"}, default="gaussian"
    tau : float or None, default=None
        Required for ``"quantile"``/``"expectile"``: the level in (0, 1).
        ``predict`` then estimates that conditional quantile/expectile
        rather than the mean.
    df : float, default=4.0
        Degrees of freedom for the default per-feature splines (ignored when a
        ``formula`` is given).
    mstop : int, default=100
    nu : float, default=0.1

    Notes
    -----
    ``fit`` accepts ``sample_weight``: a non-negative per-observation weight
    passed through to the underlying ``weights=`` argument. A weight of 0
    removes an observation's influence on the loss.

    One caveat, inherited from R ``mboost`` 2.9.11: predictors are centered by
    their *unweighted* column mean, so an integer ``sample_weight`` is not
    exactly the same as physically replicating the row — the two frames get
    different centering constants. Without centering the two are identical to
    machine precision.
    """

    def __init__(self, formula=None, family="gaussian", df=4.0, mstop=100, nu=0.1,
                 tau=None):
        self.formula = formula
        self.family = family
        self.df = df
        self.mstop = mstop
        self.nu = nu
        self.tau = tau

    def fit(self, X, y, sample_weight=None):
        return _fit_boost(
            self, X, y,
            default_rhs_of=lambda names: " + ".join(f"bbs({quote_name(nm)}, df={self.df})" for nm in names),
            family=_reg_family(self), default_engine=gamboost, classifier=False,
            sample_weight=sample_weight,
        )

    def predict(self, X):
        return _predict_response(self, X)

    def __sklearn_tags__(self):
        return _mark_dense_single_output(super().__sklearn_tags__())


class _BoostClassifier(ClassifierMixin, BaseEstimator):
    """Shared predict / predict_proba.

    Two classes are fitted with ``Binomial``; more than two with
    ``Multinomial``, whose probability columns are ordered to match
    ``classes_`` (see `_canonical_labels`).
    """

    def predict_proba(self, X):
        response = _predict_response(self, X)
        if response.ndim == 2:
            return response
        return np.column_stack([1.0 - response, response])

    def predict(self, X):
        # Compute predict_proba() (which calls check_is_fitted) before
        # touching self.classes_, so an unfitted call raises NotFittedError
        # rather than a confusing AttributeError from the unset attribute.
        proba = self.predict_proba(X)
        if proba.shape[1] > 2:
            return self.classes_[np.argmax(proba, axis=1)]
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def __sklearn_tags__(self):
        return _mark_dense_single_output(super().__sklearn_tags__())


# Kept as an alias: the class was public-ish and the old name described the
# binary-only behavior that no longer holds.
_BinomialBoostClassifier = _BoostClassifier


class GLMBoostClassifier(_BoostClassifier):
    """Component-wise linear boosting classifier. One linear term per feature.

    Two classes are fitted with ``Binomial``, more than two with ``Multinomial``.

    Parameters
    ----------
    mstop : int, default=100
    nu : float, default=0.1

    Notes
    -----
    ``fit`` accepts ``sample_weight``: a non-negative per-observation weight
    passed through to the underlying ``weights=`` argument. A weight of 0
    removes an observation's influence on the loss.

    One caveat, inherited from R ``mboost`` 2.9.11: predictors are centered by
    their *unweighted* column mean, so an integer ``sample_weight`` is not
    exactly the same as physically replicating the row — the two frames get
    different centering constants. Without centering the two are identical to
    machine precision.
    """

    def __init__(self, mstop=100, nu=0.1):
        self.mstop = mstop
        self.nu = nu

    def fit(self, X, y, sample_weight=None):
        return _fit_boost(
            self, X, y,
            default_rhs_of=lambda names: " + ".join(names),
            multiclass_rhs_of=lambda names: " + ".join(f"bols({quote_name(nm)})" for nm in names),
            family=_families.Binomial(), default_engine=glmboost, classifier=True,
            sample_weight=sample_weight,
        )


class GAMBoostClassifier(_BoostClassifier):
    """Component-wise additive boosting classifier.

    Two classes are fitted with ``Binomial``, more than two with ``Multinomial``.

    With ``formula=None`` each feature gets a ``bbs(x, df=df)`` smooth term.
    Set ``formula`` for per-term control.

    Parameters
    ----------
    formula : str or None, default=None
        Full mboostrs formula; overrides the per-feature ``bbs`` default.
    df : float, default=4.0
    mstop : int, default=100
    nu : float, default=0.1

    Notes
    -----
    ``fit`` accepts ``sample_weight``: a non-negative per-observation weight
    passed through to the underlying ``weights=`` argument. A weight of 0
    removes an observation's influence on the loss.

    One caveat, inherited from R ``mboost`` 2.9.11: predictors are centered by
    their *unweighted* column mean, so an integer ``sample_weight`` is not
    exactly the same as physically replicating the row — the two frames get
    different centering constants. Without centering the two are identical to
    machine precision.
    """

    def __init__(self, formula=None, df=4.0, mstop=100, nu=0.1):
        self.formula = formula
        self.df = df
        self.mstop = mstop
        self.nu = nu

    def fit(self, X, y, sample_weight=None):
        return _fit_boost(
            self, X, y,
            default_rhs_of=lambda names: " + ".join(f"bbs({quote_name(nm)}, df={self.df})" for nm in names),
            multiclass_rhs_of=lambda names: " + ".join(f"bbs({quote_name(nm)}, df={self.df})" for nm in names),
            family=_families.Binomial(), default_engine=gamboost, classifier=True,
            sample_weight=sample_weight,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Measured, not assumed: on sklearn's check_classifiers_train
        # benchmark (make_blobs(n_samples=300, random_state=0)), the binary
        # subset scores 0.97 at the default mstop=100, comfortably above the
        # 0.83 bar — but the 3-class subset scores 0.823, just under it.
        # Multiclass builds (K-1)*p base learners, so 100 component-wise
        # iterations give each component ~25 updates against ~50 in the
        # binary case; accuracy is monotone in mstop (0.823 / 0.893 / 0.920 /
        # 0.933 at 100 / 200 / 400 / 800 — see
        # test_multiclass_gam_accuracy_improves_with_mstop). This tag, not a
        # blanket test skip, is the honest way to say "not guaranteed to
        # clear a generic accuracy bar at default hyperparameters": sklearn's
        # own checks still run fit/predict/predict_proba shape and error
        # handling for this estimator, only the accuracy assertion is
        # waived.
        tags.classifier_tags.poor_score = True
        return tags


# --------------------------------------------------------------------------
# distributional and survival estimators
# --------------------------------------------------------------------------

_LSS_FAMILIES = {
    "gaussian": "GaussianLSS",
    "studentt": "StudentTLSS",
    "gamma": "GammaLSS",
    "negativebinomial": "NBinomialLSS",
}


class LSSRegressor(RegressorMixin, BaseEstimator):
    """Distributional boosting (``glmboostLSS``/``gamboostLSS``) as an estimator.

    Unlike the mean-only regressors this fits every distribution parameter, so
    it can report a predictive interval whose width varies with the covariates
    rather than a single fitted curve.

    ``predict`` returns the fitted conditional mean (the location parameter
    ``mu`` for the supported families, see :meth:`predict`), so the inherited
    R-squared in ``score`` is R-squared of the mean. The distributional output
    is on ``predict_interval`` and ``predict_params``.

    Parameters
    ----------
    formula : str or None, default=None
        Full mboostrs formula. With ``None`` each feature becomes a ``bbs``
        term when ``smooth=True``, else a linear term.
    family : str, default="gaussian"
        One of ``gaussian``, ``studentt``, ``gamma``, ``negativebinomial``.
        Survival families use :class:`SurvivalBoost`
        because they require a structured event/time target rather than a
        numeric regression target.
    smooth : bool, default=False
        Use per-feature P-splines (``gamboostLSS``) instead of linear terms.
    df : float, default=4.0
        Degrees of freedom for the default splines when ``smooth=True``.
    mstop : int, default=100
    nu : float, default=0.1
    method : {"cyclic", "noncyclic"}, default="cyclic"
    """

    def __init__(self, formula=None, family="gaussian", smooth=False, df=4.0,
                 mstop=100, nu=0.1, method="cyclic"):
        self.formula = formula
        self.family = family
        self.smooth = smooth
        self.df = df
        self.mstop = mstop
        self.nu = nu
        self.method = method

    def _family_object(self):
        # LSS families live in mboostrs.gamlss, not mboostrs.families.
        from mboostrs import gamlss as _gamlss

        name = _LSS_FAMILIES.get(self.family)
        if name is None:
            if self.family in ("weibull", "lognormal", "loglog"):
                raise ValueError(
                    f"family={self.family!r} is a censored survival distribution; "
                    "use SurvivalBoost with a structured event/time target"
                )
            raise ValueError(
                f"unknown LSS family {self.family!r}; expected one of "
                f"{sorted(_LSS_FAMILIES)}"
            )
        return getattr(_gamlss, name)()

    def fit(self, X, y, sample_weight=None):
        from mboostrs.gamlss import gamboostLSS, glmboostLSS

        names_in = _capture_columns(X)
        X, y_arr = _validate(self, X, y, reset=True, dtype=np.float64,
                             y_numeric=True)
        p = X.shape[1]
        if names_in is None or len(names_in) != p:
            names_in = [f"{_FEATURE_PREFIX}{j}" for j in range(p)]
        names_in = _settle_names(self, names_in)
        self._fit_columns_ = names_in

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        if self.formula is None:
            rhs = " + ".join(
                f"bbs({quote_name(nm)}, df={self.df})" if self.smooth else quote_name(nm) for nm in names_in
            )
            formula = f"{_RESPONSE_NAME} ~ {rhs}"
            response = _RESPONSE_NAME
        else:
            formula, response = _resolve_formula(self.formula, "")

        data = _named_data(X, names_in)
        data[response] = np.ascontiguousarray(y_arr, dtype=np.float64)

        engine = gamboostLSS if (self.smooth or self.formula is not None) else glmboostLSS
        self.model_ = engine(
            formula, data=data, family=self._family_object(),
            control=boost_control(self.mstop, self.nu),
            method=self.method, weights=sample_weight,
        )
        return self

    def _frame(self, X):
        check_is_fitted(self, "model_")
        X = _validate(self, X, reset=False, dtype=np.float64)
        return _named_data(X, self._fit_columns_)

    def predict(self, X):
        """The fitted conditional mean of the response.

        For ``gaussian``, ``gamma`` and ``negativebinomial`` the family's
        location parameter ``mu`` is the mean. For ``studentt`` it is the mean
        only where ``df > 1``; a row with ``df <= 1`` has no finite mean and
        raises ``ValueError``. Use :meth:`predict_params` for the location and
        scale on such rows.
        """
        # `_frame` calls `check_is_fitted` first, so an unfitted call raises
        # `NotFittedError` rather than an `AttributeError` from touching
        # `self.model_` before validation (caught by sklearn's own
        # `check_estimators_unfitted`).
        frame = self._frame(X)
        params = self.model_.params(newdata=frame)
        family = self.model_._family
        try:
            mean = family.mean(params)
        except ValueError as exc:
            raise ValueError(
                f"{type(family).__name__} has no finite conditional mean for some "
                f"rows ({exc}); use predict_params(X) for the location and scale, "
                "or predict_interval(X) for the predictive spread"
            ) from exc
        return np.asarray(mean, dtype=np.float64).ravel()

    def predict_params(self, X):
        """Every fitted distribution parameter, as a dict of arrays."""
        return self.model_.params(newdata=self._frame(X))

    def predict_interval(self, X, level=0.9):
        """Predictive interval from the fitted distribution.

        Returns ``(lower, upper)``. Width varies with the covariates whenever
        a scale parameter depends on them.
        """
        frame = self.model_.prediction_interval(level=level, newdata=self._frame(X))
        lower = np.asarray(frame["lower"], dtype=np.float64)
        upper = np.asarray(frame["upper"], dtype=np.float64)
        return lower, upper

    def score(self, X, y, sample_weight=None):
        """R-squared of the conditional-mean prediction, per sklearn's regressor API.

        Computed from :meth:`predict`, so it raises the same ``ValueError`` when
        a row has no finite mean (``studentt`` with ``df <= 1``). The value does
        not evaluate fitted scale, shape, tails, or interval quality. Use
        :meth:`distribution_log_score` when selecting a distributional model and
        a proper log score is appropriate.
        """
        return super().score(X, y, sample_weight=sample_weight)

    def distribution_log_score(self, X, y, sample_weight=None):
        """Mean held-out log density or log mass; larger is better.

        This is suitable as a callable sklearn scorer. The supplied ``X, y``
        must be an evaluation split not used for fitting or tuning when the
        value is interpreted as predictive performance.
        """
        frame = self._frame(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(next(iter(frame.values()))),) or not np.all(np.isfinite(y)):
            raise ValueError("y must be one finite value per row of X")
        params = self.model_.params(newdata=frame)
        family = self.model_._family
        density_function = getattr(family, "pdf", None)
        if density_function is None:
            raise NotImplementedError(
                f"{type(family).__name__} does not implement density or mass scoring"
            )
        density = np.asarray(density_function(y, params), dtype=np.float64)
        if density.shape != y.shape or np.any(~np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("distribution score requires finite positive density or mass")
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        else:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        return float(np.average(np.log(density), weights=sample_weight))

    def __sklearn_tags__(self):
        return _mark_dense_single_output(super().__sklearn_tags__())


class SurvivalBoost(BaseEstimator):
    """Survival boosting with a scikit-survival-compatible target.

    ``y`` is a structured array with a boolean event field and a numeric time
    field, the convention used by scikit-survival::

        y = np.array([(True, 5.1), (False, 8.0)],
                     dtype=[("event", bool), ("time", float)])

    ``predict`` returns the linear predictor, which for ``CoxPH`` is the log
    relative risk — higher means higher hazard. That is the orientation
    ``sksurv.metrics.concordance_index_censored`` expects.

    Parameters
    ----------
    formula : str or None, default=None
    family : {"coxph", "weibull", "loglog", "lognormal"}, default="coxph"
    smooth : bool, default=False
    df : float, default=4.0
    mstop : int, default=100
    nu : float, default=0.1

    Notes
    -----
    This estimator does not inherit ``RegressorMixin`` or ``ClassifierMixin``
    and has no ``score`` method: ``predict`` returns a risk score, not an
    estimate of the ``y`` passed to ``fit``, so ``__sklearn_tags__().
    estimator_type`` is honestly ``None`` rather than ``"regressor"``.

    ``fit`` requires a structured-array ``y`` (see above), not the plain
    numeric/categorical target sklearn's generic estimator-check battery
    (``check_estimator`` / ``parametrize_with_checks``) constructs — the same
    reason scikit-survival's own estimators are not run through it. This
    class is therefore verified with dedicated tests
    (``tests/test_sklearn_api.py``) instead.
    """

    _SURV_FAMILIES = {
        "coxph": "CoxPH",
        "weibull": "Weibull",
        "loglog": "Loglog",
        "lognormal": "Lognormal",
    }

    def __init__(self, formula=None, family="coxph", smooth=False, df=4.0,
                 mstop=100, nu=0.1):
        self.formula = formula
        self.family = family
        self.smooth = smooth
        self.df = df
        self.mstop = mstop
        self.nu = nu

    @staticmethod
    def _split_target(y):
        y = np.asarray(y)
        if y.dtype.names is None or len(y.dtype.names) != 2:
            raise ValueError(
                "y must be a structured array with an event field and a time "
                "field, e.g. dtype=[('event', bool), ('time', float)]"
            )
        names = y.dtype.names
        if "event" in names and "time" in names:
            event_name, time_name = "event", "time"
        else:
            boolean = [name for name in names if y.dtype[name].kind == "b"]
            if len(boolean) != 1:
                raise ValueError(
                    "structured y must name event/time fields or contain exactly "
                    "one boolean event field"
                )
            event_name = boolean[0]
            time_name = names[0] if names[1] == event_name else names[1]
        event = np.asarray(y[event_name]).astype(np.float64)
        time = np.asarray(y[time_name]).astype(np.float64)
        if not np.all(np.isin(event, (0.0, 1.0))):
            raise ValueError(f"event field {event_name!r} must be boolean or 0/1")
        if np.any(~np.isfinite(time)) or np.any(time < 0):
            raise ValueError(f"time field {time_name!r} must be finite and non-negative")
        return time, event

    def fit(self, X, y, sample_weight=None):
        time, event = self._split_target(y)
        names_in = _capture_columns(X)
        # `y` here is a structured (time, event) target, which `validate_data`
        # cannot check, so only `X` goes through it.
        X = _validate(self, X, reset=True, dtype=np.float64)
        if X.shape[0] != time.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {time.shape[0]}"
            )
        p = X.shape[1]
        if names_in is None or len(names_in) != p:
            names_in = [f"{_FEATURE_PREFIX}{j}" for j in range(p)]
        names_in = _settle_names(self, names_in)
        self._fit_columns_ = names_in

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        family_name = self._SURV_FAMILIES.get(self.family)
        if family_name is None:
            raise ValueError(
                f"unknown survival family {self.family!r}; expected one of "
                f"{sorted(self._SURV_FAMILIES)}"
            )

        if self.formula is None:
            rhs = " + ".join(
                f"bbs({quote_name(nm)}, df={self.df})" if self.smooth else f"bols({quote_name(nm)})"
                for nm in names_in
            )
        else:
            _lhs, separator, supplied_rhs = partition_tilde(self.formula)
            rhs = supplied_rhs.strip() if separator else self.formula.strip()
            if not rhs:
                raise ValueError("survival formula must contain a non-empty right-hand side")
        formula = f"Surv(_time, _event) ~ {rhs}"
        data = _named_data(X, names_in)
        data["_time"] = time
        data["_event"] = event

        self.model_ = gamboost(
            formula, data=data, family=getattr(_families, family_name)(),
            control=boost_control(self.mstop, self.nu), weights=sample_weight,
        )
        return self

    def predict(self, X):
        """Linear predictor: log relative risk for ``CoxPH``."""
        check_is_fitted(self, "model_")
        X = _validate(self, X, reset=False, dtype=np.float64)
        data = _named_data(X, self._fit_columns_)
        return np.asarray(self.model_.predict(newdata=data, type="link"),
                          dtype=np.float64).ravel()

    def __sklearn_tags__(self):
        tags = _mark_dense_single_output(super().__sklearn_tags__())
        # BaseEstimator's default is `required=False`; `fit` here always
        # raises without a structured-array `y` (see `_split_target`), so
        # leaving the base default would misstate the contract.
        tags.target_tags.required = True
        return tags


class TreeBoostRegressor(RegressorMixin, BaseEstimator):
    """Component-wise boosting with one ``btree`` term per feature.

    Deliberately not named ``BlackBoost``. R's ``blackboost`` fits partykit
    conditional-inference trees, which this does not: it composes the package's
    own ``btree`` learner, and the capability manifest records ``blackboost``
    itself as unsupported. Naming this after it would advertise upstream
    semantics that are not implemented.

    Parameters
    ----------
    formula : str or None, default=None
    family : {"gaussian", "poisson", "gamma", "laplace", "huber", "quantile", "expectile"}, default="gaussian"
    tau : float or None, default=None
    maxdepth : int, default=2
    minsplit : int, default=20
    minbucket : int, default=10
    mstop : int, default=100
    nu : float, default=0.1
    """

    def __init__(self, formula=None, family="gaussian", tau=None, maxdepth=2,
                 minsplit=20, minbucket=10, mstop=100, nu=0.1):
        self.formula = formula
        self.family = family
        self.tau = tau
        self.maxdepth = maxdepth
        self.minsplit = minsplit
        self.minbucket = minbucket
        self.mstop = mstop
        self.nu = nu

    def fit(self, X, y, sample_weight=None):
        # Capture the names *before* validation: validation returns an array,
        # so reading them afterwards always found none, and every tree fit was
        # built over `f0, f1, ...` whatever the frame was called -- which made
        # an explicit `formula="y ~ btree(a)"` fail on a frame with a column
        # named `a`.
        names_in = _capture_columns(X)
        X, y_arr = _validate(self, X, y, reset=True, dtype=np.float64,
                             y_numeric=True)
        p = X.shape[1]
        if names_in is None or len(names_in) != p:
            names_in = [f"{_FEATURE_PREFIX}{j}" for j in range(p)]
        names_in = _settle_names(self, names_in)
        self._fit_columns_ = names_in

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        if self.formula is None:
            rhs = " + ".join(
                f"btree({quote_name(nm)}, maxdepth={self.maxdepth}, minsplit={self.minsplit}, "
                f"minbucket={self.minbucket})"
                for nm in names_in
            )
            formula = f"{_RESPONSE_NAME} ~ {rhs}"
            response = _RESPONSE_NAME
        else:
            formula, response = _resolve_formula(self.formula, "")

        data = _named_data(X, names_in)
        data[response] = y_arr

        self.model_ = gamboost(
            formula,
            data=data,
            family=_reg_family(self),
            control=boost_control(self.mstop, self.nu),
            weights=sample_weight,
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = _validate(self, X, reset=False, dtype=np.float64)
        data = _named_data(X, self._fit_columns_)
        return np.asarray(self.model_.predict(newdata=data, type="response"), dtype=np.float64).ravel()

    def __sklearn_tags__(self):
        return _mark_dense_single_output(super().__sklearn_tags__())


class TreeBoostClassifier(ClassifierMixin, BaseEstimator):
    """Component-wise ``btree`` boosting as an sklearn classifier.

    See :class:`TreeBoostRegressor` on why this is not called ``BlackBoost``.

    Parameters
    ----------
    formula : str or None, default=None
    family : {"binomial", "adaexp"}, default="binomial"
    maxdepth : int, default=2
    minsplit : int, default=20
    minbucket : int, default=10
    mstop : int, default=100
    nu : float, default=0.1
    """

    _CLS_FAMILIES = {
        "binomial": _families.Binomial,
        "adaexp": _families.AdaExp,
    }

    def __init__(self, formula=None, family="binomial", maxdepth=2,
                 minsplit=20, minbucket=10, mstop=100, nu=0.1):
        self.formula = formula
        self.family = family
        self.maxdepth = maxdepth
        self.minsplit = minsplit
        self.minbucket = minbucket
        self.mstop = mstop
        self.nu = nu

    def fit(self, X, y, sample_weight=None):
        names_in = _capture_columns(X)
        X, y_arr = _validate(self, X, y, reset=True, dtype=np.float64)
        # sklearn's check_classifier_not_supporting_multiclass requires this
        # exact wording from an estimator declaring classifier_tags.multi_class
        # = False, so keep the phrase intact.
        y_type = _type_of_target(y_arr, input_name="y")
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}. Multiclass would need the Multinomial family, "
                "which accepts only bols() and bbs() base learners, not btree()."
            )

        self.classes_, y_coded = np.unique(y_arr, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classifier can't train when only one class is present.")

        p = X.shape[1]
        if names_in is None or len(names_in) != p:
            names_in = [f"{_FEATURE_PREFIX}{j}" for j in range(p)]
        names_in = _settle_names(self, names_in)
        self._fit_columns_ = names_in

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        family_cls = self._CLS_FAMILIES.get(self.family)
        if family_cls is None:
            raise ValueError(f"unknown family {self.family!r}; expected one of {sorted(self._CLS_FAMILIES)}")

        if self.formula is None:
            rhs = " + ".join(
                f"btree({quote_name(nm)}, maxdepth={self.maxdepth}, minsplit={self.minsplit}, "
                f"minbucket={self.minbucket})"
                for nm in names_in
            )
            formula = f"{_RESPONSE_NAME} ~ {rhs}"
            response = _RESPONSE_NAME
        else:
            formula, response = _resolve_formula(self.formula, "")

        data = _named_data(X, names_in)
        data[response] = y_coded.astype(np.float64)

        self.model_ = gamboost(
            formula,
            data=data,
            family=family_cls(),
            control=boost_control(self.mstop, self.nu),
            weights=sample_weight,
        )
        return self

    def decision_function(self, X):
        """Additive classification margin; zero is the class boundary."""
        check_is_fitted(self, "model_")
        X = _validate(self, X, reset=False, dtype=np.float64)
        data = _named_data(X, self._fit_columns_)
        return np.asarray(
            self.model_.predict(newdata=data, type="link"), dtype=np.float64
        ).ravel()

    @available_if(lambda estimator: estimator.family == "binomial")
    def predict_proba(self, X):
        """Binomial response probabilities; unavailable for AdaExp margins."""
        check_is_fitted(self, "model_")
        X = _validate(self, X, reset=False, dtype=np.float64)
        data = _named_data(X, self._fit_columns_)
        p1 = np.asarray(self.model_.predict(newdata=data, type="response"), dtype=np.float64).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        margin = self.decision_function(X)  # checks fitted state before classes_
        return self.classes_[(margin >= 0).astype(int)]

    def __sklearn_tags__(self):
        tags = _mark_dense_single_output(super().__sklearn_tags__())
        # Binary only, and by engine constraint rather than omission: the
        # Multinomial family accepts bols() and bbs() base learners only, so a
        # btree-based multiclass fit is not expressible. Declaring it here makes
        # sklearn's check battery supply binary targets instead of 3-class ones.
        tags.classifier_tags.multi_class = False
        return tags
