"""GLMBoost model class and glmboost() function."""

from __future__ import annotations

import numbers
import sys
import warnings

import numpy as np

from mboostrs._comparison_identity import fit_identity
from mboostrs._control import as_control, step_kwargs
from mboostrs._errors import MboostrsKeyError
from mboostrs._frame import column_names
from mboostrs._model_api import ModelAPIMixin, capture_fit_warnings
from mboostrs._names import quote_name
from mboostrs._results import AicResult, provenance_from_model
from mboostrs._weights import normalize_weights
from mboostrs._workflow import call_fit, fit_checkpoint
from mboostrs.cvrisk import (
    CvRiskResult,
    _as_fold_matrix,
    _fold_design,
    data_fingerprint,
    fold_fingerprint,
    fold_label_for,
    held_out_fraction,
    warn_if_at_boundary,
    warn_if_one_fold_dominates,
)
from mboostrs.families import (
    _TYPE_DEFAULT,
    Binomial,
    Cindex,
    CoxPH,
    Gaussian,
    Gehan,
    Loglog,
    Lognormal,
    Weibull,
    _family_params,
    _family_rust_name,
    _inverse_link,
    maybe_warn_link_scale,
    prediction_scale_text,
)
from mboostrs.folds import FoldScheme
from mboostrs.formula import expand_formula, parse_formula
from mboostrs.interactions import reject_factor_by
from mboostrs.report import ExecutionReport, path_rows_output, standard_path_rows
from mboostrs.transforms import (
    TransformRecipe,
    apply_recipe,
    encode_factors,
    expand_factor_dummies,
    extract_transforms,
    fit_recipe,
    impute_missing,
)
from mboostrs.transforms import display_names as _display_names

try:
    from mboostrs._rust import (
        boost_control as _boost_control,
    )
    from mboostrs._rust import (
        coef_glmboost_py as _coef_glmboost,
    )
    from mboostrs._rust import (
        cv_weights as _cv_weights,
    )
    from mboostrs._rust import (
        cvrisk_glmboost as _cvrisk_glmboost,
    )
    from mboostrs._rust import (
        fit_glmboost as _fit_glmboost,
    )
    from mboostrs._rust import (
        predict_glmboost as _predict_glmboost,
    )
    from mboostrs._rust import (
        predict_glmboost_path as _predict_glmboost_path,
    )

    _HAS_RUST_ENGINE = True
except ImportError:
    _HAS_RUST_ENGINE = False
    _boost_control = None
    _fit_glmboost = None
    _predict_glmboost = None
    _predict_glmboost_path = None
    _coef_glmboost = None
    _cv_weights = None
    _cvrisk_glmboost = None


class GLMBoostModel(ModelAPIMixin):
    """Fitted gradient boosting model."""

    def __init__(
        self,
        formula: str,
        family,
        control: tuple,
        response_name: str,
        learner_names: list[str],
        fitted: np.ndarray,
        risk: np.ndarray,
        selected: list[int],
        coefficients: list[np.ndarray],
        offset: float,
        nu: float,
        data=None,
        weights: np.ndarray | None = None,
        predictor_names: list[str] | None = None,
        centering_means: list[float] | None = None,
        nuisance_vec: list[float] | None = None,
        nuisance_vec_path: list[list[float]] | None = None,
        learner_store=None,
        execution_report: dict | ExecutionReport | None = None,
    ):
        self._formula = formula
        self._family = family
        self._control = control
        self._response_name = response_name
        self._learner_names = learner_names
        self._fitted = fitted
        self._risk = risk
        self._selected = selected
        self._coefficients = coefficients
        self._offset = offset
        self._nu = nu
        self._data = data
        self._weights = weights
        self._predictor_names = predictor_names
        self._centering_means = centering_means
        self._nuisance_vec = list(nuisance_vec) if nuisance_vec is not None else []
        self._nuisance_vec_path = list(nuisance_vec_path) if nuisance_vec_path is not None else []
        self._learner_store = learner_store
        self._execution_report = (
            execution_report
            if isinstance(execution_report, ExecutionReport)
            else ExecutionReport.from_mapping(execution_report)
            if execution_report is not None
            else None
        )


    def path_frame(self, output="auto"):
        """Return one structured row per boosting iteration.

        ``output`` may be ``"auto"``, ``"pandas"``, ``"polars"``, or
        ``"records"``. Rows are derived on demand, so fitting overhead is
        unchanged when this method is unused.
        """
        rows = standard_path_rows(
            selected=self._selected,
            coefficients=self._coefficients,
            risk=self._risk,
            nu=self._nu,
            learner_names=self._learner_names,
            nuisance_path=self._nuisance_vec_path,
        )
        return path_rows_output(rows, output)




    @property
    def family(self):
        """The response family the model was fitted with."""
        return self._family

    @property
    def formula(self) -> str:
        """The formula as it was written."""
        bundle = getattr(self, "_refit_bundle", None) or {}
        return bundle.get("raw_formula") or self._formula

    @property
    def fitted_(self) -> np.ndarray:
        """The fitted linear predictor.

        Read-only: this is the fit's own array, not a copy. See
        :mod:`mboostrs._state`.
        """
        from mboostrs._state import readonly

        return readonly(self._fitted)

    def fitted(self, mstop=None) -> np.ndarray:
        """Fitted values, optionally at a given mstop.

        mstop: int or None. If None, returns fitted values at the full mstop.
               If given, recomputes fitted values at iteration `mstop`.
        """
        if mstop is None or mstop >= self.mstop:
            return self._fitted.copy()
        return self._predict_training(mstop)



    @property
    def coefficients_(self) -> dict[str, float]:
        coefs: dict[str, float] = {}
        for name in self._learner_names:
            coefs[name] = 0.0
        for idx, coef in zip(self._selected, self._coefficients, strict=False):
            name = self._learner_names[idx]
            coefs[name] += self._nu * coef.item()
        return {k: v for k, v in coefs.items() if v != 0.0}

    def _selected_rust(self) -> list[int]:
        """The boosting path in learner-store indices. The store's learner
        order is the design's column order, so the fit's ``selected`` is
        already in store indices."""
        return list(self._selected)

    def _coefs_rust(self) -> list[list[float]]:
        """The per-step unshrunken updates as one vector per step: every
        glmboost learner is a single column, so each step's update carries
        one coefficient."""
        return [np.asarray(c, dtype=float).ravel().tolist()
                for c in self._coefficients]

    def _rust_to_formula(self, rust_idx: int) -> int:
        """Identity: the store's learner order is the learner-name order."""
        return rust_idx

    def concurvity(self, pairwise=False, output="auto"):
        """Concurvity of the fitted terms, as mgcv's ``concurvity`` measures
        it.

        Concurvity is the additive-model form of collinearity: how far one
        term's function space reaches into another's, which is when partial
        effects stop being separately identified -- the same fit can be
        redistributed between the terms, so their coefficients and the
        selection that chose them are unstable (Wood, *Generalized Additive
        Models*, 2nd ed., section 5.6.2). A glmboost term is one column, so
        its function space is the multiples of that column and each row of
        the default table is ordinary collinearity: the term's centered
        column measured against the intercept and every other centered
        column (mgcv's ``concurvity(full = TRUE)``). Each row gives three
        values from 0 (the column is orthogonal to the others) to 1 (it lies
        in their span), where near-collinear columns can overshoot 1
        slightly from the numerics: ``worst``, the largest fraction of the
        column's squared norm the other terms can reproduce over every
        coefficient vector; ``observed``, the same fraction at the fitted
        coefficients (NaN for a term that was never selected); and
        ``estimate``, an average over coefficient directions. For a single
        column the three coincide up to rounding.

        With ``pairwise=True`` the rows give mgcv's
        ``concurvity(full = FALSE)``: one row per ordered pair, where
        ``term`` is measured against ``against`` alone, no intercept
        involved. Each column is first centered, as mgcv measures its
        constrained smooths, so the intercept learner (fitted whenever the
        formula has an intercept) is constant after centering and reports
        NaN in the default table, and every pairwise row naming it too.

        Parameters
        ----------
        pairwise : bool
            Term-against-the-rest rows (the default) or one row per
            ordered pair of terms.
        output : {"auto", "records", "polars", "pandas", "frame"}
            The return format; see :func:`mboostrs.glmboost` for the
            convention.
        """
        if self._learner_store is None:
            raise ValueError(
                "concurvity() needs the fitted learners' designs, and this "
                "model carries no learner store (a model restored from a "
                "pickle written before the store was kept, or a fit whose "
                "store was dropped). Refit the model to measure it")
        from mboostrs._model_api import concurvity_rows
        from mboostrs._output import output_frame

        return output_frame(concurvity_rows(self, pairwise=pairwise), output)


    # ── Milestone 8: extraction / inspection ──────────────────────

    def variable_names(self):
        """Names of all variables used in the model (response + predictors)."""
        names = []
        if self._response_name is not None:
            resp = self._response_name
            if isinstance(resp, str):
                names.append(resp)
            else:
                names.extend(resp)
        if self._predictor_names is not None:
            names.extend(self._predictor_names)
        return names

    def base_learner_names(self):
        """Names of all base-learners (including intercept if present)."""
        return list(self._learner_names)



    def model_matrix(self, which=None):
        """Return the design matrix (n × p) for the specified predictors.

        Parameters
        ----------
        which : str, int, or list thereof, optional
            Predictor name(s) or index. None = all predictors.
        """
        if self._data is None or self._predictor_names is None:
            raise ValueError("model_matrix() requires stored data/predictor names")
        if which is None:
            names = self._predictor_names
        elif isinstance(which, (str, int)):
            names = [self._predictor_names[which] if isinstance(which, int) else which]
        else:
            names = [self._predictor_names[w] if isinstance(w, int) else w for w in which]
        cols = [_extract_column(self._data, n) for n in names]
        return np.column_stack(cols)



    def nuisance(self):
        """The family's nuisance parameter at the model's step count, as R's
        ``nuisance()`` returns it (the size of ``NBinomial``, ``Var = mu + mu**2 / sigma``, the shape of
        ``GammaReg``, the thresholds of ``PropOdds``); None when there is none.

        A family with no nuisance parameter stores an empty path entry. That
        is ``None``, not a zero-length array: ``gamboost`` already returned
        ``None``, and ``nuisance()[0]`` on an empty array raises ``IndexError``.
        """
        if self._nuisance_vec_path and self.mstop < len(self._nuisance_vec_path):
            value = self._nuisance_vec_path[self.mstop]
        elif self._nuisance_vec:
            value = self._nuisance_vec
        else:
            return None
        array = np.asarray(value, dtype=np.float64).copy()
        if array.size == 0:
            return None
        return array


    def predict(self, newdata=None, *, mstop=None, type=_TYPE_DEFAULT, which=None,
                aggregate="sum", offset=None):
        """Predicted values, optionally decomposed by learner.

        Parameters
        ----------
        newdata : dict or DataFrame, optional
            New observations. If None, predicts on the training data.
        offset : number or array-like, optional
            The offset these rows start from. A model fitted with a
            per-observation offset needs the offset of every *new* row, so
            predicting ``newdata`` requires this argument; the training rows
            reuse the fitted vector.
        mstop : int, optional
            Truncate the boosting path at this iteration.
        type : {"link", "response", "class", "log_odds"}, default "link"
            Link-scale or response-scale predictions. The default is the link
            scale (R ``mboost`` parity): the first default-scale call on a
            non-identity-link family warns once per fitted model, naming the
            scale, because a log-mean is not a mean. ``"log_odds"`` is the
            link scale times 2 for ``Binomial``, whose link is half the
            log-odds; with ``which=`` it gives each learner's contribution to
            the log-odds, the scale other libraries report.
            ``"class"`` returns 0/1 class codes for Binomial, AdaExp,
            HingeLoss and AUC. A score on the decision boundary returns 0.
        which : int, str, or list thereof, optional
            If given, return only the contribution of the specified learner(s).
            Integers are learner indices; strings are learner names.
        aggregate : {"sum", "cumsum", "none"}
            "sum" returns the aggregated contribution.
            "cumsum" returns cumulative contributions at each step.
            "none" returns per-step contributions.
        """
        if type_was_default := type is _TYPE_DEFAULT:
            type = "link"
        if isinstance(mstop, numbers.Integral) and mstop > self.mstop:
            raise ValueError(
                f"predict(mstop={int(mstop)}) is beyond the model's fitted {self.mstop} "
                "steps. The prediction would silently use the last fitted step. Refit the "
                f"model with mstop={int(mstop)}, or pass mstop <= {self.mstop}."
            )
        if type == "class":
            from mboostrs.families import _class_prediction

            if which is not None or aggregate != "sum":
                raise ValueError("type='class' requires the whole model and aggregate='sum'")
            return _class_prediction(self.predict(
                newdata=newdata, mstop=mstop, type="link", offset=offset), self._family)
        if type == "log_odds":
            from mboostrs.families import log_odds_factor, scale_output

            factor = log_odds_factor(self._family)
            return scale_output(self.predict(
                newdata=newdata, mstop=mstop, type="link", which=which,
                aggregate=aggregate, offset=offset), factor)
        newdata = _replay_transforms(self, newdata)
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link', 'response', 'class' or 'log_odds', got {type!r}")
        maybe_warn_link_scale(self, type_was_default, method="predict")
        if aggregate not in ("sum", "cumsum", "none"):
            raise ValueError(f"aggregate must be 'sum', 'cumsum', or 'none', got {aggregate!r}")

        if which is not None or aggregate != "sum":
            return self._prediction_contributions(
                newdata=newdata,
                mstop=mstop,
                type=type,
                which=which,
                aggregate=aggregate,
                offset=offset,
            )

        if newdata is None:
            k = mstop if mstop is not None else self.mstop
            if k >= self.mstop:
                eta = self._fitted.copy()
            else:
                # Reconstruct sliced fitted by calling predict on the training X.
                eta = self._predict_training(k)
        else:
            if self._predictor_names is None:
                raise ValueError("Cannot predict on new data: model was fitted without storing predictor names")
            x_cols = []
            for name in self._predictor_names:
                col = _extract_column(newdata, name)
                x_cols.append(col)
            x_matrix = _stack_columns(x_cols, _frame_length(newdata, "newdata"))
            effective_offset = self._prediction_offset(
                offset, len(x_matrix), requires_explicit=newdata is not None)
            coefs_flat = [c.item() for c in self._coefficients]
            eta = np.asarray(
                _predict_glmboost(
                    effective_offset,
                    self._core_selected(),
                    coefs_flat,
                    self._nu,
                    x_matrix,
                    mstop,
                    self._centering_means,
                ),
                dtype=np.float64,
            )

        if type == "response":
            from mboostrs.families import PropOdds

            if isinstance(self._family, PropOdds):
                # Per-category probabilities P(Y=k), an (n, K) matrix, computed
                # from the fitted K-1 thresholds at the requested iteration.
                # Shift-invariant, so it matches R even though the raw link
                # may differ by a constant. Use path-specific thresholds when
                # available so truncated prediction matches a separate mstop=k fit.
                from mboostrs._rust import propodds_response as _propodds_response

                k_effective = mstop if mstop is not None else self.mstop
                if self._nuisance_vec_path and k_effective >= 0 and k_effective < len(self._nuisance_vec_path):
                    sigma = self._nuisance_vec_path[k_effective]
                else:
                    sigma = self._nuisance_vec
                k_cat = len(sigma) + 1
                flat = _propodds_response(eta.astype(np.float64).tolist(), list(sigma))
                return np.asarray(flat, dtype=np.float64).reshape(len(eta), k_cat)
            return _inverse_link(eta, self._family)
        return eta

    def extrapolation_report(self, newdata, output="auto"):
        """Predictors with rows of `newdata` outside their training range.

        A GLM has no spline learners, so every term is linear. One row per
        predictor with values outside its training range, with the columns of
        :meth:`GAMBoostModel.extrapolation_report`; `kind` is ``"linear"``. A
        linear term continues the fitted line. `predict()` does not clip it, so
        this report is the only check that the line is being used where the
        data supports it. `worst_distance` is in training ranges.

        Needs the training data the model stores; a model restored from a
        pickle carries none and raises here.
        """
        from mboostrs._extrapolation import REPORT_COLUMNS, report_records, row_for
        from mboostrs._model_api import STANDARD_PREDICTOR
        from mboostrs._output import output_frame

        if self._data is None or self._predictor_names is None:
            raise ValueError(
                "extrapolation_report() needs the stored training data and predictor "
                "names, which this model does not carry (it was restored from a pickle "
                "or built without them). Refit the model to get the report.")
        newdata = _replay_transforms(self, newdata, "average")
        entries = []
        for name in self._predictor_names:
            training = np.asarray(_extract_column(self._data, name), dtype=np.float64)
            training = training[np.isfinite(training)]
            values = np.asarray(_extract_column(newdata, name), dtype=np.float64)
            entries.append((STANDARD_PREDICTOR, "linear", row_for(
                f"bols({quote_name(name)})", name, values,
                float(training.min()), float(training.max()))))
        return output_frame(report_records(entries), output, columns=REPORT_COLUMNS)

    def predict_path(self, grid, *, newdata=None, type="link"):
        """Predicted values at every `mstop` in `grid`, in one pass.

        Equivalent to ``[self.predict(newdata=newdata, mstop=k, type=type)
        for k in grid]`` but shares one incremental sweep over the boosting
        path instead of redoing the full design-application loop per grid
        point — the difference matters for sweeps with many grid points
        (path plots, manual early-stopping inspection) on models with many
        candidate predictors. Only the ``aggregate="sum"``/no-``which``
        case is supported; use `predict()` per grid point for per-learner
        contributions.

        Parameters
        ----------
        grid : sequence of int
            `mstop` values to evaluate. Need not be sorted; values above
            `self.mstop` are clamped to the full path.
        newdata : dict or DataFrame, optional
            New observations. If None, predicts on the training data.
        type : {"link", "response"}
            Link-scale or response-scale predictions.

        Returns
        -------
        np.ndarray of shape (len(grid), n)
        """
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got {type!r}")
        if _predict_glmboost_path is None:
            return np.stack([self.predict(newdata=newdata, mstop=k, type="link") for k in grid])

        source = self._data if newdata is None else newdata
        if self._predictor_names is None:
            raise ValueError("Cannot predict on new data: model was fitted without storing predictor names")
        x_cols = [_extract_column(source, name) for name in self._predictor_names]
        x_matrix = _stack_columns(x_cols, _frame_length(source, "newdata"))
        coefs_flat = [c.item() for c in self._coefficients]
        grid_list = [int(k) for k in grid]
        eta = np.asarray(
            _predict_glmboost_path(
                self._offset,
                self._core_selected(),
                coefs_flat,
                self._nu,
                x_matrix,
                grid_list,
                self._centering_means,
            ),
            dtype=np.float64,
        )
        if type == "response":
            return np.stack([_inverse_link(row, self._family) for row in eta])
        return eta

    def _predict_training(self, k: int) -> np.ndarray:
        """Recompute fitted values on the training X at iteration `k`.

        Used when `__getitem__`/`predict` requests a sliced fit. Requires that
        the model was constructed with `data` + `predictor_names` (the default
        for models built by `glmboost()`).
        """
        if self._data is None or self._predictor_names is None:
            # Best-effort fallback: return full-mstop fitted (caller asked for
            # sliced fit but model wasn't built with retained training data).
            return self._fitted.copy()
        x_cols = [_extract_column(self._data, name) for name in self._predictor_names]
        x_matrix = _stack_columns(x_cols, len(self._fitted))
        coefs_flat = [c.item() for c in self._coefficients]
        return np.asarray(
            _predict_glmboost(
                self._offset,
                self._core_selected(),
                coefs_flat,
                self._nu,
                x_matrix,
                k,
                self._centering_means,
            ),
            dtype=np.float64,
        )

    def _data_fingerprint(self):
        """`data_fingerprint` of the response and weights `cvrisk` fits."""
        if self._data is None:
            return ""
        y = _extract_response(self._data, self._response_name, self._family)
        w = self._weights if self._weights is not None else np.ones(len(self._fitted))
        return data_fingerprint(y, w)

    def _prediction_offset(self, offset, n_rows, *, requires_explicit):
        """The offset a prediction starts from, validated against its rows."""
        return _resolve_prediction_offset(
            self._offset, offset, self._family, n_rows,
            requires_explicit=requires_explicit)

    def _prediction_contributions(self, *, newdata=None, mstop=None, type="link", which=None,
                                   aggregate="sum", offset=None):
        """Decompose predictions into per-learner contributions.

        For GLMBoostModel each learner is a scalar predictor, so the
        contribution at step *s* is ``nu * coef_s * (x_{j(s)} - mean_{j(s)})``.
        """
        k = mstop if mstop is not None else self.mstop
        selected = self._selected[:k]
        coefs = [c.item() for c in self._coefficients[:k]]
        nu = self._nu

        # Resolve which learners to include.
        if which is not None:
            if isinstance(which, int | str):
                which = [which]
            allowed_idx = set()
            for w in which:
                if isinstance(w, str):
                    allowed_idx.add(self._learner_names.index(w))
                else:
                    allowed_idx.add(w)
        else:
            allowed_idx = None

        # Build the design matrix (same logic as predict()).
        if newdata is None:
            if self._data is not None and self._predictor_names is not None:
                x_cols = [_extract_column(self._data, name) for name in self._predictor_names]
            else:
                x_cols = [_extract_column(self._data, name) for name in self._predictor_names]
        else:
            x_cols = [_extract_column(newdata, name) for name in self._predictor_names]
        n_rows = len(self._fitted) if newdata is None else _frame_length(newdata, "newdata")
        x_matrix = _stack_columns(x_cols, n_rows).astype(np.float64)
        n = x_matrix.shape[0]
        effective_offset = self._prediction_offset(
            offset, n, requires_explicit=newdata is not None)
        means = self._centering_means or [0.0] * len(self._learner_names)

        # Per-step contributions: (k, n) matrix.
        # Learner index 0 = intercept (constant); 1..p map to predictor cols
        # 0..p-1. A fit without an intercept (`y ~ x - 1`) numbers the
        # predictors from 0.
        p = x_matrix.shape[1]
        first = 1 if getattr(self, "_intercept", True) else 0
        step_contrib = np.zeros((len(selected), n), dtype=np.float64)
        for s, (learner_idx, coef) in enumerate(zip(selected, coefs, strict=False)):
            if allowed_idx is not None and learner_idx not in allowed_idx:
                continue
            if learner_idx < first:
                # Intercept: constant contribution (no x column).
                step_contrib[s] = nu * coef
            elif learner_idx - first < p:
                col = learner_idx - first
                centering = means[col] if col < len(means) else 0.0
                step_contrib[s] = nu * coef * (x_matrix[:, col] - centering)

        if aggregate == "none":
            result = step_contrib
        elif aggregate == "cumsum":
            result = np.cumsum(step_contrib, axis=0)
            if which is None:
                result = result + effective_offset
        else:  # "sum"
            if which is not None and allowed_idx is not None:
                result = step_contrib.sum(axis=0)
            else:
                result = effective_offset + step_contrib.sum(axis=0)

        if type == "response":
            from mboostrs.families import PropOdds

            if aggregate == "none":
                raise ValueError(
                    "type='response' with aggregate='none' is not supported; "
                    "contributions in response space are not additive"
                )
            if isinstance(self._family, PropOdds):
                raise ValueError("type='response' with contributions not supported for PropOdds")
            if aggregate == "cumsum" and which is not None:
                raise ValueError("type='response' with aggregate='cumsum' requires which=None (full model)")
            result = _inverse_link(result, self._family)
        return result

    def prediction_contributions(self, *, newdata=None, mstop=None, which=None, aggregate="sum"):
        """Per-learner prediction contributions.

        Parameters
        ----------
        newdata : dict or DataFrame, optional
        mstop : int, optional
        which : int, str, or list thereof, optional
            Restrict to specific learner(s).
        aggregate : {"sum", "cumsum", "none"}

        Returns
        -------
        ndarray
            For ``aggregate="sum"``: 1-D array of length n.
            For ``aggregate="cumsum"``: 2-D array of shape (mstop, n).
            For ``aggregate="none"``: 2-D array of shape (mstop, n).
        """
        return self._prediction_contributions(
            newdata=newdata,
            mstop=mstop,
            type="link",
            which=which,
            aggregate=aggregate,
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is not None and key.start != 0:
                raise ValueError("Only slicing from the start (model[:k]) is supported")
            k = key.stop
            if k is None:
                k = self.mstop
            if k < 0:
                k = max(0, self.mstop + k)
            k = min(k, self.mstop)
        elif isinstance(key, int):
            if key < 0:
                key = self.mstop + key
            k = key
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

        if k <= 0:
            raise ValueError("Cannot slice to 0 or negative iterations")

        new_selected = self._selected[:k]
        new_coefficients = self._coefficients[:k]
        new_risk = self._risk[: k + 1]

        if k >= self.mstop:
            new_fitted = self._fitted.copy()
        else:
            new_fitted = self._predict_training(k)

        _sliced = GLMBoostModel(
            formula=self._formula,
            family=self._family,
            control=self._control,
            response_name=self._response_name,
            learner_names=self._learner_names,
            fitted=new_fitted,
            risk=new_risk,
            selected=new_selected,
            coefficients=new_coefficients,
            offset=self._offset,
            nu=self._nu,
            data=self._data,
            weights=self._weights,
            predictor_names=self._predictor_names,
            centering_means=self._centering_means,
            learner_store=self._learner_store,
            execution_report=(self._execution_report.for_mstop(k) if self._execution_report is not None else None),
            nuisance_vec=(self._nuisance_vec_path[k] if k < len(self._nuisance_vec_path)
                          else self._nuisance_vec),
            nuisance_vec_path=self._nuisance_vec_path[: k + 1],
        )
        # A sliced model predicts through the same derived columns.
        _sliced._transform_recipe = getattr(self, "_transform_recipe", None)
        _sliced._response_transform = getattr(self, "_response_transform", None)
        # Truncation keeps the rows, weights and response, so the sample
        # identity carries over unchanged.
        _sliced._comparison_identity = getattr(self, "_comparison_identity", None)
        _sliced._na_dropped = getattr(self, "_na_dropped", None)
        # ... and refits from the same inputs. Without this, `confint`,
        # `cvrisk` and the conformal refits refuse on a sliced model -- the
        # ordinary way to reach the tuned fit -- while working on the model it
        # was sliced from. `continue_to` carries the bundle for the same reason.
        _sliced._refit_bundle = getattr(self, "_refit_bundle", None)
        # The baseline depends on the linear predictor at `mstop`, so a
        # truncated model needs its own. Without training rows it can only
        # keep the baseline of the full model.
        if _sliced._data is not None:
            _sliced._breslow_baseline = _breslow_baseline_of(_sliced)
        elif k >= self.mstop:
            _sliced._breslow_baseline = getattr(self, "_breslow_baseline", None)
        return _sliced


    def cvrisk(self, folds=None, type="bootstrap", B=25, prob=0.5, seed=42,
               parallel=True, at_mstop=None, *, preprocessing="fixed", _inner=False):
        """Cross-validated risk along the path.

        `at_mstop=k` judges the path truncated at `k` steps: the returned
        curve, fold rows and grid end there, and the boundary warning is
        skipped because the step count is the caller's choice, not a
        search's output -- fixing the step answers the question the
        warning asks. Pass it when comparing designs or specifications at
        a prespecified step count; leave it unset when tuning.

        ``preprocessing="fixed"`` reuses the full-data transformations and
        design, matching mboost. ``"refit"`` learns predictor transformations
        on each fold's positive-weight training rows and predicts its held-out
        rows through that recipe. Refit folds run sequentially; fitting,
        prediction and family scoring use Rust. The result records the mode.

        Parameters
        ----------
        folds : FoldScheme or array_like of shape (n, B), optional
            A :class:`FoldScheme` or an ``(n, B)`` matrix of in-bag weights,
            used as given; a scheme also carries its judge mask. ``None``
            generates ``B`` folds of resampling type ``type``.
        type : {"bootstrap", "kfold", "subsampling"}, default "bootstrap"
            Resampling used when ``folds`` is ``None``.
        B : int, default 25
            Number of generated folds when ``folds`` is ``None``.
        prob : float, default 0.5
            Share of observations drawn per fold for ``"subsampling"``.
        seed : int, default 42
            Random seed for generated folds.
        parallel : bool, default True
            Run folds in parallel in the Rust engine.
        at_mstop : int, optional
            Judge the path truncated at this many steps; the curve and grid end
            there and the boundary warning is skipped.
        preprocessing : {"fixed", "refit"}, default "fixed"
            Reuse the full-data recipe, or learn it within each training fold.

        Returns
        -------
        CvRiskResult
            Out-of-fold risk matrix of shape ``(n_folds, n_grid)``, the
            aggregate curve over the grid, and ``optimal_mstop``.

        Raises
        ------
        ValueError
            If the fold matrix does not match the number of observations, or
            the model keeps no data to refit from.
        RuntimeError
            If the Rust engine is not available.
        """
        from mboostrs._cv_preprocessing import check_preprocessing, refitted_cv_path

        preprocessing = check_preprocessing(preprocessing)
        if not _HAS_RUST_ENGINE:
            raise RuntimeError("Rust engine not available")
        # The label describes the folds actually used. `type` and `B` keep
        # their defaults when `folds` is supplied, so building the label from
        # them reported "25-fold bootstrap" for every explicit fold matrix.
        if folds is None:
            n = len(self._fitted)
            w = self._weights if self._weights is not None else np.ones(n)
            folds_matrix = _cv_weights(n, type, B, prob, seed, np.asarray(w, dtype=np.float64))
            fold_label = f"{B}-fold {type}"
        else:
            folds_matrix = _as_fold_matrix(folds, len(self._fitted), self._weight_policy,
                                           self._weights)
            fold_label = fold_label_for(folds, folds_matrix)
        judge_mask = ((folds.judge if folds.judge is not None else folds.held_out)
                      if isinstance(folds, FoldScheme) else None)
        if self._data is None or self._predictor_names is None:
            from mboostrs._refit import refit_available

            _, reason = refit_available(self)
            raise ValueError(f"cvrisk is unavailable: {reason}")
        y = _extract_response(self._data, self._response_name, self._family)
        w = self._weights if self._weights is not None else np.ones(len(self._fitted))
        if isinstance(folds, FoldScheme) and folds.base_weights is not None:
            w = np.asarray(folds.base_weights, dtype=np.float64)
        x_cols = []
        for name in self._predictor_names:
            x_cols.append(_extract_column(self._data, name))
        x_matrix = _stack_columns(x_cols, len(self._fitted))
        family_name = _family_rust_name(self._family)
        family_params = _family_params(self._family)
        mstop_val, nu, risk_str, trace_bool = (
            self._control.mstop, self._control.nu, self._control.risk, self._control.trace)
        if preprocessing == "refit":
            risk_rows, aggregate, grid, opt = refitted_cv_path(self, folds_matrix, judge_mask, w)
        else:
            risk_rows, aggregate, grid, opt = call_fit(_cvrisk_glmboost,
                x_matrix,
                y,
                w,
                mstop_val,
                nu,
                family_name,
                family_params,
                list(self._predictor_names),
                folds_matrix,
                judge=judge_mask,
                # A per-row offset (log exposure) enters every fold's linear
                # predictor; a fitted scalar offset is re-estimated per fold.
                offset=self._offset if _is_vector_offset(self._offset) else None,
                risk=risk_str,
                trace=trace_bool,
                parallel=parallel,
                _fit_count=folds_matrix.shape[1],
                intercept=getattr(self, "_intercept", True),
                center=bool(np.any(np.asarray(self._centering_means or [0.0]) != 0.0)),
                **step_kwargs(self._control),
            )
        risk_matrix = np.asarray(risk_rows, dtype=np.float64)
        aggregate = np.asarray(aggregate, dtype=np.float64)
        result = CvRiskResult(
            risk_matrix, aggregate, grid, opt, fold_label,
            provenance_from_model(self, "cvrisk", folds=fold_label,
                                  n_fits=int(folds_matrix.shape[1])),
            fold_fingerprint=fold_fingerprint(folds_matrix, judge=judge_mask, weights=w,
                                              response=y),
            data_fingerprint=data_fingerprint(y, w),
            comparison_identity=getattr(self, "_comparison_identity", None),
            held_out_fraction=held_out_fraction(folds_matrix, judge_mask),
            folds=folds if isinstance(folds, FoldScheme) else None,
            design=_fold_design(folds, folds_matrix),
            preprocessing=preprocessing,
        )
        if at_mstop is None:
            if not _inner:
                warn_if_at_boundary(result)
                warn_if_one_fold_dominates(result)
        else:
            from mboostrs.cvrisk import truncated_at_mstop

            result = truncated_at_mstop(result, at_mstop)
        return result

    def coef(self, which=None, off2int=False, aggregate="sum"):
        """Extract coefficients per predictor.

        which: predictor name(s) or index/indices. None = all non-zero.
        off2int: if True, add offset to intercept coefficient.
        aggregate: "sum" (default) sums per-step coefs; "none" returns raw per-step list.
        Returns dict mapping predictor name -> coefficient value.
        """
        names = self._learner_names
        if aggregate != "sum":
            # Single pass: build per-learner lists from one scan
            from collections import defaultdict

            per_learner = defaultdict(list)
            for sel, coef in zip(self._selected, self._coefficients, strict=False):
                per_learner[names[sel]].append(self._nu * coef.item())
            if which is not None:
                if isinstance(which, str):
                    which = [which]
                elif isinstance(which, int):
                    which = [names[which]]
                result = {}
                for w in which:
                    result[w] = per_learner.get(w, [])
                return result
            return {k: v for k, v in per_learner.items()}

        if which is not None:
            if isinstance(which, str):
                which = [which]
            elif isinstance(which, int):
                which = [names[which]]
            indices = []
            for w in which:
                if isinstance(w, int):
                    indices.append(w)
                elif isinstance(w, str):
                    try:
                        indices.append(names.index(w))
                    except ValueError:
                        raise ValueError(f"Unknown predictor: {w!r}") from None
                else:
                    raise TypeError(f"Invalid which element: {w!r}")
        else:
            indices = None

        coefs_flat = [c.item() for c in self._coefficients]
        cm = self._centering_means if self._centering_means else []
        if not getattr(self, "_intercept", True):
            # The core numbers the intercept 0 and folds the centering
            # correction into it. A fit without one (`y ~ x - 1`) is given an
            # unselected intercept slot, whose entry is dropped: mboost reports
            # the raw coefficients of such a fit.
            idx_vec, val_vec = _coef_glmboost(
                self._core_selected(), coefs_flat, self._nu, len(names) + 1, cm,
                False, self._offset,
                None if indices is None else [i + 1 for i in indices])
            return {names[i - 1]: v for i, v in zip(idx_vec, val_vec, strict=False) if i > 0}
        idx_vec, val_vec = _coef_glmboost(
            self._selected,
            coefs_flat,
            self._nu,
            len(names),
            cm,
            off2int,
            self._offset,
            indices,
        )
        return {names[i]: v for i, v in zip(idx_vec, val_vec, strict=False)}

    def _core_selected(self):
        """`selected` in the core's numbering, where learner 0 is the
        intercept: shifted by one for a fit without one (`y ~ x - 1`)."""
        if getattr(self, "_intercept", True):
            return self._selected
        return [int(i) + 1 for i in self._selected]

    def residuals(self, type="response", newdata=None, mstop=None):
        """Residuals of the named kind, or a refusal saying why there are none.

        Parameters
        ----------
        type : {"response", "pearson", "working", "deviance", "martingale"}
            ``"response"`` is ``y - mu`` with ``mu`` on the response's own
            scale. ``"pearson"`` is ``(y - mu) / sqrt(V(mu))``, available only
            for families that declare a variance function. ``"working"`` is the
            fitted family's negative gradient -- what the next boosting step
            regresses on -- and is available for every family. ``"deviance"``
            is the signed square root of each observation's deviance
            contribution. ``"martingale"`` is for `CoxPH`.

        If a residual kind is undefined for the family, the method raises
        `NotImplementedError` and names the available kinds. Use
        `residual_kinds()` to inspect them.

        See :meth:`residual_kinds` for what this family offers, and
        :func:`mboostrs.quantile_residuals` for a kind that is comparable
        across families because it is uniform under a correct fit.
        """
        from mboostrs._residuals import residuals as _residuals

        return _residuals(self, kind=type, newdata=newdata, mstop=mstop)

    def residual_kinds(self):
        """The residual kinds defined for this model's family."""
        from mboostrs._residuals import available_kinds

        return available_kinds(self)

    def extract(self, what, which=None):
        """Extract model components, matching R's extract.glmboost.

        what: one of "design", "coefficients", "residuals", "variable.names",
              "offset", "nuisance", "weights", "control".
              Aliases accepted: "coef", "coefs", "coefficient" -> "coefficients";
              "x", "model_matrix" -> "design"; "w" -> "weights".
        which: learner name(s) or zero-based index, including the intercept.
               None selects the used learners, plus the intercept when centered.
        "design" returns a dict of the fitted design columns, centered using
        the training means when center=True, with a column of ones for the
        intercept. "variable.names" returns their names in design order.
        """
        from mboostrs._output import _normalize_extract_what

        what = _normalize_extract_what(what)
        valid = ("design", "coefficients", "residuals", "variable.names",
                 "offset", "nuisance", "weights", "control")
        if what not in valid:
            raise ValueError(f"what must be one of {valid}, got {what!r}")

        if what == "coefficients":
            return self.coef(which=which)
        if what == "residuals":
            return self.residuals()
        if what == "offset":
            return self._offset
        if what == "nuisance":
            return self.nuisance()
        if what == "control":
            return self.control
        if what == "weights":
            return self._weights.copy() if self._weights is not None else np.ones(len(self._fitted))

        if what in ("design", "variable.names"):
            names = self._learner_names
            intercept = bool(names and names[0] == "(Intercept)")
            if which is None:
                indices = set(self._selected)
                bundle = getattr(self, "_refit_bundle", None) or {}
                if intercept and bundle.get("center", True):
                    indices.add(0)
                indices = sorted(indices)
            else:
                choices = [which] if isinstance(which, (str, int, np.integer)) else list(which)
                indices = []
                for choice in choices:
                    if isinstance(choice, str):
                        if choice not in names:
                            raise ValueError(f"unknown learner {choice!r}")
                        choice = names.index(choice)
                    if not isinstance(choice, (int, np.integer)) or not 0 <= choice < len(names):
                        raise ValueError(f"unknown learner index {choice!r}")
                    indices.append(int(choice))
            if what == "variable.names":
                return [names[i] for i in indices]
            if self._data is None or self._predictor_names is None:
                raise ValueError("extract('design') requires stored data/predictor names")
            from mboostrs._rust import glmboost_design

            x = np.column_stack([_extract_column(self._data, name)
                                 for name in self._predictor_names])
            design = glmboost_design(x, self._centering_means or [], intercept)
            return {names[i]: design[:, i].copy() for i in indices}

    def varimp(self, percent=True):
        """Variable importance based on in-bag risk reduction per predictor.

        Returns a dict with keys:
          reduction: dict of predictor_name -> risk reduction
          selfreq: dict of predictor_name -> selection frequency
        With ``percent=True`` (the default) each reduction is divided by the
        sum of their absolute values, so the reductions are shares summing
        to 1. ``percent=False`` returns the in-bag risk reduction per
        observation, which is what mboost's ``varimp()`` returns.

        When several learners share a name, their reductions and selection
        frequencies are summed under that name; the value is the predictor's
        total, not one learner's.
        """
        from mboostrs._attribution import attribute_by_name, risk_changes

        risk = self._risk
        if len(risk) < 2:
            return {"reduction": {}, "selfreq": {}}
        learner_names = self._learner_names
        # Same-named learners sum, mirroring GAMBoostModel.varimp.
        changes = risk_changes(risk, len(self._fitted))
        reduction, selfreq = attribute_by_name(
            changes, [learner_names[j] for j in self._selected], learner_names,
            percent=percent)
        return {"reduction": reduction, "selfreq": selfreq}

    def coef_path(self, which=None):
        """Compute cumulative coefficient paths over boosting iterations.

        Parameters
        ----------
        which : list of str, optional
            Predictor names. If None, includes all non-zero predictors.

        Returns
        -------
        dict
            Mapping predictor name -> 1D array of cumulative coefficients
            over iterations 0..mstop.
        """
        mstop = self.mstop
        names = self._learner_names

        if which is None:
            active_names = list(dict.fromkeys(names[s] for s in self._selected))
        elif isinstance(which, str):
            active_names = [which]
        else:
            active_names = list(which)

        name_to_col = {name: i for i, name in enumerate(active_names)}
        contrib = np.zeros((mstop + 1, len(active_names)))
        for i in range(mstop):
            j = self._selected[i]
            name = names[j]
            if name in name_to_col:
                contrib[i + 1, name_to_col[name]] = self._nu * self._coefficients[i].item()
        paths_arr = np.cumsum(contrib, axis=0)
        return {name: paths_arr[:, col] for name, col in name_to_col.items()}

    def plot_coef_path(self, which=None):
        """Plot coefficient paths.

        Parameters
        ----------
        which : list of str, optional
            Which predictors to plot.

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_coef_path

        return plot_coef_path(self, which=which)

    def plot(self, which=None):
        """Plot coefficient paths.

        Parameters
        ----------
        which : list of str, optional
            Which predictors to plot.

        Returns
        -------
        alt.Chart
        """
        return self.plot_coef_path(which=which)





    def confint(self, parm=None, level=0.95, B=1000, B_mstop=25, seed=None, n_jobs=1,
                groups=None, *, method="bootstrap"):
        """Bootstrap confidence intervals for coefficients.

        parm: predictor name(s) to compute CIs for. None = all used.
        level: confidence level (default 0.95).
        B: number of bootstrap replications.
        B_mstop: number of inner CV folds for mstop selection (0 = skip).
        seed: random seed for reproducibility.
        method: "bootstrap". "refit" intervals are currently available through
            GAMBoostModel.confint on a gamboost fit; glmboost names this limit.
        n_jobs: bootstrap worker threads (1 = sequential, -1 = all CPUs). Results
            are independent of n_jobs (each replicate has its own seeded RNG).
            Most effective with B_mstop=0; when B_mstop>0 the inner CV is already
            parallel (rayon), so n_jobs>1 mainly contends and adds little.
        groups: a column name or one label per row. When given, every
            bootstrap sample (and every inner fold of the `B_mstop` tuning)
            draws whole groups with replacement instead of single rows, and a
            row's weight in a sample is its group's draw count times its
            model weight. Use it when rows are repeated measurements of a
            unit, since resampling such rows separately treats them as
            independent and gives intervals that are too narrow.
        Returns a dict with 'lower', 'upper', 'level', 'boot_coefs', and
        'resampled' ('rows' or 'groups').
        """
        if method not in ("bootstrap", "refit"):
            raise ValueError(f'method must be "bootstrap" or "refit", got {method!r}')
        if method == "refit":
            raise NotImplementedError(
                "confint(method='refit') is available for gamboost fits; "
                "glmboost currently provides boosted-coefficient bootstrap intervals")
        warnings.warn(
            "Bootstrap intervals for boosted coefficients can under-cover: "
            "shrinkage and selection bias remain in the replicated fits. "
            "For gamboost fits, confint(method='refit') provides conditional "
            "REML-refit partial-effect intervals.",
            UserWarning, stacklevel=2)
        from mboostrs._bootstrap import AT_CAP, FAILED, UNTUNED, BootstrapTally, bootstrap_outcome
        from mboostrs._parallel import parallel_map
        from mboostrs._refit import refit_spec as _refit_spec

        def mstop_fn(cvr):
            return cvr.optimal_mstop

        which_names = parm
        if which_names is None:
            which_names = list(self.coef().keys())
        elif isinstance(which_names, str):
            which_names = [which_names]

        names = self._learner_names
        which_indices = []
        for w in which_names:
            if isinstance(w, int):
                which_indices.append(w)
            elif isinstance(w, str):
                try:
                    which_indices.append(names.index(w))
                except ValueError:
                    raise ValueError(f"Unknown predictor: {w!r}") from None

        n = len(self._fitted)
        model_weights = self.model_weights()
        # The outer draw is R's `cv(model.weights(object), B = B)`: multinomial
        # counts with probabilities proportional to the model's own weights, so
        # a row the fit gave no weight is never drawn. It used to be
        # `binomial(1, 0.5, n)`, which ignores those weights entirely -- on a
        # fit where half the rows carry weight zero, the replicates fitted all
        # of them and produced an interval that did not contain the estimate.
        group_labels = _confint_groups(self, groups, n)
        if group_labels is None:
            outer = _cv_weights(n, "bootstrap", B, prob=0.5,
                                seed=42 if seed is None else int(seed),
                                weights=model_weights)
        else:
            from mboostrs._bootstrap import group_bootstrap_counts

            weights_now = np.asarray(model_weights, dtype=np.float64)
            outer = group_bootstrap_counts(
                group_labels, weights_now > 0, B,
                42 if seed is None else int(seed),
            ) * weights_now[:, None]
        seeds = np.random.SeedSequence(seed if seed is not None else 42).spawn(B)
        coefficients = np.full((B, len(which_indices)), np.nan)
        col_names = [names[i] for i in which_indices]

        spec = _refit_spec(self)
        tally = BootstrapTally(B=B)

        def _one(b):
            # Independent RNG per replicate -> thread-safe and n_jobs-invariant.
            child = np.random.default_rng(seeds[b])
            w_boot = np.ascontiguousarray(outer[:, b], dtype=np.float64)
            try:
                # Through the refit spec, so `weight_policy`, `center`,
                # `na_action` and the rest of the fit's hyperparameters reach
                # the replicate. Rebuilding the call by hand here dropped them.
                mod_b = spec.build(weights=w_boot, control=self._control)
                if B_mstop > 0:
                    if group_labels is None:
                        folds_b = _cv_weights(n, "bootstrap", B_mstop, 0.5, int(child.integers(0, 2**31)), w_boot)
                    else:
                        # Counts only: cvrisk multiplies the folds by the
                        # replicate's own weights.
                        from mboostrs._bootstrap import group_bootstrap_counts

                        folds_b = group_bootstrap_counts(
                            group_labels, w_boot > 0, B_mstop, int(child.integers(0, 2**31)))
                    try:
                        # Inner CV runs sequentially when the outer bootstrap loop
                        # is threaded, so the two axes never oversubscribe cores.
                        cvr = mod_b.cvrisk(folds=folds_b, parallel=(n_jobs in (None, 1)),
                                           _inner=True)
                        if cvr.at_boundary():
                            tally.mark(b, AT_CAP)
                        mod_b = mod_b[: mstop_fn(cvr)]
                    except (ValueError, RuntimeError, KeyError,
                            np.linalg.LinAlgError) as exc:
                        # The replicate still contributes, at the outer mstop.
                        tally.mark(b, UNTUNED, exc)
                coefs_b = mod_b.coef(off2int=True)
                return np.array([coefs_b.get(names[idx], 0.0) for idx in which_indices])
            except (ValueError, RuntimeError, KeyError,
                    np.linalg.LinAlgError) as exc:
                # A resample can be degenerate -- every row carrying a level
                # dropped, a singular design -- which is part of the method
                # rather than a defect, so one failure must not end the run.
                tally.mark(b, FAILED, exc)
                return None

        for b, row in enumerate(parallel_map(_one, range(B), n_jobs)):
            if row is not None:
                coefficients[b, :] = row

        counts = bootstrap_outcome(tally, B=B, what="coefficient")
        counts["weights"] = spec.describe_weights()

        from mboostrs import _rust

        lower, upper = _rust.percentile_interval(coefficients, float(level), False)
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        return {
            "lower": dict(zip(col_names, lower, strict=False)),
            "upper": dict(zip(col_names, upper, strict=False)),
            "level": level,
            "boot_coefs": coefficients,
            # Provenance: how the interval was produced, so a reader does not
            # have to assume it came from all `B` replicates.
            "seed": seed,
            "B_mstop": B_mstop,
            "resampled": "rows" if group_labels is None else "groups",
            **counts,
        }

    def __getstate__(self):
        return {
            "formula": self._formula,
            "family": self._family,
            "control": self._control,
            "response_name": self._response_name,
            "learner_names": self._learner_names,
            "fitted": self._fitted,
            "risk": self._risk,
            "selected": self._selected,
            "coefficients": self._coefficients,
            "offset": self._offset,
            "nu": self._nu,
            "weights": self._weights,
            "predictor_names": self._predictor_names,
            "centering_means": self._centering_means,
            "nuisance_vec": self._nuisance_vec,
            "nuisance_vec_path": self._nuisance_vec_path,
            "execution_report": self._execution_report,
            # Needed to rebuild derived columns for `predict(newdata=)`; a
            # restored model without it cannot see new rows.
            "transform_recipe": getattr(self, "_transform_recipe", None),
            "response_transform": getattr(self, "_response_transform", None),
            "comparison_identity": getattr(self, "_comparison_identity", None),
            # The weight contract has to survive the round trip: without
            # these, a restored model reported the default policy and no
            # input vector, so `input_weights()` came back None and
            # `weight_policy` said "upstream" for a fit made with "raw".
            "input_weights": getattr(self, "_input_weights", None),
            "weight_policy": getattr(self, "_weight_policy", None),
            "weights_rescaled": getattr(self, "_weights_rescaled", None),
            "breslow_baseline": getattr(self, "_breslow_baseline", None),
            # The fit's own call after preprocessing, which `continue_to`,
            # `update(**kwargs)`, `cvrisk` and the other refits rebuild the
            # model from, and which `formula` reads the written formula from.
            # A restored model without it refused every refit.
            "refit_bundle": getattr(self, "_refit_bundle", None),
        }

    def __setstate__(self, state):
        # Restored explicitly rather than left to the mixin's defaults: a
        # missing key means an artifact written before the weight contract
        # existed, and "upstream" with no input vector is the right reading
        # of that, but only because nothing else was recorded.
        self._refit_bundle = state.get("refit_bundle")
        self._input_weights = state.get("input_weights")
        self._weight_policy = state.get("weight_policy") or "upstream"
        self._weights_rescaled = state.get("weights_rescaled")
        self._breslow_baseline = state.get("breslow_baseline")
        self._formula = state["formula"]
        self._family = state["family"]
        # An artifact decodes the control record as a plain tuple; the
        # record restores `control.nu` and the other named fields the
        # refit and partial-effect paths read.
        control = state["control"]
        self._control = None if control is None else as_control(control)
        self._response_name = state["response_name"]
        self._learner_names = state["learner_names"]
        self._fitted = state["fitted"]
        self._risk = state["risk"]
        self._selected = state["selected"]
        self._coefficients = state["coefficients"]
        self._offset = state["offset"]
        self._nu = state["nu"]
        self._data = None
        self._weights = state.get("weights")
        self._predictor_names = state.get("predictor_names")
        self._centering_means = state.get("centering_means")
        self._nuisance_vec = state.get("nuisance_vec", [])
        self._nuisance_vec_path = state.get("nuisance_vec_path", [])
        self._execution_report = state.get("execution_report")
        self._transform_recipe = state.get("transform_recipe")
        self._response_transform = state.get("response_transform")
        self._comparison_identity = state.get("comparison_identity")
        self._learner_store = None

    def aic(self, method="corrected", k=2.0):
        """Compute AIC for each boosting iteration.

        Returns an AicResult with AIC values, optimal mstop, and df trace.
        Requires that the model was fitted with inbag risk and that all
        learners have hat matrices (true for all GLMBoost linear learners).

        ``method='corrected'`` (default) is the upstream per-observation
        Gaussian stopping criterion, not a total GAIC. ``'gmdl'`` is also
        Gaussian-only. ``'classical'`` uses total likelihood risk and refuses
        Gaussian squared-error loss. ``k`` is positive and applies only to
        the classical method. Results identify the criterion, normalization
        and conditional hat-trace df; the trace excludes model-search df.
        """
        # AIC is `-2 * loglik + k * df`. When the risk is not a
        # log-likelihood the first term is a negated robust or ranking loss, so
        # the result is not an information criterion and differences between
        # two of them are not on the AIC scale. Refuse rather than return a
        # number that compares like one.
        from mboostrs._likelihood import loglik_refusal

        refusal = loglik_refusal(self._family)
        if refusal:
            raise NotImplementedError(
                refusal.replace("logLik is not available", "aic() is not available", 1)
                + " Choose mstop by cross-validated risk (cvrisk) instead, "
                "which needs no likelihood."
            )
        if isinstance(self._family, (CoxPH, Loglog, Lognormal, Weibull, Gehan, Cindex)):
            raise NotImplementedError(
                "aic() is not available for survival families: the hat-matrix "
                "degrees of freedom this implementation uses are derived for "
                "the squared-error trace and are not the effective dimension "
                "of a censored or partial likelihood. Choose mstop by "
                "cross-validated risk (cvrisk) instead."
            )
        from mboostrs._criteria import aic_contract

        contract = aic_contract(self._family, method, k, self._control[2])
        if self._learner_store is None:
            from mboostrs._refit import refit_available

            _, reason = refit_available(self)
            raise ValueError(
                "aic() needs the fitted learners' Gram factorizations to take "
                f"the hat-matrix trace, and {reason} Returning None here left "
                "the caller to guess which of the several reasons applied."
            )
        w = self._weights
        sumw = float(np.sum(w)) if w is not None else float(len(self._fitted))
        y = _extract_response(self._data, self._response_name, self._family)
        # Upstream gMDL uses sum(response^2), without centering or weights.
        # The weighted residual risk and sumw enter the criterion separately.
        y_ss = float(np.sum(y**2))
        selected_rust = list(self._selected)
        from mboostrs._criteria import weighted_hat

        weighted = weighted_hat(self, self._learner_store, selected_rust)
        result = self._learner_store.compute_aic(
            selected_rust,
            self._risk.tolist(),
            sumw,
            y_ss,
            method,
            k,
            df=None if weighted is None else weighted[0],
        )
        if result is None:
            raise NotImplementedError(
                "hatvalues() needs a hat matrix for every selected learner, "
                "and at least one of them has none: `btree` fits a tree, whose "
                "prediction is not a linear map of the response, and `bmono` "
                "solves a constrained problem whose solution is not either. "
                "The learners in this fit are "
                f"{sorted(set(self.selected))}. Without a hat matrix there is "
                "no trace to take, so there are no leverages and no "
                "hat-matrix degrees of freedom; choose mstop by "
                "cross-validated risk (cvrisk) instead."
            )
        aic_vals, mstop, df = result
        return AicResult(np.array(aic_vals), mstop, np.array(df),
                         provenance_from_model(self, "aic"), **contract)

    def hatvalues(self):
        """Per-observation leverage: the hat-matrix diagonal at the current
        mstop, as a 1-d array of length n.

        Refuses when there is no hat matrix to take a diagonal of -- a `btree`
        or `bmono` learner among the selected ones, or a model with no
        retained learner store. It used to return `None` for both, so the
        caller had a missing value and no reason for it.
        """
        if self._learner_store is None:
            from mboostrs._refit import refit_available

            _, reason = refit_available(self)
            raise ValueError(f"hatvalues() is unavailable: {reason}")
        selected_rust = list(self._selected)
        from mboostrs._criteria import weighted_hat

        weighted = weighted_hat(self, self._learner_store, selected_rust)
        diag = (self._learner_store.compute_hat_diag_py(selected_rust)
                if weighted is None else weighted[1])
        if diag is None:
            raise NotImplementedError(
                "hatvalues() needs a hat matrix for every selected learner, "
                "and at least one of them has none: `btree` fits a tree, whose "
                "prediction is not a linear map of the response, and `bmono` "
                "solves a constrained problem whose solution is not either. "
                "The learners in this fit are "
                f"{sorted(set(self.selected))}. Without a hat matrix there is "
                "no trace to take, so there are no leverages and no "
                "hat-matrix degrees of freedom; choose mstop by "
                "cross-validated risk (cvrisk) instead."
            )
        return np.array(diag)

    def logLik(self):
        """The fitted model's log-likelihood, or a refusal naming the reason.

        Available only where the fitted loss *is* a log-likelihood. `Binomial`,
        `Poisson`, `NBinomial`, `PropOdds` and the AFT families minimise the
        exact negative log-likelihood, so this is `-risk`. `Gaussian` minimises the
        residual sum of squares, which is not the Gaussian NLL -- no variance,
        no `2*pi`, and a factor of two out -- so the log-likelihood is
        reconstructed with `sigma^2 = RSS/n`, the MLE the loss implies but
        does not carry.

        For a robust or ranking loss -- `Laplace`, `Quantile`, `Expectile`,
        `Huber`, `AdaExp`, `HingeLoss`, `AUC`, `Gehan`, `Cindex` -- there is no
        probability model, so this method raises `NotImplementedError` for
        them. `GammaReg` does not refuse. Its risk omits `log(y)` on each row,
        and `logLik` adds that term back, so the number is the gamma density
        at the fitted mean and the `nuisance()` shape. `negative_risk()`
        leaves those terms out. `Multinomial` refuses, because its reported
        risk omits the reference class; see :meth:`likelihood_contract`.

        Unlike `mboost`, this method reports a log-likelihood only when the
        family defines one. The Gaussian value includes the normalizing
        terms implied by its fitted variance. Use :meth:`negative_risk` for
        the negated empirical risk convention used by `mboost`.
        """
        from mboostrs._likelihood import normalized_loglik

        return normalized_loglik(self)

    def negative_risk(self):
        """The negated empirical risk at the current mstop.

        This matches `mboost`'s `logLik` convention, and for most families is
        the quantity the fit minimised with its sign flipped. `Multinomial` is
        the exception: its reported risk omits the reference class, so
        `negative_risk()` is the mboost-comparable *reported* value while the
        fit actually descends the full categorical NLL (see
        :meth:`likelihood_contract`). It is a log-likelihood
        only for the families where the risk happens to be one; for the rest it
        is a negated robust or ranking loss, and differences between two of
        them are not likelihood ratios. Use it to compare two fits of the *same
        family* on the *same data*, and :meth:`logLik` when a log-likelihood is
        what the calculation needs.
        """
        return float(-self._risk[self.mstop])

    def likelihood_contract(self):
        """What this family's risk is: a likelihood, a loss, or neither.

        Returns the :class:`~mboostrs._likelihood.FamilyLikelihood` row, whose
        fields say whether the loss comes from a probability model, whether
        `-risk` is the complete log-likelihood, what measure that likelihood is
        taken with respect to, and which residual kinds are defined.
        """
        from mboostrs._likelihood import family_likelihood

        return family_likelihood(self._family)

    def to_dict(self, *, include_arrays=False):
        """The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry:
        the library build and environment, and the fingerprint of the
        training data. ``include_arrays`` is accepted for the uniform export
        signature; the summary holds no per-row arrays."""
        from mboostrs._output import export_model

        return export_model(self, training_data=self._data)

    def summary(self, as_dict=False, cv=None):
        """Summarise the fitted model.

        Returns a :class:`~mboostrs.summary.Summary`, which prints as
        fixed-width text and renders as a table in a notebook. Use
        ``as_dict=True`` for the machine-readable form.

        Parameters
        ----------
        as_dict : bool
            If True, return a JSON-serializable dict instead of the Summary.
        cv : CvRiskResult, optional
            The cross-validation result for this model. Supplying it adds a
            block comparing the iteration the model stopped at with the one
            cross-validation prefers. A model keeps no record of how it was
            tuned, so this is not inferred.
        """
        ms = self.mstop
        family_name = self._family.__class__.__name__
        names = self._learner_names

        selected = self._selected[:ms]
        freq = {}
        for s in selected:
            name = names[s]
            freq[name] = freq.get(name, 0) + 1

        coefs = self.coef()
        nonzero = {k: v for k, v in coefs.items() if abs(v) > 1e-12}

        info = {
            "model_class": "GLMBoostModel",
            "family": family_name,
            "prediction_scale": prediction_scale_text(self._family),
            "formula": self._formula,
            "mstop": ms,
            "nu": float(self._nu),
            "offset": float(self._offset),
            "risk_at_mstop": float(self._risk[ms]),
            "n": len(self._fitted),
            "n_omitted": len(getattr(self, "_na_dropped", []) or []),
            "selection_frequency": {
                name: {"count": count, "pct": 100.0 * count / ms}
                for name, count in sorted(freq.items(), key=lambda x: -x[1])
            },
            "nonzero_coefficients": {
                name: float(val) for name, val in sorted(nonzero.items(), key=lambda x: -abs(x[1]))
            },
            # `n` alone does not say what a row counted for. A fit made with
            # fractional weights under the upstream policy ran on a rescaled
            # vector, and a summary that reported only the row count described
            # a sample nobody fitted.
            "weights": self._weight_summary(),
        }
        if as_dict:
            return info
        from mboostrs.summary import build_summary  # noqa: PLC0415

        return build_summary(self, cv=cv)

    def update(self, mstop=None, **kwargs):
        """Update the model by changing mstop or other parameters.

        With only ``mstop``, returns the model sliced to it, or grown to it
        through :meth:`continue_to` when it is larger than the current one.
        With keyword arguments, fits the model again with those arguments
        replaced and every other argument as the original fit had it.

        Parameters
        ----------
        mstop : int, optional
            New stopping iteration. Without keyword arguments it slices or
            grows the existing model; with them it is the refit's ``mstop``,
            which otherwise stays at the current one.
        kwargs : dict
            Arguments to change in the refit: the control fields ``nu``,
            ``risk``, ``trace``, ``step`` and ``step_iterations``, a whole
            ``control=`` record, or the fit arguments ``family``, ``weights``,
            ``offset``, ``center``, ``subsample``, ``subsample_seed`` and
            ``weight_policy``. Any other name raises ``TypeError``.

        Returns
        -------
        GLMBoostModel
            Updated model.
        """
        if kwargs:
            from mboostrs._refit import refit_with_updates

            return refit_with_updates(self, mstop, kwargs)
        if mstop is not None:
            if mstop > self.mstop:
                return self.continue_to(mstop)
            return self[:mstop]
        return self

    def continue_to(self, mstop: int) -> GLMBoostModel:
        """Return the model grown to a larger ``mstop``.

        Equivalent to a cold ``glmboost`` with ``mstop=N`` to machine precision,
        so ``fit.continue_to(N)[:fit.mstop]`` reproduces ``fit``. See
        :meth:`GAMBoostModel.continue_to` for the full contract. A linear fit
        has no resume path, so this refits from iteration zero: it saves the
        re-specification, not the computation.
        """
        if mstop <= self.mstop:
            if mstop < 1:
                raise ValueError(f"mstop must be >= 1, got {mstop}")
            return self[:mstop]

        from mboostrs._refit import refit_spec

        # Through the refit spec, so `center`, `subsample`, `na_action` and the
        # rest of the fit's hyperparameters reach the grown model rather than
        # only the five the hand-written call listed.
        spec = refit_spec(self)
        control = as_control(self._control)._replace(mstop=mstop, nu=self._nu)
        return spec.build(control=control)

    def __repr__(self) -> str:
        family_name = self._family.__class__.__name__
        return f"GLMBoostModel(formula={self._formula!r}, family={family_name}, mstop={self.mstop})"


def _breslow_survfit(time, event, weights, linpred_train, linpred_new=None):
    """Breslow baseline-survival estimator for a fitted CoxPH boosting model.

    Matches R ``mboost:::survFit.mboost``: build the Breslow cumulative
    baseline hazard from the training linear predictor centered at its own
    mean, then either return the resulting baseline survival curve (one
    column) or, if ``linpred_new`` is given, the per-subject survival curves
    ``exp(-H0(t) * exp(linpred_new))`` (one column per new subject).

    Returns ``(surv, event_times, n_event)`` where ``surv`` has shape
    ``(n_event_times, n_curves)``.
    """
    if weights is not None and not np.allclose(weights, 1.0):
        raise NotImplementedError("survfit() cannot (yet) deal with weights")
    from mboostrs._rust import breslow_survfit

    time = np.ascontiguousarray(time, dtype=np.float64)
    event = np.ascontiguousarray(event, dtype=np.float64)
    linpred_train = np.ascontiguousarray(linpred_train, dtype=np.float64)
    if linpred_new is not None:
        linpred_new = np.ascontiguousarray(linpred_new, dtype=np.float64)
    surv, event_times, n_event = breslow_survfit(time, event, linpred_train, linpred_new)
    return np.asarray(surv), np.asarray(event_times), np.asarray(n_event)


def _breslow_baseline_of(model):
    """Breslow baseline of a fitted CoxPH model, as the arrays it is stored with.

    The baseline is computed from the model's training rows and its training
    linear predictor at the fitted ``mstop``, then kept on the model, so
    ``survfit`` and ``survival_at`` work without the training data (after
    ``save_model`` / ``load_model``). Returns None for families other than
    CoxPH, for a response that is not ``Surv(time, status)``, and for weighted
    fits, which the Breslow estimator here does not support.
    """
    from mboostrs._rust import breslow_baseline
    from mboostrs.families import CoxPH

    if not isinstance(model._family, CoxPH) or not isinstance(model._response_name, tuple):
        return None
    if model._data is None:
        return None
    if model._weights is not None and not np.allclose(model._weights, 1.0):
        return None
    time = _extract_column(model._data, model._response_name[0])
    event = _extract_column(model._data, model._response_name[1])
    linpred = np.asarray(model.predict(type="link"), dtype=np.float64)
    event_times, n_event, cumhaz, center = breslow_baseline(
        np.ascontiguousarray(time, dtype=np.float64),
        np.ascontiguousarray(event, dtype=np.float64),
        np.ascontiguousarray(linpred, dtype=np.float64),
    )
    return {
        "event_times": np.asarray(event_times, dtype=np.float64),
        "n_event": np.asarray(n_event, dtype=np.float64),
        "cumhaz": np.asarray(cumhaz, dtype=np.float64),
        "center": float(center),
    }



def _scalar_offset(offset):
    """Validate `offset` and return it as a float, or None.

    The engine applies one starting value to every observation, so a
    per-observation offset is not representable. Passing an array reached the
    Rust boundary and failed there with "only 0-dimensional arrays can be
    converted to Python scalars", which names neither the argument nor the
    limitation.
    """
    if offset is None:
        return None
    if np.ndim(offset) == 0:
        return float(offset)
    values = np.asarray(offset, dtype=np.float64).ravel()
    if values.size and np.allclose(values, values[0]):
        # A constant vector is what a caller writing np.full(n, c) means.
        return float(values[0])
    raise ValueError(
        "offset must be a single number applied to every observation; got an "
        f"array of {values.size} differing values. Per-observation offsets are "
        "not supported by the boosting engine. To model a known "
        "per-observation term, subtract it from the response before fitting, "
        "or include it as a predictor with a fixed coefficient."
    )


def _prepare_offset(offset, n, family):
    """Validate `offset` for a fit and return it as float, ndarray, or None.

    A per-observation offset is R's `offset=` semantics: one known
    contribution to the linear predictor per row (`log(exposure)` for a rate
    model), which boosting never re-estimates. A constant vector collapses to
    the scalar it means. The vector must describe exactly the rows the fit
    sees, with every entry finite: `log(0)` of a zero-exposure row is `-inf`,
    and a row with no exposure carries no information about the rate, so it
    is refused with the two remedies (drop the row or weight it zero) rather
    than entering the first risk evaluation.

    Distributional response shapes are refused for this tranche: a Multinomial
    response has one linear predictor per class and a survival response one
    per time and event column, so "one offset per observation" names no unique
    vector for them.
    """
    if offset is None:
        return None
    if np.ndim(offset) == 0:
        return float(offset)
    values = np.asarray(offset, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"offset must be a scalar or a 1-D vector, got {values.ndim} dimensions")
    if values.size and np.allclose(values, values[0]):
        return float(values[0])
    from mboostrs.families import Multinomial

    if isinstance(family, Multinomial):
        raise ValueError(
            "a per-observation offset is not defined for Multinomial: the "
            "response carries one linear predictor per class, so one offset "
            "per observation names no unique vector. Pass a scalar offset "
            "or recode the family.")
    if isinstance(family, CoxPH | Weibull | Loglog | Lognormal | Gehan | Cindex):
        raise ValueError(
            f"a per-observation offset is not supported for the survival "
            f"family {type(family).__name__} in this version")
    if len(values) != n:
        raise ValueError(
            f"offset has {len(values)} entries but the fit sees {n} rows; "
            "supply one known contribution per row")
    bad = ~np.isfinite(values)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"offset must be finite in every entry; entry {row} is not. "
            "log(0) of a zero-exposure row is -inf: such a row carries no "
            "information about the rate, so drop it or give it weight zero "
            "instead of offsetting it.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _is_vector_offset(offset) -> bool:
    """Whether a stored or supplied offset is per-observation."""
    return isinstance(offset, np.ndarray)


def _resolve_prediction_offset(stored, supplied, family, n_rows, *,
                               requires_explicit):
    """The offset a prediction starts from, validated against its rows.

    A scalar-offset model defaults to its fitted scalar. A model fitted
    with a per-observation offset reuses that vector for the training
    rows, and new rows must say their own: the fitted vector describes
    rows the model has already seen, so applying it to new rows would
    silently attach the wrong known contributions.
    """
    if (_is_vector_offset(stored) and requires_explicit and supplied is None):
        raise ValueError(
            "this model was fitted with a per-observation offset; predicting "
            f"{n_rows} new rows needs offset=, one known contribution per "
            "row (for a rate model, log(exposure) of the new rows)")
    if supplied is None:
        return stored
    return _prepare_offset(supplied, n_rows, family)


def _drop_aligned(weights, offset, dropped_indices):
    """Drop the same rows from per-observation weights/offset as from the data.

    Scalars and ``None`` pass through unchanged; only array-valued
    weights/offset are filtered so they stay aligned with the omitted frame.
    """
    if not dropped_indices:
        return weights, offset

    def _filter(value):
        if value is None or np.ndim(value) == 0:
            return value
        arr = np.asarray(value)
        mask = np.ones(arr.shape[0], dtype=bool)
        mask[dropped_indices] = False
        return arr[mask]

    return _filter(weights), _filter(offset)


def _replay_transforms(model, newdata, unseen_levels="error"):
    """Rebuild any inline-transform columns on prediction data.

    Uses the parameters recorded at fit time, so `scale(x)` keeps the training
    centre and scale instead of recomputing them from `newdata`. Under
    ``unseen_levels="average"`` or ``"zero_effect"`` a label the fit never saw is let through in
    a factor column that only ``brandom`` or ``btree`` terms read: a zero
    coefficient for ``brandom``, the right branch of each split for
    ``btree``; any other factor column still refuses it. A label that
    differs from a fitted one only in dtype (fitted as strings, arriving as
    integers) is refused under every policy, because it is not a new level
    and the policy would silently average or zero it (#334).
    """
    recipe = getattr(model, "_transform_recipe", None)
    if newdata is None or not recipe:
        return newdata
    from mboostrs.transforms import apply_recipe, refuse_unfilled_missing

    refuse_unfilled_missing(model, recipe, newdata)
    columns = frozenset()
    if unseen_levels in ("zero_effect", "average"):
        columns = _brandom_only_columns(getattr(model, "_learners_spec", None) or ())
    return apply_recipe(recipe, newdata, unseen_level_columns=columns)


def _brandom_only_columns(learners):
    """Columns read only by ``brandom`` and ``btree`` terms (and by no other
    term or ``by=``): an unseen level has a defined prediction there, a zero
    coefficient for ``brandom`` and the right branch of every split on that
    column for ``btree``."""
    from mboostrs.learners import Brandom, Btree

    random_columns, other_columns = set(), set()
    for spec in learners:
        target = random_columns if isinstance(spec, (Brandom, Btree)) else other_columns
        for attribute in ("name", "name1", "name2"):
            value = getattr(spec, attribute, None)
            if isinstance(value, str):
                target.add(value)
        target.update(getattr(spec, "names", None) or ())
        if getattr(spec, "by", None) is not None:
            other_columns.add(spec.by)
    return frozenset(random_columns - other_columns)


def _part_of(arrays, position, part):
    """``arrays[position]``, or its ``part`` (a slice) when one is given."""
    value = arrays[position]
    return value if part is None else np.asarray(value)[part]


def _merge_part(current, part, value):
    """A float64 array holding ``value`` in ``part`` of ``current``, or
    ``value`` itself when ``part`` is None. ``current`` is not modified."""
    if part is None:
        return np.ascontiguousarray(value, dtype=np.float64)
    merged = np.array(current, dtype=np.float64, copy=True)
    merged[part] = value
    return merged


def _tensor_margin_rows(model, source, interactions=None):
    """Range rows for the B-spline marginals of ``%X%`` terms against ``source``.

    A ``bbs(x) %X% bbs(z)`` term is a row-wise product of two B-spline bases,
    each built over the training range of its column. Each marginal whose
    column is in ``source`` gives one row, named by the term as written.
    A marginal fitted on ``log`` or ``sqrt`` of its column is measured on that
    transformed scale, which is the scale its boundary knots are on.

    Returns ``(rows, values, transforms)``: ``values`` is keyed by row
    position, and ``transforms`` holds each row's margin transform (``None``
    for an untransformed column), so a clamp can be taken back to raw values.
    """
    from mboostrs._extrapolation import row_for
    from mboostrs.interactions import _transformed

    rows, values, transforms = [], {}, []
    if source is None:
        return rows, values, transforms
    present = set(column_names(source))
    for interaction in (getattr(model, "_interactions", None) or ()
                        if interactions is None else interactions):
        for marginal in (interaction.left, interaction.right):
            if marginal.kind != "bbs" or marginal.column not in present:
                continue
            low, high = (float(v) for v in marginal.params["boundary_knots"])
            transform = marginal.params.get("transform")
            index = len(rows)
            # A margin fitted with na_action="impute" reads its training
            # median for a missing value, as the design does.
            from mboostrs.transforms import _column

            raw = _column(source, marginal.column)
            fill = marginal.params.get("impute_value")
            if fill is not None:
                raw = np.where(np.isfinite(raw), raw, fill)
            values[index] = np.asarray(_transformed(raw, transform), dtype=np.float64)
            transforms.append(transform)
            rows.append(row_for(interaction.name, marginal.column,
                                values[index], low, high))
    return rows, values, transforms


def _tensor_margin_clamp(model, newdata, policy, *, stacklevel=5):
    """Apply ``extrapolation=`` to the ``%X%`` B-spline marginals of ``newdata``.

    Returns ``{column: clamped values}`` for the columns the policy changes
    under ``"clip"``, or ``None``. The caller passes the mapping to
    :func:`_replay_interactions`, so only the marginal's basis sees the
    clamped values; other terms that read the same column keep the raw ones.
    """
    from mboostrs._extrapolation import POLICIES, apply_policy

    if policy not in POLICIES:
        raise ValueError(f"extrapolation must be one of {POLICIES}, got {policy!r}")
    if newdata is None or policy == "allow":
        return None
    rows, values, transforms = _tensor_margin_rows(model, newdata)
    if not rows:
        return None
    updated = apply_policy(rows, values, policy, stacklevel=stacklevel)
    if updated is values:
        return None
    clamp = {}
    for index, row in enumerate(rows):
        if np.array_equal(updated[index], values[index]):
            continue
        # The clamp is taken back to the raw column, because the replay
        # applies the margin's transform again.
        clamp[row.variable] = _raw_from_transformed(updated[index], transforms[index])
    return clamp or None


def _raw_from_transformed(values, transform):
    """Invert a margin transform: ``exp`` for ``log``, a square for ``sqrt``."""
    if not transform:
        return np.asarray(values, dtype=np.float64)
    kind, _lower = transform
    values = np.asarray(values, dtype=np.float64)
    return np.exp(values) if kind == "log" else np.square(values)


def _clamped_for(data, interaction, clamp):
    """``data`` with the clamped marginal columns of ``interaction`` replaced.

    The replacement is a new mapping; the frame the caller holds is unchanged.
    """
    if not clamp:
        return data
    overrides = {column: clamp[column]
                 for column in (interaction.left.column, interaction.right.column)
                 if column in clamp}
    return {**data, **overrides} if overrides else data


def _replay_model_fills(model, newdata):
    """Fill the model-based imputations of ``newdata`` before a tensor margin reads them.

    A tensor margin over a column fitted with ``na_action="impute_model"`` is
    rebuilt from the raw column, so the model's fill and that column's
    missing-value indicator are applied first. The rest of the recipe runs
    after the margins, as before. Nothing is applied for a model without
    model-based fills.
    """
    from mboostrs.transforms import TransformRecipe, apply_recipe

    recipe = getattr(model, "_transform_recipe", None)
    steps = tuple(recipe.steps) if recipe else ()
    targets = {s.name for s in steps if s.kind == "impute_model"}
    if not targets:
        return newdata
    pre = tuple(s for s in steps
                if s.kind == "impute_model"
                or (s.kind == "missing_indicator" and s.source in targets))
    return apply_recipe(TransformRecipe(pre), newdata)


def _replay_interactions(model, newdata, buser_newdata, stage=None,
                         unseen_levels="error", clamp=None):
    """Rebuild each %X% interaction's tensor design on prediction data.

    ``clamp`` maps a marginal column to the values its B-spline basis is
    evaluated at instead (see :func:`_tensor_margin_clamp`).

    Returns updated ``(newdata, buser_newdata)``. Each marginal design is
    rebuilt at the new rows and their row-wise tensor is supplied as the buser
    design for the interaction's derived column.

    The fit builds the two kinds at different points: a ``%X%`` tensor
    (derived column ``__ix<k>``) from the raw columns before the factor
    encoding, a factor ``by=`` (``__by<k>``) from the encoded columns after
    it. ``stage="raw"`` or ``"encoded"`` replays only that kind, so the
    caller can run the recipe between them; ``None`` replays both.

    ``unseen_levels`` is the predict policy. ``zero_effect`` codes a level
    the fit never saw to contribute nothing; ``average`` codes it as the mean
    of the level rows, so the term predicts it at the mean of its level
    effects. The two differ whenever those effects do not average to zero:
    a factor ``by=`` fits each level's difference from the reference level.
    """
    interactions = getattr(model, "_interactions", None)
    if interactions and stage is not None:
        prefix = "__ix" if stage == "raw" else "__by"
        interactions = [i for i in interactions if i.derived.startswith(prefix)]
    if newdata is None or not interactions:
        return newdata, buser_newdata
    if stage in (None, "raw"):
        newdata = _replay_model_fills(model, newdata)
    from mboostrs.interactions import build_tensor_design

    unseen = {"zero_effect": "zero", "average": "average"}.get(unseen_levels, "error")

    newdata = (
        dict(newdata)
        if isinstance(newdata, dict)
        else {c: np.asarray(newdata[c]) for c in newdata.columns}
    )
    buser_newdata = dict(buser_newdata) if buser_newdata else {}

    n_rows = None
    for interaction in interactions:
        design = build_tensor_design(
            interaction, _clamped_for(newdata, interaction, clamp), unseen=unseen)
        buser_newdata[interaction.derived] = design
        n_rows = design.shape[0]
    for interaction in interactions:
        if interaction.derived not in newdata:
            newdata[interaction.derived] = np.zeros(n_rows)
    return newdata, buser_newdata

def _missing_column_message(name: str, available: list) -> str:
    """Explain a missing column in terms of the model rather than the frame.

    Indexing a frame for a column the model needs raises the backend's own
    error — polars says only `"yearc" not found` — which does not say that a
    *model* wanted it, nor what else it will want. On the prediction path the
    caller usually has the training frame to hand and needs to know which
    columns to carry over.
    """
    close = [c for c in available if c.lower() == name.lower()]
    hint = f" Did you mean {close[0]!r}?" if close else ""
    shown = ", ".join(map(repr, available[:12]))
    if len(available) > 12:
        shown += f", ... ({len(available)} columns)"
    return (
        f"column {name!r} is required by the model but is not in the data."
        f"{hint} Available: {shown}"
    )


def _missing_column(name: str, available: list) -> MboostrsKeyError:
    """Raise a ``MboostrsKeyError`` for a column the model needs.

    A ``KeyError`` subclass, so existing ``except KeyError`` handlers keep
    working; carries ``code="missing_column"`` and ``subject=<name>`` for
    handlers that want the name without parsing the message.
    """
    return MboostrsKeyError(
        _missing_column_message(name, available),
        code="missing_column",
        subject=name,
        details=_missing_column_message(name, available),
    )


def _confint_groups(model, groups, n):
    """The group label of every training row for ``confint(groups=)``.

    ``groups`` is a column of the training data, by name, or one label per
    row. ``None`` means rows are resampled individually.
    """
    if groups is None:
        return None
    if isinstance(groups, str):
        data = getattr(model, "_data", None)
        if data is None:
            raise ValueError(
                f"groups={groups!r} names a column, but this model does not "
                f"carry its training data; pass the {n} group labels as an array")
        labels = np.asarray(_extract_column(data, groups))
    else:
        labels = np.asarray(groups)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(
            f"groups= needs one label per training row ({n}); got "
            f"{labels.shape[0] if labels.ndim else 0}")
    return labels


def _has_column(data, name: str) -> bool:
    """Whether `data` holds a column literally called `name`."""
    if hasattr(data, "columns"):
        return name in list(data.columns)
    if isinstance(data, dict):
        return name in data
    return False


def _frame_length(data, name: str) -> int:
    """Row count a pseudo-column must have: the length of the frame itself.

    A frame is a pandas/polars object with ``columns``, a mapping of column
    name to array, or a Series-like object. A mapping reports the number of
    its keys as ``len``, so its rows are read off the first value instead.
    """
    if hasattr(data, "columns"):
        return int(len(data))
    if isinstance(data, dict):
        if not data:
            raise MboostrsKeyError(
                f"column {name!r} is required by the model but the data is empty",
                code="missing_column", subject=name, details="empty data")
        first = next(iter(data.values()))
        return int(np.asarray(first).ravel().size or len(first))
    if hasattr(data, "to_numpy") or hasattr(data, "__len__"):
        return int(len(data))
    raise TypeError(f"Cannot extract column {name!r} from {type(data)}")


def _stack_columns(columns: list, n: int) -> np.ndarray:
    """Stack predictor columns into an ``(n, p)`` design, with ``p = 0`` when
    there are none (an offset-only ``y ~ 1`` model)."""
    if not columns:
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack(columns)


_SURVIVAL_RUST_NAMES = ("coxph", "weibull", "loglog", "lognormal", "gehan", "cindex")


def _is_binomial_cbind(family, response_name) -> bool:
    """Whether the response is Binomial ``cbind(successes, failures)``."""
    return (isinstance(family, Binomial) and isinstance(response_name, tuple)
            and len(response_name) == 2)


def _binomial_cbind_response(data, response_name):
    """Success proportion and trial count of ``cbind(successes, failures)``.

    The fit uses the proportion as its response and multiplies the trial
    count into the observation weights, which makes the model identical to
    ``p ~ ...`` fitted with ``weights=trials``.
    """
    success = _extract_column(data, response_name[0])
    failure = _extract_column(data, response_name[1])
    total = success + failure
    if np.any(total == 0):
        raise ValueError("Binomial cbind response has zero trials in some observations")
    return success / total, total


def _extract_response(data, response_name, family):
    """The response vector the engine fits.

    ``Surv(time, event)`` becomes the stacked ``[times..., events...]``
    vector, Binomial ``cbind(successes, failures)`` the success proportion
    (its trial counts are in the model's weights), and a single name its
    column.
    """
    if _is_binomial_cbind(family, response_name):
        return _binomial_cbind_response(data, response_name)[0]
    if _family_rust_name(family) in _SURVIVAL_RUST_NAMES and isinstance(response_name, tuple):
        return np.concatenate([_extract_column(data, response_name[0]),
                               _extract_column(data, response_name[1])])
    return _extract_column(data, response_name)


def _extract_column(data, name: str) -> np.ndarray:
    """Extract a numeric column by name from a pandas/polars DataFrame or dict.

    Selects the named column explicitly. (Relying on a DataFrame's
    ``__dataframe__`` attribute as a proxy is fragile — without it the whole
    frame would be returned instead of the column.)

    The name ``"1"`` is the formula intercept, as in R's ``y ~ 1``, but only
    when no column of the frame is actually called that — the convention
    :func:`~mboostrs.formula.expand_formula` documents: a real column named
    ``"1"`` keeps referring to the column, and the intercept is spelled
    differently or fitted through an LSS parameter formula, which is where an
    intercept-only right-hand side is meaningful.
    """
    if name == "1" and not _has_column(data, "1"):
        return np.ones(_frame_length(data, name), dtype=np.float64)
    if hasattr(data, "columns"):
        # pandas / polars DataFrame: index the named column, then to numpy.
        if name not in list(data.columns):
            raise _missing_column(name, list(data.columns))
        col = data[name]
        arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    elif isinstance(data, dict):
        if name not in data:
            raise _missing_column(name, list(data))
        arr = np.asarray(data[name])
    elif hasattr(data, "to_numpy"):
        # A Series-like object that already represents the column.
        arr = data.to_numpy()
    else:
        raise TypeError(f"Cannot extract column {name!r} from {type(data)}")
    try:
        result = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(_non_numeric_message(name, arr)) from exc
    if not np_all_finite(result):
        raise ValueError(_nonfinite_message(f"Column '{name}'", result))
    return result


def _cbind_proportion(data, response_name) -> np.ndarray:
    """The response vector a Binomial ``cbind(success, failure)`` fit reads.

    The fit itself reads the two columns as the proportion
    ``success / (success + failure)`` and folds the trial count into the
    weights, which the model keeps. Every path that rebuilds the response
    from the retained columns -- cross-validated risk, stability
    selection and the fingerprint
    of what it fits -- must rebuild this same one-row-per-observation
    vector, not stack the two raw columns: stacking is the layout a
    survival ``Surv(time, event)`` response is stored in, and handing it
    to the engine for a cbind fit pairs ``2 * n`` responses with ``n``
    design rows.
    """
    success = _extract_column(data, response_name[0])
    failure = _extract_column(data, response_name[1])
    total = success + failure
    if np.any(total == 0):
        raise ValueError("Binomial cbind response has zero trials in some observations")
    return success / total


def _non_numeric_message(name, arr) -> str:
    """Name the column and a few of its values when they will not convert.

    The bare numpy error names one value and neither the column nor the fix:
    ``could not convert string to float: 'high'``. This names the column,
    what it holds, and the two ways out.
    """
    seen: list[str] = []
    try:
        for value in np.asarray(arr).ravel().tolist():
            if value is None or value != value:
                continue
            text = repr(value)
            if text not in seen:
                seen.append(text)
            if len(seen) == 3:
                break
    except (TypeError, ValueError):
        seen = []
    example = ", ".join(seen) if seen else "its first rows are not numbers"
    return (
        f"Column {name!r} holds values that are not numbers (for example "
        f"{example}); supply the numeric values, or -- for a categorical "
        "response -- fit gamboost with family=Multinomial(), which encodes "
        "three or more class labels itself, or code two classes as 0/1 for "
        "family=Binomial()"
    )


def _nonfinite_message(subject: str, values: np.ndarray) -> str:
    """Describe where an input went non-finite.

    Names the kind, the count, and the first offending row, so the caller can
    go look at the data instead of bisecting the frame. `subject` is the
    already-formatted description of what failed, e.g. ``Column 'x'``.
    """
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())

    parts = []
    if n_nan:
        parts.append(f"{n_nan} NaN")
    if n_inf:
        parts.append(f"{n_inf} Inf")
    counts = " and ".join(parts)

    first = int(np.flatnonzero(nan_mask | inf_mask)[0])
    remedy = "drop or impute these rows first."
    if n_nan and not n_inf and subject.startswith("Column"):
        # A missing predictor value has a built-in remedy; an infinite one,
        # or a missing weight, does not.
        remedy = ("pass na_action='impute' to the fit (median fill plus a "
                  "missing-value indicator term, replayed when predicting new "
                  "rows), na_action='impute_model' when the other covariates "
                  "predict the column (the same, with a model-based fill), or "
                  "na_action='omit' (drop those rows), or drop or impute them "
                  "yourself first.")
    return (
        f"{subject} contains {counts} value(s) out of {values.size} "
        f"(first at row {first}). Fitting and prediction require finite "
        f"values; {remedy}"
    )


def np_all_finite(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))


@capture_fit_warnings
@fit_checkpoint
def glmboost(
    formula: str,
    data,
    *,
    family=None,
    control=None,
    weights=None,
    offset=None,
    center: bool = True,
    subsample: float | None = None,
    subsample_seed: int = 0,
    na_action: str = "error",
    weight_policy: str = "upstream",
) -> GLMBoostModel:
    """Fit a component-wise boosted linear model.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Each predictor becomes one base learner holding a single coefficient. At
    every iteration the learner that best fits the current negative gradient
    is chosen and its coefficient updated by ``nu`` times the fitted value, so
    the path adds one predictor's worth of signal at a time. Predictors never
    selected keep a coefficient of exactly zero, which is what makes the fit
    sparse.

    Parameters
    ----------
    formula : str
        Patsy-style formula, ``"y ~ x1 + x2"``. ``"y ~ ."`` expands to every
        column except the response. Transformations such as ``log(x)`` and
        ``I(x**2)`` are applied before fitting and replayed on new data.
    data : frame or mapping
        Polars or pandas frame, or a mapping of column name to array.
    family : Family, optional
        The loss to minimise. Defaults to :class:`~mboostrs.families.Gaussian`
        (squared error). The family also fixes the link and the offset.
    control : boost_control or tuple, optional
        ``boost_control(mstop, nu)``. Defaults to 100 iterations at ``nu=0.1``.
        ``mstop`` is the main tuning parameter; choose it with ``cvrisk``
        rather than by eye.
    weights : array, optional
        Non-negative observation weights, one per row. Integer weights are
        read as frequencies -- weight 3 stands for three identical rows -- and
        are used at the scale given. Fractional weights are read as analytic
        weights, where only the ratios carry meaning, and are rescaled to sum
        to the number of weight-positive rows before fitting; see
        ``weight_policy``. A zero weight excludes a row while keeping its
        position, so every row-aligned object stays aligned. An all-zero
        vector is rejected: the resulting object would report a completed path
        over an empty sample.
    offset : float, optional
        A single starting value for the linear predictor, applied to every
        observation. Defaults to the family's own offset. Per-observation
        offsets are not supported; a constant array is read as its one value.
    center : bool, default True
        Subtract each predictor's mean before fitting, matching R's
        ``glmboost(center = TRUE)``. Centering does not change the solution
        the path converges to, only how fast it gets there; ``False``
        reproduces R's uncentered example.
    subsample : float, optional
        Fit on this fraction of rows, drawn once before fitting.
    subsample_seed : int
        Seed for that draw.
    na_action : {"error", "omit", "impute", "impute_model", "impute_model_split"}
        What to do with rows holding a missing value: refuse them, drop them,
        or (``"impute"``) fill each missing predictor value with the training
        median and add a ``<x>_missing`` indicator term, which lets the model
        predict rows with missing values; a missing response still drops its
        row. ``"impute_model"`` keeps the indicator and fills each numeric
        predictor from a penalized additive regression on the other predictors
        (see ``mboostrs.transforms``). ``"impute_model_split"`` also fits a smooth or linear term on a filled
        column as two terms, one on the rows where it was observed and one on the
        filled rows. It is opt-in: on the claim-cost benchmark it lowered the
        coverage of one effect interval from 0.90 to 0.60. If the data already has a column named ``<x>_missing``, the
        indicator is named ``<x>_missing_`` instead. The indicator is built
        from ``<x>`` at prediction time, so prediction data and
        ``sensitivity`` need only ``<x>``. ``"error"`` names the
        offending column and row; ``"omit"`` drops them and records how many.
    weight_policy : {"upstream", "raw"}
        How the scale of ``weights`` is settled before fitting. ``"upstream"``
        (the default) reproduces ``mboost``'s ``rescale_weights``: integer
        weights pass through and fractional weights are rescaled to sum to the
        number of positive weights. ``"raw"`` passes the vector through
        unchanged, for a caller whose weight scale is itself meaningful (survey
        weights summing to a population total, for example). The scale is not
        cosmetic -- it enters ``X'WX`` and the risk -- so ``"raw"`` gives a
        different path from R's on the same fractional weights. The chosen
        policy is on the model as ``weight_policy``.

    Returns
    -------
    GLMBoostModel
        Call ``summary()`` for the fit, ``coef()`` for the coefficients,
        ``predict()`` for fitted values, and ``cvrisk()`` to choose ``mstop``.

    See Also
    --------
    gamboost : the same algorithm with smooth and other structured learners.
    gamboostLSS : one linear predictor per distribution parameter.

    Examples
    --------
    >>> import mboostrs as mb
    >>> data = mb.datasets.bodyfat()
    >>> fit = mb.glmboost("DEXfat ~ age + waistcirc + hipcirc", data=data,
    ...                   family=mb.Gaussian(),
    ...                   control=mb.boost_control(100, 0.1))
    >>> cv = fit.cvrisk(B=10)
    >>> tuned = fit[:cv.optimal_mstop]
    """
    if not _HAS_RUST_ENGINE:
        raise RuntimeError("Rust engine not available. Run `maturin develop --release` first.")

    if family is None:
        family = Gaussian()
    from mboostrs.families import Multinomial, require_standard_family

    require_standard_family(family, "glmboost")

    if isinstance(family, Multinomial):
        # The coefficients of a multinomial linear model are one vector per
        # non-reference class, which the per-column learners here do not
        # carry. mboost's glmboost refuses Multinomial for the same reason.
        raise ValueError(
            "glmboost does not fit family=Multinomial(): a multinomial model "
            "needs one coefficient per column for each non-reference class. "
            "Fit gamboost with linear terms instead, e.g. "
            "gamboost('y ~ bols(x1) + bols(x2)', data, family=Multinomial())."
        )

    if control is None:
        control = (100, 0.1, "inbag", False)
    if len(control) == 2:
        control = (control[0], control[1], "inbag", False)

    control = as_control(control)
    mstop, nu, risk_str, trace_bool = control.mstop, control.nu, control.risk, control.trace

    _na_dropped: list[int] = []
    # "error" runs too: it names a label column with missing values.
    from mboostrs.formula import apply_na_action

    data, _na_dropped = apply_na_action(formula, data, na_action)
    if _na_dropped:
        weights, offset = _drop_aligned(weights, offset, _na_dropped)
    # Retain the effective rows before stateful transforms are fitted. A
    # prediction replay uses the fitted recipe; a validation refit learns a
    # new recipe from its proper training rows.
    _raw_formula = formula
    _raw_data = data
    # An ordered response (levels= or an ordered categorical) becomes
    # PropOdds' 1..K codes here, once, so every later read of the response
    # (cross-validation, refits, diagnostics) sees the same coding.
    from mboostrs.families import ordinal_response
    from mboostrs.formula import parse_response

    data, family = ordinal_response(data, parse_response(formula), family)

    # `y ~ x - 1` (or `+ 0`): no intercept learner, as in R's model.matrix.
    from mboostrs.formula import _intercept_only, split_intercept

    # `y ~ 1` is offset-only: the fit has no learner at all, and every row is
    # predicted by the family's offset.
    offset_only = _intercept_only(formula, data)
    formula, intercept = split_intercept(formula, data)
    if not intercept and center:
        # mboost's glmboost warns the same way and centers anyway.
        warnings.warn("model with centered covariates does not contain intercept",
                      UserWarning, stacklevel=2)
    formula = expand_formula(formula, data)
    # Imputation is planned before the transforms are extracted, so the steps
    # that fill x run before a transform such as log(x) reads it (see gamboost).
    imputed = None
    if na_action in ("impute", "impute_model", "impute_model_split"):
        # Median imputation (model-based for "impute_model") plus a
        # missing-value indicator term per predictor with a missing value;
        # replayed on new data.
        formula, imputed = impute_missing(formula, data, learner='',
                                          model_based=na_action in ("impute_model", "impute_model_split"),
                                          split_filled=na_action == "impute_model_split")
    formula, _recipe = extract_transforms(formula, data)
    if imputed is not None:
        _recipe = TransformRecipe(imputed.steps + _recipe.steps)
        # The factor encoding below reads its levels from `data`, so a
        # categorical column's missing-value level must already be in place.
        data = apply_recipe(imputed, data)
    reject_factor_by(formula, data, "glmboost")
    # A bare factor term becomes R's treatment-contrast columns (`fb`, `fc`),
    # each its own component, as `model.matrix` builds them for glmboost.
    formula, _dummies = expand_factor_dummies(formula, data)
    _recipe = TransformRecipe(
        _recipe.steps + _dummies.steps + encode_factors(formula, data).steps
    )
    _response_transform = None
    if _recipe:
        _recipe = fit_recipe(_recipe, data)
        data = apply_recipe(_recipe, data)
        # A transformed response rides in the recipe as the derived column the
        # fit reads; prediction data never carries it. Split it out so the
        # replay recipe keeps predictor steps only, and the model keeps the
        # step that inverts quantiles and carries the Jacobian. A linear
        # model reports its fit on the transformed scale and claims nothing
        # about the original one.
        from mboostrs.formula import parse_response

        _response_name = parse_response(formula)
        if isinstance(_response_name, str):
            response_steps = [s for s in _recipe.steps
                              if s.name == _response_name]
            if response_steps:
                _response_transform = response_steps[0]
                _recipe = TransformRecipe(tuple(
                    s for s in _recipe.steps if s.name != _response_name))

    # Fully-processed inputs, captured so continue_to(N) can refit to a larger
    # mstop without re-running preprocessing. See GLMBoostModel.continue_to.
    # Every argument the call took, so a refit reproduces this fit rather than
    # a default-valued approximation of it. `conformal.py` used to rebuild the
    # call from the formula, data, family and control alone, which silently
    # dropped the weights, the centering, and the subsampling.
    _refit_bundle = {
        "formula": formula,
        "data": data,
        "family": family,
        "weights": weights,
        "offset": offset,
        "center": center,
        "subsample": subsample,
        "subsample_seed": subsample_seed,
        "na_action": na_action,
        "weight_policy": weight_policy,
    }
    _refit_bundle["raw_formula"] = _raw_formula
    _refit_bundle["raw_data"] = _raw_data
    _refit_bundle["raw_arguments"] = {
        "family": family, "control": control, "weights": weights,
        "offset": offset, "center": center, "subsample": subsample,
        "subsample_seed": subsample_seed, "na_action": na_action,
        "weight_policy": weight_policy,
    }

    response_name, predictor_names = parse_formula(formula)
    if offset_only:
        # The `1` names the intercept, which an offset-only fit does not carry
        # as a learner, so no predictor is left.
        predictor_names = []

    is_survival = isinstance(family, CoxPH | Weibull | Loglog | Lognormal | Gehan | Cindex)
    is_binomial_cbind = _is_binomial_cbind(family, response_name)

    if is_survival:
        if isinstance(response_name, str):
            raise ValueError(
                f"Survival family {family.__class__.__name__} requires a Surv(time, event) response, "
                f"got single variable {response_name!r}"
            )
        time_name, event_name = response_name
        time_arr = _extract_column(data, time_name).copy()
        event_arr = _extract_column(data, event_name)
        # R's Weibull/Loglog replace time==0 with sort(unique(time))[2]/10
        if isinstance(family, Weibull | Loglog | Lognormal) and np.any(time_arr == 0):
            nonzero = np.sort(np.unique(time_arr[time_arr > 0]))
            if len(nonzero) > 0:
                time_arr[time_arr == 0] = nonzero[0] / 10.0
        y = np.concatenate([time_arr, event_arr])
        n = len(time_arr)
    elif is_binomial_cbind:
        y, total = _binomial_cbind_response(data, response_name)
        n = len(y)
        if weights is None:
            weights = total
        else:
            weights = np.asarray(weights, dtype=np.float64) * total
    else:
        y = _extract_column(data, response_name)
        n = len(y)

    offset = _prepare_offset(offset, n, family)

    if weights is not None and not np_all_finite(np.asarray(weights)):
        raise ValueError(
            _nonfinite_message("Fitting weights", np.asarray(weights, dtype=np.float64).ravel())
        )
    normalized = normalize_weights(weights, n, policy=weight_policy)

    x_cols = []
    for name in predictor_names:
        col = _extract_column(data, name)
        x_cols.append(col)
    x_matrix = _stack_columns(x_cols, n)

    w = normalized.fit

    family_name = _family_rust_name(family)
    family_params = _family_params(family)
    # An intercept learner is a column of ones the engine fits; an offset-only
    # model has no learner, so the engine gets no intercept column either.
    fit_intercept = intercept and not offset_only

    (
        offset_val,
        selected,
        coefs_flat,
        risk,
        fitted,
        nuisance_vec,
        nuisance_vec_path,
        learner_store,
        execution_report,
    ) = call_fit(_fit_glmboost,
        x_matrix,
        y,
        w,
        mstop,
        nu,
        family_name,
        family_params,
        predictor_names,
        offset,
        risk=risk_str,
        trace=trace_bool,
        component_fraction=subsample,
        subsample_seed=subsample_seed,
        center=center,
        intercept=fit_intercept,
        **step_kwargs(control),
    )
    execution_report = ExecutionReport.from_mapping(execution_report)
    if trace_bool:
        print(execution_report.format_text(), file=sys.stderr)

    # A per-observation fit reports the scalar start it never used; the model
    # keeps the caller's vector, which every prediction and refit must supply.
    if _is_vector_offset(offset):
        offset_val = offset

    coefficients = [np.array([c]) for c in coefs_flat]

    # Learner names are what the user sees: coefficient keys, `selected`, and
    # `which=`. A transformed term is reported as the expression that was
    # written, `log(x)`, not as the derived column `log__x` it became. R does
    # the same. `predictor_names` below keeps the derived names, because that
    # is what the columns are actually called.
    all_learner_names = ["(Intercept)"] * fit_intercept + _display_names(
        _recipe, list(predictor_names)
    )

    # R's glmboost centers the design by default and takes `center=FALSE` to
    # turn it off. Centering changes how quickly the path converges, not what
    # it converges to: on `cars` at mstop=2000 the centered fit reaches the
    # least-squares coefficients while the uncentered one is still short of
    # them, which is what mboost's own example demonstrates.
    if center:
        centering_means = [float(np.mean(x_matrix[:, j]))
                           for j in range(x_matrix.shape[1])]
    else:
        centering_means = [0.0] * x_matrix.shape[1]

    comparison_identity = fit_identity(
        data=data, response_name=response_name, response_transform=_response_transform,
        weights=w, weight_policy=normalized.policy, dropped=_na_dropped, family=family)
    _model = GLMBoostModel(
        formula=formula,
        family=family,
        control=control,
        response_name=response_name,
        learner_names=all_learner_names,
        fitted=np.asarray(fitted, dtype=np.float64),
        risk=np.asarray(risk, dtype=np.float64),
        selected=selected,
        coefficients=coefficients,
        offset=offset_val,
        nu=nu,
        data=data,
        weights=w,
        predictor_names=predictor_names,
        centering_means=centering_means,
        nuisance_vec=nuisance_vec,
        nuisance_vec_path=nuisance_vec_path,
        learner_store=learner_store,
        execution_report=execution_report,
    )
    _model._transform_recipe = _recipe if _recipe else None
    _model._response_transform = _response_transform
    _model._intercept = fit_intercept
    _model._refit_bundle = _refit_bundle
    _model._na_dropped = _na_dropped
    _model._comparison_identity = comparison_identity
    _model._set_weight_policy(normalized)
    _model._breslow_baseline = _breslow_baseline_of(_model)
    from mboostrs._model_api import warn_if_risk_rose

    warn_if_risk_rose(_model)
    return _model
