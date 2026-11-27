"""Rearrangement of separately fitted quantile or expectile levels.

``Quantile(tau)`` and ``Expectile(tau)`` models each fit a single level, so
the predicted ``tau = 0.9`` quantile can fall below the predicted
``tau = 0.5`` quantile at some covariate values. Chernozhukov,
Fernandez-Val and Galichon (2010, *Econometrica* 78(3)) repair this at
prediction time: sort each row's predictions across levels into increasing
order. :func:`rearrange_quantiles` predicts every model on the same rows,
reports how many rows crossed before sorting, and returns the sorted values.

References
----------
Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010). Quantile and
probability curves without crossing. *Econometrica* 78(3), 1093-1125.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mboostrs import _rust
from mboostrs._errors import MboostrsValueError
from mboostrs._output import output_frame


def _label(kind: str, tau: float) -> str:
    prefix = "q" if kind == "quantile" else "e"
    return f"{prefix}{100.0 * tau:g}".replace(".", "_")


def _frame(columns: dict[str, np.ndarray], output: str) -> Any:
    """Column dict to the requested frame type without a per-row Python loop."""
    if output == "records":
        names = list(columns)
        n = len(columns[names[0]]) if names else 0
        return [{name: float(columns[name][i]) for name in names} for i in range(n)]
    if output in ("auto", "polars"):
        try:
            import polars as pl

            return pl.DataFrame(columns)
        except ImportError:
            if output == "polars":
                raise
    if output in ("auto", "pandas"):
        try:
            import pandas as pd

            return pd.DataFrame(columns)
        except ImportError:
            if output == "pandas":
                raise
    if output == "auto":
        return _frame(columns, "records")
    raise ValueError(
        f"output must be 'auto', 'pandas', 'polars', or 'records', got {output!r}"
    )


@dataclass(frozen=True)
class RearrangedQuantiles:
    """Sorted predictions at several levels and the crossings found before sorting.

    Attributes
    ----------
    kind : {"quantile", "expectile"}
        The family kind of the fitted models.
    levels : tuple of float
        The ``tau`` of each column, increasing.
    predictions : DataFrame
        Predictions after sorting, one column per level, named ``q10``,
        ``q50``, ... for quantiles and ``e10``, ... for expectiles. Every
        row is non-decreasing from left to right.
    unsorted : DataFrame
        The predictions of each model before sorting, same columns.
    crossing_rows : int
        Number of rows where some level was predicted above the next higher
        level.
    crossing_share : float
        ``crossing_rows`` divided by the number of rows.
    pair_crossings : DataFrame
        One row per pair of adjacent levels: ``lower`` and ``upper`` (the
        two ``tau`` values), ``n_crossing`` (rows where the lower level was
        predicted strictly above the upper level) and ``share`` (that count
        over the number of rows).
    max_crossing : float
        Largest amount by which a lower level was predicted above the next
        higher level, in the units of the response; 0 when nothing crossed.
    n : int
        Number of rows.
    """

    kind: str
    levels: tuple
    predictions: Any
    unsorted: Any
    crossing_rows: int
    crossing_share: float
    pair_crossings: Any
    max_crossing: float
    n: int

    def to_dict(self, *, include_arrays=False) -> dict:
        """A plain-Python record of the crossing report.

        The scalar summary and the per-pair crossing counts are always
        included; the sorted and unsorted predictions (one list per level) only
        with ``include_arrays=True``."""
        record = {
            "kind": self.kind,
            "levels": list(self.levels),
            "n": self.n,
            "crossing_rows": self.crossing_rows,
            "crossing_share": self.crossing_share,
            "max_crossing": self.max_crossing,
            "pair_crossings": _as_rows(self.pair_crossings),
        }
        if include_arrays:
            record["predictions"] = _as_columns(self.predictions)
            record["unsorted"] = _as_columns(self.unsorted)
        return record


def _as_columns(table) -> dict:
    """A table in any supported output format as ``{column: list}``."""
    if isinstance(table, list):
        return {k: [row[k] for row in table] for k in (table[0] if table else {})}
    if hasattr(table, "to_dict") and not hasattr(table, "to_series"):
        return {str(k): list(v) for k, v in table.to_dict("list").items()}
    return {k: v for k, v in table.to_dict(as_series=False).items()}


def _as_rows(table) -> list:
    """A table in any supported output format as a list of row dicts."""
    columns = _as_columns(table)
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    return [{k: columns[k][i] for k in names} for i in range(n)]


def _level_of(model, index: int) -> tuple[str, float]:
    family = getattr(model, "_family", None)
    name = type(family).__name__
    if name == "Quantile":
        return "quantile", float(family.tau)
    if name == "Expectile":
        return "expectile", float(family.tau)
    raise MboostrsValueError(
        f"models[{index}] has family {name or 'unknown'}; rearrange_quantiles() "
        "needs Quantile or Expectile fits"
    )


def rearrange_quantiles(models, newdata=None, *, levels=None, output="auto", unseen_levels="error"):
    """Predict several quantile or expectile fits and sort each row across levels.

    Each model estimates one level ``tau`` of the conditional distribution, so
    the separately fitted curves can cross: at some rows the predicted
    ``tau = 0.9`` value lies below the predicted ``tau = 0.5`` value. This
    function predicts every model on ``newdata`` and, for each row, sorts the
    predictions into increasing order of ``tau`` (the rearrangement of
    Chernozhukov, Fernandez-Val and Galichon, 2010). Rows that did not cross
    are returned unchanged. The result also reports, from the unsorted
    predictions, the share of rows with a crossing, the number of crossings
    per pair of adjacent levels and the largest crossing.

    Use it when two or more ``Quantile`` fits, or two or more ``Expectile``
    fits, are read together as a conditional distribution or as the edges of
    a prediction band. The identity link makes the predictions of these
    families response-scale values. Rows without a crossing are unchanged.
    Chernozhukov et al. show that, when the true curve is increasing in
    ``tau``, the sorted estimate is no farther from it than the unsorted
    estimate in every ``L^p`` norm. The fitted models are not modified and
    nothing is refit.

    Parameters
    ----------
    models : sequence of fitted models, or dict
        Fitted ``glmboost``/``gamboost`` models, all with ``Quantile``
        families or all with ``Expectile`` families, at distinct ``tau``.
        A dict is read as ``{tau: model}`` and its keys must equal each
        model's family ``tau``.
    newdata : dict or DataFrame, optional
        Rows to predict, with the predictor columns every model needs.
        ``None`` predicts the training rows; all models must then share the
        same training rows.
    levels : sequence of float, optional
        If given, the ``tau`` of each model in ``models`` order, checked
        against the ``tau`` read from each model's family.
    output : {"auto", "polars", "pandas", "records"}
        Frame type of the returned tables.

    Returns
    -------
    RearrangedQuantiles
        Sorted and unsorted predictions with one column per level, ordered by
        increasing ``tau``, and the crossing report.

    Raises
    ------
    MboostrsValueError
        If fewer than two models are given; if a model is not a Quantile or
        Expectile fit; if Quantile and Expectile fits are mixed; if two
        models share a ``tau``; if ``levels`` disagrees with the models; or if
        the predictions differ in length or contain NaN.

    Examples
    --------
    >>> fits = [mb.glmboost("y ~ bols(x)", data=d, family=mb.Quantile(tau=t))
    ...         for t in (0.1, 0.5, 0.9)]                      # doctest: +SKIP
    >>> r = mb.rearrange_quantiles(fits, d)                    # doctest: +SKIP
    >>> r.crossing_share, r.predictions                        # doctest: +SKIP
    """
    if isinstance(models, dict):
        keys = list(models)
        mlist = [models[k] for k in keys]
        if levels is not None:
            raise MboostrsValueError(
                "pass either a dict {tau: model} or levels=, not both"
            )
        levels = keys
    else:
        mlist = list(models)
    if len(mlist) < 2:
        raise MboostrsValueError(
            "rearrange_quantiles() needs at least two models at different levels"
        )
    info = [_level_of(m, i) for i, m in enumerate(mlist)]
    kinds = {k for k, _ in info}
    if len(kinds) > 1:
        raise MboostrsValueError(
            "models mix Quantile and Expectile families; rearrange one kind at a time"
        )
    kind = kinds.pop()
    taus = [t for _, t in info]
    if levels is not None:
        given = [float(v) for v in levels]
        if len(given) != len(taus) or any(
            not np.isclose(a, b, rtol=0.0, atol=1e-12) for a, b in zip(given, taus, strict=True)
        ):
            raise MboostrsValueError(
                f"levels={given} do not match the tau read from the models' "
                f"families {taus}"
            )
    if len(set(taus)) != len(taus):
        raise MboostrsValueError(f"two models share a tau; taus are {taus}")

    import inspect

    order = sorted(range(len(taus)), key=lambda i: taus[i])
    cols = []
    for i in order:
        predict_fn = mlist[i].predict
        if unseen_levels != "error" and "unseen_levels" in inspect.signature(predict_fn).parameters:
            pred = np.asarray(predict_fn(newdata, type="link", unseen_levels=unseen_levels), dtype=float)
        else:
            pred = np.asarray(predict_fn(newdata, type="link"), dtype=float)
        if pred.ndim != 1:
            raise MboostrsValueError(
                f"models[{i}] returned predictions of shape {pred.shape}; "
                "expected one value per row"
            )
        cols.append(pred)
    if len({c.shape[0] for c in cols}) != 1:
        raise MboostrsValueError(
            "models predicted different numbers of rows: "
            f"{[c.shape[0] for c in cols]}"
        )
    matrix = np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)
    if np.isnan(matrix).any():
        raise MboostrsValueError(
            "predictions contain NaN; a row with a NaN cannot be sorted"
        )
    sorted_m, crossing_rows, share, pair_counts, max_crossing = _rust.rearrange_levels(
        matrix
    )
    sorted_m = np.asarray(sorted_m)
    n = matrix.shape[0]
    sorted_taus = [taus[i] for i in order]
    names = [_label(kind, t) for t in sorted_taus]
    pairs = [
        {
            "lower": sorted_taus[j],
            "upper": sorted_taus[j + 1],
            "n_crossing": int(pair_counts[j]),
            "share": (pair_counts[j] / n) if n else 0.0,
        }
        for j in range(len(sorted_taus) - 1)
    ]
    return RearrangedQuantiles(
        kind=kind,
        levels=tuple(sorted_taus),
        predictions=_frame({nm: sorted_m[:, j] for j, nm in enumerate(names)}, output),
        unsorted=_frame({nm: matrix[:, j] for j, nm in enumerate(names)}, output),
        crossing_rows=int(crossing_rows),
        crossing_share=float(share),
        pair_crossings=output_frame(pairs, output),
        max_crossing=float(max_crossing),
        n=int(n),
    )
