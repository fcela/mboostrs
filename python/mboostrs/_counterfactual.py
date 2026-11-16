"""Counterfactual frames predicted through a few stacked `predict` calls (#296).

A profile, an ALE or a grid fallback predicts the same rows many times, each
time with one column replaced. Every `predict` call pays a fixed cost before
the core evaluates a single basis: frame conversion, transform and interaction
replay, the extrapolation check and the boundary crossing. Stacking the
counterfactual frames row-wise and predicting them in one call pays that cost
once. The prediction of a row depends only on that row, so the stacked result,
split back into its frames, equals one `predict` call per frame.
"""

from __future__ import annotations

import numpy as np

# Upper bound on the rows sent to one `predict` call. A profile over 25 values
# of a million-row frame would otherwise build a 25-million-row copy of every
# column; chunks of at most this many rows bound that copy while keeping the
# number of calls small.
MAX_STACKED_ROWS = 1 << 20


def _frame_rows(frame):
    lengths = {len(np.asarray(values)) for values in frame.values()}
    if len(lengths) != 1:
        raise ValueError("counterfactual frame columns must share one length")
    return lengths.pop()


def _stack(chunk, names):
    """One frame holding the rows of every frame in ``chunk``, in order.

    A column that is the same array object in every frame is repeated with
    ``np.tile``; the others are concatenated.
    """
    first = chunk[0]
    stacked = {}
    for name in names:
        if all(frame[name] is first[name] for frame in chunk):
            stacked[name] = np.tile(np.asarray(first[name]), len(chunk))
        else:
            stacked[name] = np.concatenate([np.asarray(frame[name]) for frame in chunk])
    return stacked


def predict_frames(predict, frames, *, max_rows=None):
    """Predictions of each frame in ``frames``, from as few calls as possible.

    ``frames`` is a list of column mappings with the same column names.
    ``predict(frame, copies)`` predicts one stacked frame, where ``copies`` is
    the number of frames stacked into it, and returns an array whose first axis
    has one entry per stacked row. Consecutive frames are concatenated until
    the next one would take a call past ``max_rows`` rows (default
    ``MAX_STACKED_ROWS``); a single frame larger than that is sent alone.

    Returns a list with one array per frame.
    """
    if not frames:
        return []
    limit = MAX_STACKED_ROWS if max_rows is None else max(1, int(max_rows))
    names = list(frames[0])
    lengths = [_frame_rows(frame) for frame in frames]
    out = []
    start = 0
    while start < len(frames):
        stop, rows = start + 1, lengths[start]
        while stop < len(frames) and rows + lengths[stop] <= limit:
            rows += lengths[stop]
            stop += 1
        chunk = frames[start:stop]
        stacked = chunk[0] if len(chunk) == 1 else _stack(chunk, names)
        predicted = np.asarray(predict(stacked, len(chunk)))
        if predicted.ndim == 0 or predicted.shape[0] != rows:
            raise ValueError(
                f"predict returned shape {predicted.shape} for {rows} stacked rows; "
                "expected one entry per row"
            )
        bounds = np.cumsum(lengths[start:stop])[:-1]
        out.extend(np.split(predicted, bounds, axis=0))
        start = stop
    return out


def replaced_frames(columns, variable, values, n):
    """One frame per value: ``columns`` with ``variable`` set to that value in
    every row. The other columns are shared, not copied."""
    return [{**columns, variable: np.full(n, value)} for value in values]


def tiled_offset(offset, n, copies):
    """The offset of ``copies`` stacked copies of ``n`` rows.

    A per-row offset is repeated once per copy; a scalar or ``None`` applies to
    every row as it is.
    """
    if offset is None:
        return None
    values = np.asarray(offset)
    if values.ndim == 1 and values.shape[0] == n and copies > 1:
        return np.tile(values, copies)
    return offset
