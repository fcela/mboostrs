# `ipc_weights`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.ipc_weights`.


*Function*

```python
ipc_weights(time, status, maxweight: float = 5.0) -> ndarray
```

Inverse probability of censoring weights (mboost `IPCweights`).

Fits the Kaplan-Meier estimate of the *censoring* distribution (event and
censoring indicators swapped), evaluates it at each subject's own
observed time, and returns `status / G(time)` clipped to
`maxweight`. Censored observations (`status == 0`) always receive
weight 0. Weighting a standard-family fit by these values approximates
the outcome for the censoring-free population, the standard IPCW
construction used with `Weibull`, `Loglog`, and other
non-survival mboost families on right-censored data.

**Parameters**

- `time` (array-like): Non-empty vector of finite, non-negative observed event times.
- `status` (array-like): 1 if the event was observed, 0 if the observation was censored.
- `maxweight` (float): Positive, finite upper clip applied to the raw inverse-probability weights.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 935-975](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L935-L975)
