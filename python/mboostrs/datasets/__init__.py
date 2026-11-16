"""Built-in datasets for mboostrs examples and case studies.

These are the same datasets used in the R mboost and gamboostLSS
vignettes and papers, repackaged as CSV.

This module also provides simulation generators (`friedman1`,
`sparse_linear`, `heteroscedastic_lss`, `kinked_threshold`,
`interaction_surface` and `spatial_regime`) for scenarios with no
real-data equivalent shipped with the package, or where a known ground
truth is more useful than a real dataset. Each generator returns its
data together with the noiseless mean as a column, so a caller can
compare a fit against the mean that generated the data.

Every loader and generator takes ``output=``, following the same convention
as the rest of the package (see :func:`mboostrs.report.path_rows_output`):

* ``"auto"`` (default) — Polars, which is a hard dependency of this package.
* ``"polars"`` / ``"pandas"`` — that library specifically. Polars is always
  available; pandas is not a dependency and raises if it is not installed.
* ``"records"`` — a plain ``dict[str, numpy.ndarray]``, for callers who want
  arrays rather than a frame.

Fitting accepts all three, so the default costs nothing.
"""

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent

_VALID_OUTPUT = ("auto", "polars", "pandas", "records")


#: Provenance for the packaged datasets, keyed by loader name. Built by
#: `tools/generate_data_cards.py`, which measures the row count, column dtypes
#: and missingness from the shipped data so those cannot drift from it.
#:
#: The fields it cannot measure -- licence, retrieval date, curation script,
#: exclusions, units, known biases -- are `None` until somebody records them.
#: `None` means "not recorded", never "none apply": a licence invented here
#: would be worse than an absent one, and a reader deciding whether they may
#: redistribute a dataset needs to see that the answer is missing.
_DATA_CARDS: dict | None = None


def data_card(name: str) -> dict:
    """What is known about a packaged dataset, and what is not.

    Returns the card for `name` -- the loader's own name, e.g. ``"bodyfat"``.
    Measured fields (`rows`, `columns` with dtype and missing count) are
    generated from the shipped file. Unrecorded fields are `None`.

    >>> from mboostrs.datasets import data_card
    >>> card = data_card("bodyfat")
    >>> card["rows"]
    71
    >>> card["license"] is None       # not recorded, not "unlicensed"
    True
    """
    cards = data_cards()
    if name not in cards:
        raise KeyError(
            f"no data card for {name!r}; the packaged datasets are "
            f"{sorted(cards)}"
        )
    return cards[name]


def data_cards() -> dict:
    """Every packaged dataset's card, keyed by loader name."""
    global _DATA_CARDS
    if _DATA_CARDS is None:
        import json

        path = Path(__file__).parent / "data_cards.json"
        _DATA_CARDS = json.loads(path.read_text(encoding="utf-8"))
    return _DATA_CARDS


def _resolve_output(output: str) -> str:
    """Map ``"auto"`` onto a concrete backend, and reject unknown values.

    ``"auto"`` is always Polars: it is declared in ``[project].dependencies``,
    so it cannot be missing. This used to fall back to pandas and then to
    records, two branches that could not be reached.
    """
    if output not in _VALID_OUTPUT:
        raise ValueError(
            f"output must be one of {_VALID_OUTPUT}, got {output!r}"
        )
    if output == "auto":
        return "polars"
    return output


def _frame(columns: dict[str, np.ndarray], output: str):
    """Build a frame of the requested kind from a dict of equal-length columns."""
    resolved = _resolve_output(output)
    if resolved == "records":
        return columns
    if resolved == "polars":
        import polars as pl

        return pl.DataFrame(columns)
    import pandas as pd

    return pd.DataFrame(columns)


# The packaged CSVs were written by R, which spells a missing value `NA`.
# Polars has no reason to guess at that convention, so an `NA` in an otherwise
# numeric column makes the whole column parse as String (wpbc's `pnodes` is the
# case that shows it). `null_values` is the parameter for exactly this.
#
# Naming the single token the files actually contain, rather than enumerating
# pandas' wider NA vocabulary, keeps the rule tight: nothing else in these
# files is a missing-value marker, no quoted "NA" string values exist that
# could be nulled by mistake, and an empty string stays an empty string.
_R_NULL_TOKEN = "NA"


def _load(name: str, output: str = "auto"):
    """Read a packaged CSV into the requested frame type.

    Every path goes through Polars, which is a hard dependency. ``"records"``
    used to be routed through pandas — so the one mode documented as needing
    nothing was the only mode that required a package this project never
    declares — and ``"pandas"`` read the file with pandas, which does not know
    the R ``NA`` convention these CSVs use.
    """
    import polars as pl

    path = _DATA_DIR / f"{name}.csv"
    resolved = _resolve_output(output)
    # infer_schema_length=None reads the whole file before fixing dtypes.
    # The default window is 100 rows, which types a column from its opening
    # rows: `munichrent`'s `yearc` looks like an integer until row 116 holds
    # 1957.5, and parsing then fails outright. These files are small enough
    # that a full pass costs nothing.
    frame = pl.read_csv(path, null_values=_R_NULL_TOKEN, infer_schema_length=None)
    if resolved == "polars":
        return frame
    if resolved == "records":
        return {name: frame[name].to_numpy() for name in frame.columns}
    return frame.to_pandas()


def _load_parquet(name: str, output: str = "auto"):
    """Read a packaged Parquet extract with the usual dataset output contract."""
    import polars as pl

    frame = pl.read_parquet(_DATA_DIR / f"{name}.parquet")
    resolved = _resolve_output(output)
    if resolved == "polars":
        return frame
    if resolved == "records":
        return {column: frame[column].to_numpy() for column in frame.columns}
    return frame.to_pandas()


def bodyfat(output: str = "auto"):
    """Body fat prediction data (71 women, 10 variables).

    From Garcia et al., distributed via R package TH.data.
    Used in the mboost tutorial (Bühlmann & Hothorn, 2007) for
    GLMBoost and GAMBoost examples.

    Columns: age, DEXfat (response), waistcirc, hipcirc, elbowbreadth,
    kneebreadth, anthro3a, anthro3b, anthro3c, anthro4.
    """
    return _load("bodyfat", output)


def wpbc(output: str = "auto"):
    """Wisconsin Prognostic Breast Cancer data (198 patients, 34 variables).

    From the UCI ML repository, distributed via R package TH.data.
    Used in the mboost illustrations vignette for Binomial and CoxPH
    boosting examples.

    Columns: time, status (1=recurrence, 0=no-recurrence), plus 32
    morphological features from digitized fine-needle aspirates.
    """
    return _load("wpbc", output)


def dutchboys(output: str = "auto"):
    """Body mass index against age for 7294 Dutch boys (2 variables).

    From the Fourth Dutch Growth Study (Fredriks et al., 2000), distributed via
    R package ``gamlss.data`` as ``dbbmi``, where it is the running example for
    the LMS centile method.

    Columns: ``age`` (years, 0.03 to 21.7) and ``bmi``.

    One row per boy. Reference centile charts are built from data shaped like
    this, and the shape of the BMI distribution changes with age in all three
    of location, spread and skewness, which is what makes it the standard
    illustration.
    """
    return _load("dutchboys", output)


def india(output: str = "auto"):
    """Child undernutrition in India (4000 observations, 7 variables).

    Distributed via R package gamboostLSS. Used in the gamboostLSS
    tutorial for GaussianLSS and spatial (bmrf) modeling. Each row
    is a mother-child pair; the response is child stunting z-score.

    Columns: stunting (response, z-score), cbmi (child BMI),
    cage (child age, months), mbmi (mother BMI), mage (mother age),
    mcdist (district code, 1-based for brandom), mcdist_lab (district name).
    """
    return _load("india", output)


def iris(output: str = "auto"):
    """Anderson's Iris data (150 observations, 5 variables).

    The classic three-class classification dataset from R's built-in
    `iris`. Used here for Multinomial boosting examples.

    Columns: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Species.
    """
    return _load("iris", output)


def motorcycle(output: str = "auto"):
    """Head acceleration in a simulated motorcycle crash (133 observations).

    From Silverman (1985), "Some aspects of the spline smoothing approach to
    non-parametric regression curve fitting", J. R. Statist. Soc. B 47(1),
    1-52, and distributed with R's ``MASS`` package as ``mcycle``. It is the
    standard benchmark for nonparametric curve fitting.

    Columns: ``times`` (milliseconds after impact) and ``accel`` (head
    acceleration in g).

    The response is sharply heteroscedastic, which is the feature that makes
    it interesting beyond curve fitting: the acceleration barely moves before
    impact and scatters widely afterwards. Its standard deviation runs from
    about 1.5 g in the first 14 ms to about 53 g between 20 and 30 ms. Several
    observations share a ``times`` value with different ``accel`` values, so
    the record is not a function of time.
    """
    return _load("mcycle", output)


def munich_rent(output: str = "auto"):
    """Munich rent survey, 1999 (3082 apartments, 9 variables).

    The city of Munich's official rent survey (*Mietspiegel*), distributed via
    R package ``gamlss.data`` as ``rent99``. Landlords and tenants use the
    published survey to establish what a legally defensible rent is for a
    given apartment, so the model output is the deliverable.

    Columns: ``rent`` (monthly net rent in euros), ``rentsqm`` (rent per square
    metre in euros), ``area`` (floor space in m²), ``yearc`` (year of construction),
    ``location`` (1 average, 2 good, 3 top), ``bath`` / ``kitchen`` /
    ``cheating`` (0/1 indicators for above-standard fittings and central
    heating), and ``district`` (Munich postal district code).

    ``district`` is the reason this dataset is interesting: it has 336 levels
    with a known spatial arrangement, so the same predictor can be modelled
    as an unstructured factor, a neighbourhood graph, or a smooth surface.
    See :func:`munich_rent_districts` and :func:`munich_rent_adjacency` for
    the geography.
    """
    return _load("munichrent", output)


def munich_rent_districts(output: str = "auto"):
    """Centroid coordinates for the 336 districts in :func:`munich_rent`.

    Columns: ``district`` (matching the column of the same name in
    ``munich_rent``), and ``x`` / ``y``, the mean of the district's boundary
    vertices in the Gauss-Krüger projection metres that the original polygons
    use. These are the coordinates for a `bspatial` surface.

    A centroid is a lossy summary of a polygon — it discards size and shape,
    and for a non-convex district it need not lie inside the district. It is
    adequate for a smooth surface over the city and not adequate for
    questions about boundaries.
    """
    return _load("munichrent_districts", output)


def munich_rent_adjacency(output: str = "auto"):
    """Which districts in :func:`munich_rent` border which.

    One row per adjacent pair, with columns ``a`` and ``b`` holding district
    codes. 685 pairs over 336 districts; each unordered pair appears once.

    Two districts count as adjacent when their boundary polygons share at
    least two vertices, so a shared edge qualifies and a single touching
    corner does not. One district (a detached exclave) has no neighbours
    under this rule.

    To use this with `bmrf`, convert it to a graph Laplacian — the diagonal
    is the neighbour count, off-diagonal entries are -1 for adjacent pairs::

        codes = sorted(set(rent["district"]))
        index = {d: i for i, d in enumerate(codes)}
        K = np.zeros((len(codes), len(codes)))
        for a, b in zip(adj["a"], adj["b"]):
            i, j = index[a], index[b]
            K[i, j] = K[j, i] = -1.0
            K[i, i] += 1.0
            K[j, j] += 1.0

    Passing the adjacency matrix itself is rejected: it has the same sparsity
    pattern but is indefinite, so it would reward the differences between
    neighbouring districts that the penalty exists to shrink.
    """
    return _load("munichrent_adjacency", output)


def munich_rent_polygons(output: str = "auto"):
    """District boundary outlines for :func:`munich_rent`, for drawing maps.

    Columns: ``district``, ``order`` (vertex sequence within a district), and
    ``x`` / ``y`` in the same projected metres as
    :func:`munich_rent_districts`. Draw one filled polygon per district with
    the vertices in ``order``.

    The outlines are thinned to at most 41 vertices per district, which draws
    a clean city map at figure size for about a sixth of the original vertex
    count. They are not suitable for area or boundary-length calculations, and
    the adjacency in :func:`munich_rent_adjacency` was computed from the
    full-resolution boundaries rather than from these.

    To draw a map, convert the rings to GeoJSON and use Altair's
    ``mark_geoshape`` with ``project(type="identity", reflectY=True)`` — the
    coordinates are projected metres, not longitude and latitude, so a
    geographic projection would distort them. Encoding the vertices directly
    with ``mark_area`` and ``detail="district"`` also draws a map, but emits
    geometry per row and produces output two orders of magnitude larger.
    """
    return _load("munichrent_polygons", output)


def usmacro(output: str = "auto"):
    """US quarterly macro aggregates, 1959Q1-2009Q3 (203 observations).

    The extract distributed with ``statsmodels`` as
    ``statsmodels.datasets.macrodata``, itself assembled from FRED in 2009.
    It is packaged here as a frozen vintage: the same FRED series pulled today
    carry three later benchmark revisions, so a rerun against current FRED
    would not reproduce published estimates made on this extract.

    Columns: ``year``, ``quarter``, ``realgdp`` (chained 2005 dollars, SAAR),
    ``realcons``, ``realdpi``, ``cpi`` (index, 1982-84 = 100), ``m1``
    (nominal $bn), ``tbilrate`` (3-month Treasury bill, % p.a.), ``unemp``
    (%), ``pop`` (millions), ``infl`` (CPI inflation, % p.a.), ``realint``
    (``tbilrate`` minus ``infl``).

    M1 measures balances held for transactions. Retail sweep programs from
    January 1994 moved checkable deposits into money-market accounts
    overnight, which drained measured M1 without changing what depositors
    held, so the series before 1994 and after it are not the same
    measurement. Deposit studies on this extract stop at 1993Q4.
    """
    return _load("usmacro", output)


def deposit_market(output: str = "auto"):
    """US monthly deposit market, May 2009 to August 2026 (208 observations).

    Assembled by ``tools/curate_deposit_data.py``, retrieved 2026-08-17, from
    FRED's ``fredgraph.csv`` endpoint and the FDIC's *National Rates and Rate
    Caps* pages. Weekly and daily source series are averaged within the
    calendar month; monthly series pass through unchanged.

    Columns:

    * ``date`` — first day of the month, ISO.
    * ``fedfunds`` — effective federal funds rate, % p.a. (FRED ``FEDFUNDS``).
    * ``tbill3m`` — 3-month Treasury bill, secondary market, % p.a.
      (``TB3MS``).
    * ``natl_mmda``, ``natl_savings``, ``natl_checking``, ``natl_cd12m`` —
      FDIC national deposit rates on money-market, savings, interest-checking
      and 12-month CD accounts, % p.a. These are averages of rates *offered*
      on new accounts, not rates paid on the existing book.
    * ``mmda_source`` — ``fred_mmnrnj`` for months taken from FRED series
      ``MMNRNJ``, which the FDIC discontinued on 2021-03-29, and
      ``fdic_page`` for months scraped from the published rate-cap pages.
      The FDIC changed the calculation on 2021-04-01, from an average
      weighted by branch count and excluding credit unions to one weighted by
      domestic deposit share and including them. Both methods put the
      money-market rate at 0.06% either side of the seam, so the level does
      not jump there, but the two segments are not the same estimator.
      The other three rate columns are null before April 2021.
    * ``deposits`` — deposits at all commercial banks, $bn, seasonally
      adjusted (H.8, ``DPSACBM027SBOG``).
    * ``walcl``, ``wresbal`` — Federal Reserve total assets and reserve
      balances held at the Fed, $mn.
    * ``personal_income`` — personal income, $bn at an annual rate (``PI``).
    * ``cpi`` — CPI-U all items, seasonally adjusted index (``CPIAUCSL``).

    The last month is incomplete: the FDIC publishes its rates on the third
    Monday, before that month's macro releases exist.
    """
    return _load("deposit_market", output)


def bank_funding(output: str = "auto"):
    """Quarterly Call Report totals for all insured US banks, 1992Q1 onward.

    Every insured institution's balances summed by report date, from the FDIC
    BankFind financials API (index ``risview_20260608210616``, retrieved
    2026-08-17) by ``tools/curate_deposit_data.py``. Dollar amounts are in
    thousands, as the API reports them.

    Columns: ``repdte`` (report date, ISO), ``n_banks``, ``deposits``,
    ``deposits_domestic``, ``deposits_interest_bearing``,
    ``deposits_noninterest``, ``interest_expense_ytd``,
    ``interest_expense_qtr``, ``assets``.

    Call Report income items are cumulative within the calendar year.
    ``interest_expense_qtr`` is the within-year first difference of
    ``interest_expense_ytd``, so it is a single quarter's flow, and
    ``400 * interest_expense_qtr / deposits`` is an annualized rate.

    Interest expense here covers all interest-bearing liabilities, deposits
    and borrowings together; the public quarterly aggregate does not separate
    them. :func:`bank_funding_annual` does separate them, annually.
    """
    return _load("bank_funding", output)


def bank_funding_annual(output: str = "auto"):
    """Annual deposit interest expense for all insured US banks, 1992 onward.

    State aggregates from the FDIC BankFind summary API, summed across states,
    retrieved 2026-08-17. This endpoint is the only public aggregate that
    separates interest expense on deposits from interest expense on all
    liabilities, and it is published annually, at year end.

    Columns: ``year``, ``n_states`` (state rows summed, including territories),
    ``dep`` (total deposits), ``depi`` (interest-bearing deposits), ``depni``
    (noninterest-bearing), ``edep`` (interest expense on deposits, full year),
    ``edepdom`` (on domestic deposits), ``eintexp`` (interest expense on all
    liabilities). Dollar amounts are in thousands.

    ``100 * edep / depi`` is the average rate paid on interest-bearing
    deposits over the year. It is a backward-looking average over the whole
    book, which is a different object from the offered rates in
    :func:`deposit_market`: it includes time deposits repricing on their own
    schedule, and it is weighted by balances rather than by institution.
    """
    return _load("bank_funding_annual", output)


def fdic_bank_yearend(output: str = "auto"):
    """FDIC BankFind year-end institution financials, 2017–2025.

    A frozen 44,828-row extract of December Call Reports retrieved on
    2026-08-21 from the FDIC BankFind Suite ``financials`` endpoint. One row
    identifies a bank (``CERT``) and report date (``REPDTE``). Balances and
    income amounts, including ``ASSET``, ``DEP``, ``DEPI`` and ``EDEP``, are
    reported in thousands of US dollars. ``OFFDOM`` counts domestic offices.
    The extract is unfiltered; case studies apply their own complete-case and
    response-range rules. ``100 * EDEP / DEPI`` is a derived annual deposit
    interest expense rate in percent, not a column in this raw extract.
    See :func:`data_card` for source and missingness by column.
    """
    return _load_parquet("fdic_bank_yearend", output)


def friedman1(
    n: int = 500,
    n_features: int = 10,
    noise: float = 1.0,
    seed: int | None = None,
    output: str = "auto",
):
    """Friedman's first synthetic regression benchmark (Friedman, 1991).

    Five features enter the mean nonlinearly and additively; any features
    beyond the first five are pure noise. Used for model-selection and
    overfitting demonstrations, where the noiseless signal gives an oracle
    to compare against.

    y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + N(0, noise^2)

    Parameters
    ----------
    n : number of observations.
    n_features : total feature count (minimum 5); features beyond the
        first 5 carry no signal.
    noise : standard deviation of the additive Gaussian noise.
    seed : passed to `numpy.random.default_rng`.

    Columns: x0..x{n_features-1}, y (response), y_signal (the noiseless
    mean, for oracle comparisons).
    """
    if n_features < 5:
        raise ValueError("n_features must be >= 5")
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, n_features))
    signal = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    y = signal + rng.normal(scale=noise, size=n)
    cols = {f"x{i}": X[:, i] for i in range(n_features)}
    return _frame({**cols, "y": y, "y_signal": signal}, output)


def sparse_linear(
    n: int = 200,
    p: int = 50,
    n_signal: int = 3,
    effect_size: float = 2.0,
    noise: float = 0.5,
    seed: int | None = None,
    output: str = "auto",
):
    """Sparse linear regression: p candidate predictors, only the first
    n_signal have a nonzero coefficient. Used for variable-selection
    demonstrations (stability selection, sparse-vs-smooth screening).

    Coefficients alternate sign (+effect_size, -effect_size, ...) across
    the n_signal active predictors so a fitted model is easy to check
    against the known truth.

    Columns: x0..x{p-1}, y.
    """
    if n_signal > p:
        raise ValueError("n_signal must be <= p")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    signs = np.where(np.arange(n_signal) % 2 == 0, 1.0, -1.0)
    beta[:n_signal] = effect_size * signs
    y = X @ beta + rng.normal(scale=noise, size=n)
    cols = {f"x{i}": X[:, i] for i in range(p)}
    return _frame({**cols, "y": y}, output)


def heteroscedastic_lss(n: int = 500, seed: int | None = None, output: str = "auto"):
    """Two-covariate location-scale scenario for distributional regression.

    The mean depends on x1 and the standard deviation depends on x2 (log
    link), so a mean-only model misses the x2-driven heteroscedasticity.
    Used for GaussianLSS / gamboostLSS demonstrations.

    mu = 1 + 2*x1
    sigma = exp(0.3 + 0.5*x2)
    y = mu + sigma * N(0, 1)

    Columns: x1, x2, y, mu (the true mean), sigma (the true scale).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    mu = 1.0 + 2.0 * x1
    sigma = np.exp(0.3 + 0.5 * x2)
    y = mu + sigma * rng.normal(size=n)
    return _frame({"x1": x1, "x2": x2, "y": y, "mu": mu, "sigma": sigma}, output)


def kinked_threshold(
    n: int = 600,
    kink0: float = 0.3,
    kink1: float = 0.7,
    slope0: float = 3.0,
    slope1: float = -2.0,
    noise: float = 0.5,
    seed: int | None = None,
    output: str = "auto",
):
    """Two kinked (hinge) features, a smooth nuisance term and four noise
    features — the threshold/kink scenario `bmono` and `btree` are built for.

    The mean is piecewise linear in ``x0`` and ``x1`` with a single kink in
    each, plus one smooth nuisance curve the additive default path fits
    well, plus features that carry no signal:

    y = slope0*max(0, x0 - kink0) + slope1*max(0, kink1 - x1)
        + 2*sin(pi*x2) + N(0, noise^2)

    with ``x0``..``x6`` iid U(0, 1). ``x0`` is flat below ``kink0`` and rises
    at ``slope0`` after it; ``x1`` falls at ``slope1`` up to ``kink1`` and is
    flat beyond; ``x2`` enters smoothly; ``x3``..``x6`` are pure noise.

    Parameters
    ----------
    n : number of observations.
    kink0, kink1 : kink locations in ``x0`` (upward hinge) and ``x1``
        (downward hinge), on the feature scale.
    slope0, slope1 : the slope after the kink in ``x0`` and before it in
        ``x1``. Signs carry the effect direction of each segment.
    noise : standard deviation of the additive Gaussian noise.
    seed : passed to `numpy.random.default_rng`.
    output : see the module docstring.

    Columns: x0..x6, y (response), y_signal (the noiseless mean).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 7))
    signal = (
        slope0 * np.maximum(0.0, X[:, 0] - kink0)
        + slope1 * np.maximum(0.0, kink1 - X[:, 1])
        + 2.0 * np.sin(np.pi * X[:, 2])
    )
    y = signal + rng.normal(scale=noise, size=n)
    cols = {f"x{i}": X[:, i] for i in range(7)}
    return _frame({**cols, "y": y, "y_signal": signal}, output)


def interaction_surface(
    n: int = 600,
    strength: float = 3.0,
    noise: float = 0.7,
    seed: int | None = None,
    output: str = "auto",
):
    """An explicit product-surface interaction plus additive mains and noise
    features — the scenario pair screening (`screen_interactions`) is meant
    to recover.

    Both factors are centered at 1/2 so the product has *no* additive
    component: an additive fit leaves the whole interaction in its
    residuals, so the pair is found by a screen on the residuals, not by the
    main effects.

    y = strength*(x0 - 0.5)*(x1 - 0.5) + 1.5*sin(pi*x2) + 1.0*x3 + N(0, noise^2)

    with ``x0``..``x5`` iid U(0, 1). The surface is a saddle: y is higher
    than its additive base where ``x0`` and ``x1`` are on the same side of
    0.5 and lower where they are on opposite sides, so the direction of
    x0's effect flips with x1.

    Parameters
    ----------
    n : number of observations.
    strength : the product coefficient; the interaction's magnitude.
    noise : standard deviation of the additive Gaussian noise.
    seed : passed to `numpy.random.default_rng`.
    output : see the module docstring.

    Columns: x0..x5, y (response), y_signal (the noiseless mean), y_product
    (the interaction term alone, for oracle comparisons).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 6))
    product = strength * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
    mains = 1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3]
    signal = product + mains
    y = signal + rng.normal(scale=noise, size=n)
    cols = {f"x{i}": X[:, i] for i in range(6)}
    return _frame(
        {**cols, "y": y, "y_signal": signal, "y_product": product}, output
    )


def spatial_regime(
    n: int = 800,
    boundary_intercept: float = 0.5,
    boundary_slope: float = 0.3,
    jump: float = 2.0,
    within_slope: float = 1.5,
    noise: float = 0.5,
    seed: int | None = None,
    output: str = "auto",
):
    """A spatial regime discontinuity: the mean jumps across a slanted
    boundary line, with smooth spatial variation within each regime and two
    noise features — the scenario `bspatial` and threshold learners address.

    Points sit on a jittered 2-D grid in the unit square: ``side =
    ceil(sqrt(n))`` grid cells per axis, one uniform draw inside each cell,
    the first n cells after a seeded permutation. The regime is the upper
    side of the boundary line in (s0, s1):

    y = jump*I[s1 > boundary_intercept + boundary_slope*s0]
        + within_slope*s0 + 1.0*s1 + N(0, noise^2)

    ``n1`` and ``n2`` are pure-noise features with no spatial structure.

    Parameters
    ----------
    n : number of observations (the grid is the first n permuted cells).
    boundary_intercept, boundary_slope : the boundary line's s1-intercept at
        s0=0 and its slope in s0; the discontinuity sits where
        ``s1 = boundary_intercept + boundary_slope*s0``.
    jump : the mean difference crossing the boundary upward (the regime
        means differ by exactly this once the smooth terms are partialled
        out). Its sign is the jump's direction.
    within_slope : the linear within-regime slope in s0 (regimes share it).
    noise : standard deviation of the additive Gaussian noise.
    seed : passed to `numpy.random.default_rng`.
    output : see the module docstring.

    Columns: s0, s1, n1, n2, y (response), y_signal (the noiseless mean),
    regime (the true regime indicator, 1 above the boundary).
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    cells = np.arange(side * side)
    picked = rng.permutation(cells)[:n]
    rows, cols_ix = picked // side, picked % side
    s0 = (cols_ix + rng.uniform(0, 1, n)) / side
    s1 = (rows + rng.uniform(0, 1, n)) / side
    n1 = rng.uniform(0, 1, n)
    n2 = rng.uniform(0, 1, n)
    regime = (
        s1 > boundary_intercept + boundary_slope * s0
    ).astype(float)
    signal = (
        jump * regime + within_slope * s0 + 1.0 * s1
    )
    y = signal + rng.normal(scale=noise, size=n)
    return _frame(
        {
            "s0": s0, "s1": s1, "n1": n1, "n2": n2,
            "y": y, "y_signal": signal, "regime": regime,
        },
        output,
    )


def hip_cz(output: str = "auto"):
    """Commuting-zone life expectancy and local characteristics (US, 2001-2014).

    The Health Inequality Project merge behind Chetty, Stepner, Abraham, Lin,
    Scuderi, Turner, Bergeron & Cutler (JAMA 2016, 315:1750-1766): race-adjusted
    expected age at death at 40 for the bottom (Q1) and top (Q4) income
    quartiles, by gender, joined to 39 area-level covariates in nine groups
    (health behaviours, healthcare access and quality, income inequality,
    segregation, social cohesion, labour market, education, migration and
    composition, local government), CZ population (2000), person counts per
    gender-quartile cell, census division, and the CZ centroid longitude and
    latitude (vertex mean of the 1990 CZ polygon's largest ring). The 595 rows
    are the commuting zones above the 25,000-population reporting cutoff the
    paper's CZ-level tables use; covariates are missing for some CZs, and the
    complete-case analysis sample is 437.

    Source files: health_ineq_online_table_6.csv, health_ineq_online_table_10.csv,
    cz1990_shapefile.zip from healthinequality.org (data downloaded 2026-08).
    Published eTable 8 of the paper's supplement reproduces from these columns
    as population-weighted (pop2000) Pearson correlations.

    Columns: cz, czname, pop2000, stateabbrv, census_div, lon, lat,
    le_raceadj_q1_M, le_raceadj_q1_F, le_raceadj_q4_M, le_raceadj_q4_F and
    the four matching sd_le_raceadj_* standard errors, count_q1_M,
    count_q1_F, count_q4_M, count_q4_F, plus the 39 covariate columns named
    as in the source tables. The sd columns are the published standard
    errors of the estimates — squared, they are the known sampling
    variances GaussianKnownVarLSS consumes via a cbind(outcome, s2)
    response.
    """
    return _load("hip_cz", output)


def hip_county(output: str = "auto"):
    """County life expectancy and local characteristics (US, 2001-2014).

    The county-level counterpart of :func:`hip_cz`, built from Health
    Inequality Project Tables 11 and 12. Each row contains race-adjusted
    expected age at death at 40 for the bottom and top income quartiles by
    gender, their published standard errors and cell counts, 39 local-area
    covariates, county population and identifiers, census division, and the
    centroid of the county's commuting zone. The coordinates are therefore
    suitable only for the study's CZ-resolution spatial adjustment; use a
    county boundary file to map county results.

    The 935 rows are counties above the source's 25,000-population reporting
    threshold that are complete on all four outcomes, all 39 covariates,
    division, and position. Covariates remain on their source scales; the case
    standardizes them immediately before fitting.

    Source files: ``health_ineq_online_table_11.csv`` and
    ``health_ineq_online_table_12.csv`` from healthinequality.org, joined to
    the Table-6 county-to-CZ crosswalk and 1990 CZ geometry (downloaded
    2026-08). The merged data reproduce all 72 correlations printed in the
    JAMA supplement's eTable 9 to its two-decimal precision.
    """
    return _load("hip_county", output)


def resp_inf(output: str = "auto"):
    """Respiratory infection in Indonesian preschool children (cohort).

    The Sommer cohort: 275 children examined on up to six consecutive
    quarters for the presence of respiratory infection, with xerophthalmia
    (vitamin A deficiency) recorded at each exam, age in months centered at
    36, height-for-age centered at 90% of the NCHS standard, a stunting
    indicator, and the annual seasonal cycle as a season code plus cosine
    and sine terms. Distributed with R package ``gamlss.data`` as
    ``respInf``; analysed by Zeger and Karim (1991), Diggle et al. (2002)
    and Skrondal and Rabe-Hesketh (2004), among others. The original study
    is Sommer et al. (1983).

    Columns are renamed from the upstream frame where the upstream names
    mislead: upstream ``time`` is the binary response (presence of
    infection) and is called ``infection`` here; upstream ``time.1`` is
    the exam round (``round``, 1 to 6); upstream ``time2`` is calendar
    months since the study start (``month``, 1 to 36). Upstream ``resp``
    is a vector of ones the upstream documentation marks "not used
    further"; it is kept as shipped so the frame stays the one analysed in
    the literature. A child appears in up to 6 rows, so rows within a
    child share more than the columns record: analyses that cross-validate
    should hold children out whole (see
    :func:`mboostrs.folds.grouped_folds`).
    """
    return _load("resp_inf", output)


def polio(output: str = "auto"):
    """Monthly US poliomyelitis cases, 1970-1983 (168 observations).

    Cases reported to the US Centers for Disease Control, distributed with
    R package ``gamlss.data`` as ``polio``, a monthly time series; the
    calendar is made explicit here as ``year``, ``month`` and a
    first-of-month ``date`` column (a Polars ``Date``), which orders the
    rows the way the series is read. The series was analysed by Zeger
    (1988), Zeger and Qaqish (1988), Li (1994), Davis et al. (1999) and
    Benjamin et al. (2003); Zeger's finding is evidence of a decrease in
    the polio infection rate across the vaccination era.

    The counts are overdispersed relative to a Poisson mean and decline
    over the period. Analyses that cross-validate should judge the past on
    the future it predicts rather than split months at random (see
    :func:`mboostrs.folds.rolling_folds`).
    """
    import polars as pl

    frame = _load("polio", "polars")
    frame = frame.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    resolved = _resolve_output(output)
    if resolved == "polars":
        return frame
    if resolved == "records":
        return {name: frame[name].to_numpy() for name in frame.columns}
    return frame.to_pandas()
