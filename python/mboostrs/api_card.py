"""One-glance API cards: signature, return type and a minimal example (#505).

``api_card(stabsel)`` answers the three questions an agent has before its
first call -- what arguments the function takes, what it returns, and what a
short call looks like -- in a bounded number of lines, so a name discovered
through ``__all__`` is usable without reading the whole docstring. The
``python -m mboostrs sig stabsel`` command prints the same card from a shell.
"""

from __future__ import annotations

import inspect
import re
import textwrap

#: A card never grows past this many lines, so an agent can budget one read.
MAX_LINES = 20

_WIDTH = 100

#: One-line return types for callables whose signature carries no annotation
#: and whose ``Returns`` section is prose rather than a bare type name.
_RETURNS: dict[str, str] = {
    "tune_mstop": "MstopTuning",
    "compare_specifications": "list[dict] (`CompareReport` with output='polars'/'pandas')",
    "conformal_interval": "ConformalInterval",
    "save_model": "pathlib.Path",
    "load_model": "GLMBoostModel | GAMBoostModel | LSSModel",
    "stabsel": "StabSelResult",
    "predict": "numpy.ndarray | polars.DataFrame",
    "gamlss_refit": "GamlssRefit",
    "gam_refit": "GamRefit",
}

#: The shortest call that shows the target in use. Kept hand-written for the
#: high-churn targets; other names fall back to their docstring's doctest.
_EXAMPLES: dict[str, str] = {
    "tune_mstop": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'tuned = mb.tune_mstop("DEXfat ~ bbs(age, df=4)", df)\n'
        "tuned.mstop, tuned.at_boundary  # boundary=True: the risk had not turned"
    ),
    "compare_specifications": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        "rows = mb.compare_specifications(\n"
        '    {"base": "bbs(age, df=4)", "smooth": "bbs(age, df=6)"}, df, "DEXfat")\n'
        'rows[0]["specification"], rows[0]["score"]  # the first entry is the reference'
    ),
    "scan_lss_families": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'rows = mb.scan_lss_families("DEXfat ~ bbs(age, df=4)", df, support="realplus")\n'
        "rows[0]  # best family first; one row per candidate"
    ),
    "GamlssRefit": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'model = mb.gamboostLSS("DEXfat ~ bbs(age, df=4)", df, families=mb.GaussianLSS())\n'
        "refit = mb.gamlss_refit(model)\n"
        'refit.effects(df, parameter="mu")'
    ),
    "conformal_interval": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'fit = mb.gamboost("DEXfat ~ bbs(age, df=4)", df, family=mb.Gaussian())\n'
        "ci = mb.conformal_interval(fit, level=0.9, seed=1)\n"
        "ci.lower, ci.upper  # (n,) bounds, marginal coverage"
    ),
    "GAMBoostModel.predict": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'fit = mb.gamboost("DEXfat ~ bbs(age, df=4)", df, family=mb.Gaussian())\n'
        'fit.predict(newdata=df, type="response")\n'
        'fit.predict(df, mstop=100, which="age (bbs)")  # one learner\'s contribution'
    ),
    "save_model": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'fit = mb.gamboost("DEXfat ~ bbs(age, df=4)", df, family=mb.Gaussian())\n'
        'path = mb.save_model(fit, "fit.json")  # scoring_only=True drops training state\n'
        "mb.load_model(path).predict(df)"
    ),
    "load_model": (
        "import mboostrs as mb\n"
        "df = mb.datasets.bodyfat()\n"
        'fit = mb.load_model("fit.json")\n'
        "fit.predict(df)  # the same predictions the saved fit produced"
    ),
    "stabsel": (
        "import mboostrs as mb\n"
        'model = mb.glmboost("y ~ x", df, control=mb.boost_control(50))\n'
        "result = mb.stabsel(model, q=5, PFER=1)  # 2*B fits under sampling_type='SS'\n"
        "result.selected_names  # the stable set"
    ),
    "api_card": (
        "import mboostrs as mb\n"
        "print(mb.api_card(mb.tune_mstop))\n"
        "# or from a shell: python -m mboostrs sig tune_mstop"
    ),
    "agent_guide": (
        "import mboostrs as mb\n"
        'print(mb.agent_guide("index"))\n'
        'print(mb.agent_guide("fit-and-tune"))'
    ),
}

_DOCTEST = re.compile(r"^\s*(?:>>>|\.\.\.)\s?(.*)$", re.M)


def _resolve(obj):
    """``(target, display_name)`` for an object or a name.

    A string is resolved from the ``mboostrs`` top level first, then as a
    dotted path (``GAMBoostModel.predict``), then as a bare method name that
    is unique across the fitted-model classes. An object keeps its own name.
    """
    if not isinstance(obj, str):
        return obj, getattr(obj, "__name__", type(obj).__name__)
    import mboostrs

    name = obj.strip()
    if "." in name:
        head, *rest = name.split(".")
        target = getattr(mboostrs, head, None)
        if target is None:
            raise KeyError(name)
        for attribute in rest:
            target = getattr(target, attribute)
        return target, name
    if hasattr(mboostrs, name):
        return getattr(mboostrs, name), name
    for owner in ("GAMBoostModel", "GLMBoostModel", "LSSModel", "GamlssRefit", "GamRefit"):
        cls = getattr(mboostrs, owner, None)
        if cls is not None and hasattr(cls, name):
            return getattr(cls, name), f"{owner}.{name}"
    raise KeyError(name)


def _signature(obj) -> str:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in ("self", "cls"):
        signature = signature.replace(parameters=parameters[1:])
    # The return type is a separate line's business, so it is not repeated in
    # the signature half of the header.
    signature = signature.replace(return_annotation=inspect.Signature.empty)
    text = str(signature)
    # `from __future__ import annotations` makes every annotation a string;
    # the quotes are noise in a one-line card.
    text = re.sub(r"(:\s*)'([^']*)'", r"\1\2", text)
    # A sentinel default renders as a memory address; a name is more useful.
    text = re.sub(r"=<object object at 0x[0-9a-f]+>", "=<sentinel>", text)
    return re.sub(r"(->\s*)'([^']*)'", r"\1\2", text)


def _return_type(obj, name: str) -> str:
    if inspect.isclass(obj):
        return f"{name.split('.')[-1]} instance"
    try:
        annotation = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        annotation = inspect.Signature.empty
    if annotation is not inspect.Signature.empty:
        text = annotation if isinstance(annotation, str) else getattr(
            annotation, "__name__", None) or str(annotation)
        if text and text != "None":
            return text
    short = name.split(".")[-1]
    if short in _RETURNS:
        return _RETURNS[short]
    doc = inspect.getdoc(obj) or ""
    match = re.search(r"\nReturns\n-+\n\s*(\S[^\n]*)", doc)
    if match:
        return match.group(1).strip().rstrip(".")
    return "see the docstring"


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _example(obj, name: str) -> str:
    short = name.split(".")[-1]
    for key in (name, short):
        if key in _EXAMPLES:
            return _EXAMPLES[key]
    doc = inspect.getdoc(obj) or ""
    return "\n".join(_DOCTEST.findall(doc))


def api_card(obj, *, width: int = _WIDTH) -> str:
    """A bounded card for one function, method or class.

    Prints the signature, a one-line return type and a minimal call in at most
    ``MAX_LINES`` lines. ``obj`` is the object itself (``mb.tune_mstop``) or
    its name (``"tune_mstop"``, ``"GAMBoostModel.predict"``)::

        import mboostrs as mb
        print(mb.api_card(mb.tune_mstop))

    From a shell::

        python -m mboostrs sig tune_mstop

    Raises ``KeyError`` when a name does not resolve.
    """
    target, name = _resolve(obj)
    header = f"{name}{_signature(target)}"
    returns = _return_type(target, name)
    if returns:
        header = f"{header} -> {returns}"
    lines = textwrap.wrap(header, width=width, subsequent_indent="    ",
                          break_long_words=False, break_on_hyphens=False)
    summary = _summary(target)
    if summary:
        lines.append(f"  {summary}")
    example = _example(target, name)
    if example:
        lines.append("  Example:")
        lines.extend(f"    {line}" for line in example.splitlines())
    return "\n".join(lines[:MAX_LINES])


__all__ = ["MAX_LINES", "api_card"]
