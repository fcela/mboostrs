"""A portable, versioned model artifact.

Models are picklable, but a pickle ties the file to a Python version, a package
version, and the compiled extension: a model archived beside a paper stops
loading after a dependency bump, and a colleague working in R cannot open it at
all.

`save_model` writes JSON instead — a schema-versioned, human-inspectable record
of everything needed to reproduce predictions: the formula, the family and its
parameters, the base-learner specifications, the coefficient path, the offset,
the fitted transform recipe, and the execution report.

    mb.save_model(fit, "model.json")
    fit2 = mb.load_model("model.json")

The guarantee is that `fit2` predicts identically to `fit`, which is what the
round-trip tests assert.

Values are encoded with explicit type tags (`{"__type__": "ndarray", ...}`)
rather than positionally, so the format stays readable and a future reader can
reject anything it does not understand instead of guessing.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

__all__ = ["load_model", "save_model", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

#: Class names the artifact format can name in its ``model_class`` field, and
#: the classes they decode to. This is a wire-format registry, not a dispatch
#: table — deciding whether a live object is serializable goes through
#: ``_dispatch`` so it survives subclassing.
#:
#: `LSSModel` was added without a schema bump: the document shape is unchanged,
#: and the only new things a reader can meet are this `model_class` value and
#: the `lss_family` type tag, which cannot appear without it. The Rust and R
#: readers already refuse an unrecognised `model_class` by name, so an older
#: reader given a distributional artifact says so rather than misreading it.
_MODEL_CLASSES = ("GLMBoostModel", "GAMBoostModel", "LSSModel")

#: The subset the Rust and R readers can score. Python reads all of
#: `_MODEL_CLASSES`.
_PORTABLE_MODEL_CLASSES = ("GLMBoostModel", "GAMBoostModel")


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def _encode(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.ravel().tolist(),
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {"__type__": "dict", "items": {k: _encode(v) for k, v in value.items()}}
        return {
            "__type__": "pairs",
            "items": [[_encode(k), _encode(v)] for k, v in value.items()],
        }

    module = type(value).__module__ or ""
    name = type(value).__name__

    # LSSModel.__getstate__ serialises its whole __dict__, which carries the
    # training frame it needs on load to rebuild the learner stores. GLM and
    # GAM models whitelist plain arrays instead, so only the LSS path reached
    # here and only when fitted from a polars or pandas frame.
    if name == "DataFrame" and (module.startswith("polars") or module.startswith("pandas")):
        columns = list(value.columns)
        return {
            "__type__": "frame",
            "backend": "polars" if module.startswith("polars") else "pandas",
            "columns": columns,
            "data": {c: _encode(np.asarray(value[c])) for c in columns},
        }

    if module.startswith("mboostrs.families"):
        return {"__type__": "family", "name": name, "params": _encode(vars(value))}
    if module.startswith("mboostrs.gamlss") and name.endswith("LSS"):
        # Distributional families are almost all stateless — the parameter
        # count and names are class facts — so this usually encodes an empty
        # mapping. `DirichletLSS(K)` is the exception, and its `vars()` also
        # carries `n_params` and `param_names`, which the constructor derives
        # rather than accepts; `_constructor_arguments` filters those out on
        # the way back.
        # `param_names` rides at the top level (not inside `params`, which
        # reconstructs the constructor arguments) so a reader without the
        # Python class can still name the parameter blocks.
        return {"__type__": "lss_family", "name": name,
                "params": _encode(vars(value)),
                "param_names": list(value.param_names)}
    if module.startswith("mboostrs.learners"):
        return {"__type__": "learner", "name": name, "params": _encode(vars(value))}
    if name == "ExecutionReport":
        return {"__type__": "execution_report", "fields": _encode(value.to_dict())}
    if name == "TransformRecipe":
        return {"__type__": "transform_recipe", "steps": _encode(value.describe())}
    if name == "_Step":
        # A single recipe step, stored on its own: the response transform a
        # model carries beside its predictor recipe. The fields are the
        # dataclass's own (name, kind, sources, expression, params).
        return {"__type__": "transform_step", "fields": _encode(vars(value))}
    if module.startswith("mboostrs.interactions"):
        # `TensorInteraction` and its two `_Marginal`s are plain records of how
        # to rebuild a tensor design at new rows — the recipe a `%X%` term or a
        # factor `by=` leaves behind. They hold only names, level lists and
        # numbers, so `vars()` round-trips them.
        return {"__type__": "interaction", "name": name, "fields": _encode(vars(value))}

    raise TypeError(
        f"cannot serialize {module}.{name!r}; the model artifact supports plain "
        f"data, families, base learners, execution reports, and transform recipes"
    )


def _constructor_arguments(cls, stored: dict) -> dict:
    """Keep only the stored attributes `cls.__init__` actually accepts.

    An encoded object is its `vars()`, which can include attributes the
    constructor *derives* rather than takes — `DirichletLSS(K)` stores `K`
    alongside the `n_params` and `param_names` it computes from it. Passing
    those back raises `TypeError`. Reading the signature is the same approach
    the learner branch below uses for `Bspatial`'s two positional names.

    A constructor taking `**kwargs` accepts everything, so nothing is dropped
    there.
    """
    import inspect

    if cls.__init__ is object.__init__:
        # No constructor of its own: it takes no arguments at all.
        return {}
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(stored)
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(stored)
    accepted = {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in stored.items() if k in accepted}


def _unknown_field(owner, field):
    return ValueError(
        f"the artifact's {owner} carries a field {field!r} this release of "
        f"mboostrs does not know; it was probably written by a newer release, "
        f"and ignoring the field could change predictions. Load it with the "
        f"release that wrote it."
    )


def _construct_checked(cls, stored, owner):
    """Construct `cls` from its stored attributes, refusing unknown ones.

    A stored attribute the constructor does not take is accepted only when
    the constructed object derives the same value itself (`DirichletLSS`
    stores `n_params` beside the `K` it is computed from). Anything else is a
    field this reader does not understand, which dropping would silently
    change the model.
    """
    arguments = _constructor_arguments(cls, stored)
    obj = cls(**arguments)
    for name, value in stored.items():
        if name in arguments:
            continue
        if not hasattr(obj, name):
            raise _unknown_field(owner, name)
        derived = getattr(obj, name)
        try:
            same = bool(np.all(np.asarray(derived) == np.asarray(value)))
        except (TypeError, ValueError):
            same = derived == value
        if not same:
            raise _unknown_field(owner, name)
    return obj


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("__type__")
    if kind is None:
        return {k: _decode(v) for k, v in value.items()}
    if kind == "transform_step":
        from mboostrs.transforms import _Step

        fields = _decode(value["fields"])
        return _Step(
            name=fields["name"],
            kind=fields["kind"],
            sources=tuple(fields["sources"]),
            expression=fields["expression"],
            params=tuple(fields["params"]),
        )
    if kind == "ndarray":
        array = np.asarray(value["data"], dtype=value["dtype"])
        return array.reshape(value["shape"])
    if kind == "tuple":
        return tuple(_decode(v) for v in value["items"])
    if kind == "dict":
        return {k: _decode(v) for k, v in value["items"].items()}
    if kind == "pairs":
        return {_decode(k): _decode(v) for k, v in value["items"]}
    if kind == "frame":
        columns = {c: _decode(value["data"][c]) for c in value["columns"]}
        backend = value.get("backend", "polars")
        try:
            if backend == "pandas":
                import pandas as pd  # noqa: PLC0415

                return pd.DataFrame(columns)
            import polars as pl  # noqa: PLC0415

            return pl.DataFrame(columns)
        except ImportError:
            # Every consumer of the stored data accepts a column mapping, so
            # a missing backend degrades rather than failing the load.
            return columns
    if kind == "family":
        from mboostrs import families

        cls = getattr(families, value["name"], None)
        if cls is None:
            raise ValueError(f"unknown family {value['name']!r} in artifact")
        return _construct_checked(cls, _decode(value["params"]), value["name"])
    if kind == "lss_family":
        from mboostrs import gamlss

        cls = getattr(gamlss, value["name"], None)
        if cls is None or not value["name"].endswith("LSS"):
            raise ValueError(
                f"unknown distributional family {value['name']!r} in artifact"
            )
        return _construct_checked(cls, _decode(value["params"]), value["name"])
    if kind == "learner":
        from mboostrs import learners

        cls = getattr(learners, value["name"], None)
        if cls is None:
            raise ValueError(f"unknown base learner {value['name']!r} in artifact")
        params = _decode(value["params"])
        # Read the constructor's own positional parameter names (e.g. `name`
        # for most learners, `names` for Btree, `name1`/`name2` for the
        # two-coordinate Bspatial) instead of guessing a fixed set, so a
        # learner whose constructor doesn't match the single-`name` shape
        # still round-trips.
        sig = inspect.signature(cls.__init__)
        positional_names = [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        positional_args = [params.pop(n) for n in positional_names]
        # Attributes a fit sets after construction -- `Btree.categorical`,
        # which records which inputs are factor codes -- are stored with the
        # rest but are not constructor arguments. Passing them to the
        # constructor made every tree model unloadable; dropping them would
        # lose state prediction needs. Construct from the accepted arguments,
        # then restore the others.
        accepted = _constructor_arguments(cls, params)
        learner = cls(*positional_args, **accepted)
        for name, stored in params.items():
            if name not in accepted:
                if not hasattr(learner, name):
                    raise _unknown_field(value["name"], name)
                setattr(learner, name, stored)
        return learner
    if kind == "execution_report":
        from mboostrs.report import ExecutionReport

        return ExecutionReport.from_mapping(_decode(value["fields"]))
    if kind == "transform_recipe":
        return _decode_recipe(value["steps"])
    if kind == "interaction":
        import mboostrs.interactions as interactions

        cls = getattr(interactions, value["name"])
        return cls(**_decode(value["fields"]))

    raise ValueError(f"unknown encoded type {kind!r} in artifact")


def _decode_recipe(steps):
    from mboostrs.transforms import TransformRecipe, _Step

    rebuilt = []
    for step in _decode(steps):
        params = step["params"]
        # `describe()` flattens params to a list; factor levels must go back to
        # a tuple so encoding stays stable across a save/load/save cycle.
        if step["kind"] == "factor":
            params = (tuple(params[0]),)
        elif step["kind"] == "dummy":
            params = (params[0], tuple(params[1]))
        elif step["kind"] == "ecdf":
            params = (tuple(params[0]), tuple(params[1]))
        else:
            params = tuple(params)
        # `sources` replaced `source` when I() gained multi-column support.
        # An artifact written before that carries only the singular key, and
        # every transform it could contain reads exactly one column.
        sources = step.get("sources") or [step["source"]]
        rebuilt.append(
            _Step(
                name=step["derived"],
                kind=step["kind"],
                sources=tuple(sources),
                expression=step["expression"],
                params=params,
            )
        )
    return TransformRecipe(tuple(rebuilt))


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def _scoring_only_state(state, model):
    """The state a scoring-only artifact keeps: prediction, not refitting.

    Drops the response column the caller measured from every stored column
    set (a loaded model predicts from predictors; training-row diagnostics
    need the response and are not what a scoring artifact claims), and the
    formula/control a refit needs,
    so `update`/`with_mstop` raise their retention error on a loaded
    scoring-only model instead of quietly refitting from state the artifact
    no longer holds. Recipes, interaction designs, the response transform,
    fitted values, coefficients and the learner definitions all stay: they
    are what `predict` reads.
    """
    stripped = dict(state)
    # The column the caller measured, and with a response transform the
    # derived column too: the stored response name is the derived column
    # (`log__rent`) and the original outcome is the transform step's source
    # (`rent`). Every column store leaves them out -- the training frame
    # (`_data`) and a gamboost model's cached learner columns
    # (`replay_columns`), which carried the response for `aic()` and so kept
    # it in every scoring-only gamboost artifact.
    response = getattr(model, "_response_name", None)
    names = set(response) if isinstance(response, (tuple, list)) else {response}
    step = getattr(model, "_response_transform", None)
    if step is not None:
        names.add(step.source)
    names = {name for name in names if isinstance(name, str)}
    for key in ("_data", "replay_columns"):
        data = stripped.get(key)
        if data is None or not names:
            continue
        if isinstance(data, dict):
            stripped[key] = {k: v for k, v in data.items() if k not in names}
        elif hasattr(data, "drop") and hasattr(data, "columns"):
            # polars: drop by name; pandas shares the spelling.
            present = [name for name in names if name in data.columns]
            if present:
                stripped[key] = data.drop(present)
    if names:
        stripped["scoring_only"] = True
    # The working residual a gamboost fit resumes from: with the fitted values
    # it gives back the response, which a scoring-only artifact leaves out.
    if stripped.get("residual") is not None:
        stripped["residual"] = None
    # Set to None rather than removed: the refit guards read `is None` to
    # say the model cannot be refit, and a missing attribute would surface
    # as AttributeError instead of that message.
    for key in ("_formula", "_control", "_refit_bundle", "_raw_formula",
                "_raw_data", "_raw_inputs"):
        if key in stripped:
            stripped[key] = None
    return stripped


def save_model(model, path, *, scoring_only=False,
               indent: int | None = None) -> Path:
    """Write `model` to `path` as a schema-versioned JSON artifact.
    Usage::

        import mboostrs as mb
        mb.save_model(fit, "fit.json")
        mb.load_model("fit.json").predict(newdata)

    Accepts `GLMBoostModel`, `GAMBoostModel`, and `LSSModel`. A distributional
    artifact carries one learner block per distribution parameter and the
    family's own state; it is readable by `load_model` here and by the R
    reader for inspection.

    A full artifact also stores the fit's call after preprocessing (the
    rewritten formula, the training frame, the weights, the offset and the
    fit arguments), so a loaded model can `continue_to`, `update`, `cvrisk`
    and run the other refits. That record sits in a top-level
    ``refit_bundle`` key, which the Rust and R readers do not read. When it
    holds a value the format cannot encode (an object in ``env``), the
    artifact is written without it and the loaded model refuses refits.

    ``scoring_only=True`` keeps what prediction needs and drops the rest of
    the training state: the response column leaves every stored column set,
    and the formula and control a refit would need are omitted, so a loaded
    scoring-only model predicts (on new rows and on its stored fitted path)
    and refuses `update`/`with_mstop` with the retention error rather than
    silently carrying a full training checkpoint. Nothing about the fitted
    model changes; the same fit saved both ways predicts identically.

    A scoring-only artifact is not free of training rows. It keeps the
    training predictor columns that `gamboost` and distributional models
    rebuild their learners from, the observation weights, and the in-sample
    fitted values, which are a smooth function of the response. Treat it as
    containing row-level training data when deciding where it may be shared.
    """
    from mboostrs._dispatch import is_lss, is_standard, model_kind_name

    if not (is_standard(model) or is_lss(model)):
        raise TypeError(
            f"save_model supports {', '.join(_MODEL_CLASSES)}, got "
            f"{model_kind_name(model)!r}"
        )
    # The wire name is the *base* class, so a user subclass round-trips back
    # through the base's ``__setstate__`` rather than failing to resolve.
    class_name = model_kind_name(model)

    from mboostrs import __version__

    recipe = getattr(model, "_transform_recipe", None)
    state = model.__getstate__()
    # The refit bundle travels beside the state, not in it. It holds the
    # training frame, which the Rust and R readers cannot decode, and they
    # decode the whole state object; a top-level key they never read keeps
    # the artifact portable.
    refit_bundle = state.pop("refit_bundle", None)
    if scoring_only:
        state = _scoring_only_state(state, model)
        refit_bundle = None
    document = {
        "schema_version": SCHEMA_VERSION,
        "mboostrs_version": __version__,
        "model_class": class_name,
        "parity_notes": model.parity_notes(
            include_global=True, output="records"
        ),
        "state": _encode(state),
        "transform_recipe": _encode(recipe) if recipe else None,
    }
    if refit_bundle is not None:
        try:
            document["refit_bundle"] = _encode(refit_bundle)
        except TypeError:
            # An `env` entry or a learner argument of a type the format does
            # not encode. The artifact then predicts but does not refit, which
            # is what every artifact did before the bundle was stored.
            pass

    document["content_sha256"] = _content_digest(document)
    path = Path(path)
    # Write beside the target and rename over it, so a crash or a full disk
    # leaves the previous file (or none) rather than a truncated artifact.
    import os
    import tempfile

    handle, temporary = tempfile.mkstemp(dir=path.parent or Path("."),
                                         prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=indent))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _content_digest(document) -> str:
    """SHA-256 of the document without its digest field, serialized
    canonically (sorted keys, no whitespace), so indentation does not change
    it and any edit to the content does."""
    body = {k: v for k, v in document.items() if k != "content_sha256"}
    import hashlib

    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_model(path):
    """Read a model artifact written by :func:`save_model`.
    Usage::

        import mboostrs as mb
        fit = mb.load_model("fit.json")
        fit.predict(newdata)

    The inverse of :func:`save_model`, restoring the recorded model class with
    all of its state, including a distributional artifact's per-parameter
    learners and any response transform.
    """
    document = json.loads(Path(path).read_text())
    # Artifacts written before the digest was added have none to check.
    expected = document.get("content_sha256")
    if expected is not None and expected != _content_digest(document):
        raise ValueError(
            f"{path}: the artifact's content does not match its checksum; the "
            f"file was edited or damaged after it was written")

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"artifact schema version {version!r} is not supported by this "
            f"release (expected {SCHEMA_VERSION}). Re-export it with the "
            f"version of mboostrs that wrote it."
        )

    class_name = document.get("model_class")
    if class_name not in _MODEL_CLASSES:
        raise ValueError(f"unknown model class {class_name!r} in artifact")

    if class_name == "GLMBoostModel":
        from mboostrs.glmboost import GLMBoostModel as cls
    elif class_name == "GAMBoostModel":
        from mboostrs.gamboost import GAMBoostModel as cls
    else:
        from mboostrs.gamlss import LSSModel as cls

    model = cls.__new__(cls)
    state = _decode(document["state"])
    if document.get("refit_bundle") is not None:
        state["refit_bundle"] = _decode(document["refit_bundle"])
    model.__setstate__(state)

    # Artifacts written before the recipe joined the model state carry it in
    # this document key instead; a newer one has both, with the same value.
    recipe = document.get("transform_recipe")
    if recipe:
        model._transform_recipe = _decode(recipe)
    return model
