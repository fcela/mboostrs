#!/usr/bin/env python3
"""Generate the per-symbol Python API manual under docs/book/api/.

One Markdown page is written for every public function and class exported by
``mboostrs`` (``mboostrs.__all__``). A class page also documents each public
method and property defined on the class, as a section with a stable anchor.
Each entry shows

* the live signature (``inspect.signature``) and return annotation,
* the docstring summary and description,
* the Parameters, Returns, Raises and Attributes sections of the numpydoc
  docstring, and the exception types raised by ``raise`` statements in the
  function body (read with ``ast``),
* a link to the source lines on GitHub.

Run from the repository root with the environment that has the built
extension::

    source .venv/bin/activate
    python tools/generate_api_manual.py            # write docs/book/api/
    python tools/generate_api_manual.py --check    # fail if pages are stale
    python tools/generate_api_manual.py --audit    # list missing docstring parts

``--check`` regenerates in memory and compares with the committed pages, so a
changed signature or docstring fails until the pages are regenerated.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/book/api"
SOURCE_URL = "https://github.com/fcela/mboostrs/blob/main"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

SECTION_NAMES = {
    "Parameters", "Returns", "Yields", "Raises", "Attributes", "Examples",
    "Notes", "See Also", "References", "Warns", "Other Parameters",
}
STRUCTURED = ("Parameters", "Other Parameters", "Returns", "Yields", "Raises",
              "Attributes")


# ---------------------------------------------------------------- docstrings
def split_doc(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (free text before the first section, {section: body})."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    head: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] = head
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (line.strip() in SECTION_NAMES and nxt.strip()
                and set(nxt.strip()) <= {"-"} and len(nxt.strip()) >= 3):
            current = sections.setdefault(line.strip(), [])
            i += 2
            continue
        current.append(line)
        i += 1
    return "\n".join(head).strip(), {k: "\n".join(v).strip("\n")
                                      for k, v in sections.items()}


def parse_entries(body: str) -> list[tuple[str, str]]:
    """Split a numpydoc section into (header line, description) pairs."""
    entries: list[tuple[str, list[str]]] = []
    for line in body.splitlines():
        if line and not line[0].isspace():
            entries.append((line.strip(), []))
        elif entries:
            entries[-1][1].append(line)
        elif line.strip():
            entries.append((line.strip(), []))
    return [(h, textwrap.dedent("\n".join(d)).strip()) for h, d in entries]


def rst_to_md(text: str) -> str:
    """Convert the few RST constructs the docstrings use to Markdown."""
    text = re.sub(r":(?:class|func|meth|mod|attr|exc|obj|data):`~?([^`]+)`",
                  r"`\1`", text)
    text = re.sub(r"(?<!`)``([^`\n]+)``(?!`)", r"`\1`", text)
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::"):
            stripped = line.rstrip()[:-2].rstrip()
            if stripped:
                out.append(stripped + ":")
            out.append("")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out.append("```python")
            out.extend(textwrap.dedent("\n".join(block)).splitlines())
            out.append("```")
            out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def md(text: str) -> str:
    return rst_to_md(text)


# ------------------------------------------------------------------- source
def source_info(obj) -> tuple[str, int, int] | None:
    try:
        target = inspect.unwrap(obj)
        if isinstance(target, property):
            target = target.fget
        path = Path(inspect.getsourcefile(target) or "").resolve()
        lines, start = inspect.getsourcelines(target)
        rel = path.relative_to(ROOT)
    except (TypeError, OSError, ValueError):
        return None
    return rel.as_posix(), start, start + len(lines) - 1


def source_link(obj) -> str:
    info = source_info(obj)
    if info is None:
        return "Source: compiled extension module (no Python source)."
    rel, a, b = info
    return f"Source: [`{rel}` lines {a}-{b}]({SOURCE_URL}/{rel}#L{a}-L{b})"


def raised_names(obj) -> list[str]:
    """Exception class names in `raise` statements of a function body."""
    try:
        target = inspect.unwrap(obj)
        if isinstance(target, property):
            target = target.fget
        tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    except (TypeError, OSError, SyntaxError, ValueError):
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = ast.unparse(exc)
            if name not in found:
                found.append(name)
    return found


# -------------------------------------------------------------- signatures
def signature_of(obj) -> tuple[str, str]:
    """Return (parameter list text, return annotation text)."""
    sig = _signature(obj)
    if sig is None:
        return "(...)", ""
    ret = ""
    if sig.return_annotation is not inspect.Signature.empty:
        ret = _ann(sig.return_annotation)
        sig = sig.replace(return_annotation=inspect.Signature.empty)
    return str(sig), ret


def _signature(obj):
    """Signature with string annotations resolved to types when possible."""
    for kwargs in ({"eval_str": True}, {}):
        try:
            return inspect.signature(obj, **kwargs)
        except (TypeError, ValueError, NameError, SyntaxError, AttributeError):
            continue
    return None


def _ann(a) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, type):
        return a.__qualname__
    return str(a).replace("typing.", "")


def _clean(text: str) -> str:
    """Remove memory addresses so output is reproducible."""
    return re.sub(r"<object object at 0x[0-9a-f]+>", "<unset>",
                  re.sub(r" at 0x[0-9a-f]+", "", text))


def render_signature(name: str, obj) -> list[str]:
    params, ret = signature_of(obj)
    if "." in name:
        params = params.replace("(self, ", "(", 1).replace("(self)", "()", 1)
    text = _clean(f"{name}{params}" + (f" -> {ret}" if ret else ""))
    if len(text) > 88:
        sig = _signature(obj)
        if sig is not None:
            body = ",\n".join("    " + str(p) for p in sig.parameters.values()
                              if not ("." in name and p.name == "self"))
            text = _clean(f"{name}(\n{body}\n)" + (f" -> {ret}" if ret else ""))
    return ["```python", text, "```", ""]


# ------------------------------------------------------------ entry writing
def render_entry(level: int, qualname: str, obj, *, kind: str,
                 callable_sig: bool = True, member_of: str | None = None
                 ) -> list[str]:
    h = "#" * level
    anchor = qualname.lower().replace(".", "-")
    doc = inspect.getdoc(obj) if not isinstance(obj, property) else obj.__doc__
    head, secs = split_doc(doc)
    out = [f"({anchor})=", f"{h} `{qualname}`", ""]
    out.append(f"*{kind}*" + (f" of `{member_of}`" if member_of else ""))
    out.append("")
    if callable_sig:
        target = obj.fget if isinstance(obj, property) else obj
        if isinstance(obj, property):
            ann = ""
            try:
                r = inspect.signature(target).return_annotation
                if r is not inspect.Signature.empty:
                    ann = _ann(r)
            except (TypeError, ValueError):
                pass
            out += ["```python", f"{qualname}" + (f"  # -> {ann}" if ann else ""),
                    "```", ""]
        else:
            out += render_signature(qualname, target)
    out.append(md(head) if head else "*No description in the docstring.*")
    out.append("")
    for sec in STRUCTURED:
        body = secs.get(sec)
        if not body:
            continue
        out += [f"**{sec}**", ""]
        if sec in ("Parameters", "Other Parameters", "Attributes"):
            for hdr, desc in parse_entries(body):
                name, _, typ = hdr.partition(" : ")
                line = f"- `{name.strip()}`" + (f" ({typ.strip()})" if typ else "")
                if desc:
                    desc = md(desc)
                    first, *rest = desc.splitlines()
                    line += f": {first}"
                    out.append(line)
                    out += ["  " + r if r.strip() else "" for r in rest]
                else:
                    out.append(line)
        elif sec == "Raises":
            for hdr, desc in parse_entries(body):
                out.append(f"- `{hdr}`" + (f": {md(desc)}" if desc else ""))
        else:
            out.append(md(body))
        out.append("")
    # Exception types visible in the function body but absent from Raises.
    documented = set(re.findall(r"\b[A-Z]\w*(?:Error|Warning|Exception)\b",
                                secs.get("Raises", "")))
    if callable_sig and not isinstance(obj, type):
        extra = [n for n in raised_names(obj)
                 if n.split(".")[-1] not in documented]
        if extra:
            out += ["**Exceptions raised in the body**", "",
                    "Found by reading the function's `raise` statements; "
                    "conditions are in the source.", ""]
            out += [f"- `{n}`" for n in extra[:12]]
            out.append("")
    for sec in ("Notes", "Examples", "See Also"):
        if secs.get(sec):
            out += [f"**{sec}**", "", md(secs[sec]), ""]
    out += [source_link(obj), ""]
    return out


def public_members(cls) -> list[tuple[str, object, str]]:
    """Public methods and properties defined in the class body."""
    rows = []
    for name, raw in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(raw, property):
            rows.append((name, raw, "Property"))
        elif isinstance(raw, (staticmethod, classmethod)):
            rows.append((name, getattr(cls, name), "Method"))
        elif inspect.isfunction(raw):
            rows.append((name, raw, "Method"))
    return rows


def page_for(name: str, obj) -> str:
    is_cls = inspect.isclass(obj)
    kind = "Class" if is_cls else "Function"
    mod = getattr(obj, "__module__", "mboostrs")
    out = [f"# `{name}`", "",
           f"Defined in `{mod}`; exported as `mboostrs.{name}`.", ""]
    out += render_entry(2, name, obj, kind=kind)[2:]  # drop duplicate heading
    # render_entry emits an anchor and heading first; keep body only.
    members = public_members(obj) if is_cls else []
    if members:
        out += ["## Members", "", "| Member | Kind | Summary |", "|---|---|---|"]
        for mname, m, mk in members:
            d = (m.__doc__ if isinstance(m, property) else inspect.getdoc(m)) or ""
            summary = d.strip().splitlines()[0].replace("|", "\\|") if d.strip() else ""
            out.append(f"| [`{mname}`](#{name.lower()}-{mname.lower()}) | {mk} | {summary} |")
        out.append("")
        for mname, m, mk in members:
            out += render_entry(2, f"{name}.{mname}", m, kind=mk,
                                member_of=name)
    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------------ driver
def public_symbols() -> list[tuple[str, object]]:
    rows = []
    for name in sorted(mb.__all__):
        obj = getattr(mb, name, None)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            rows.append((name, obj))
    return rows


def page_path(name: str, obj) -> str:
    """Functions and classes sit in separate folders: ``fit`` and ``Fit`` would
    otherwise name the same file on a case-insensitive file system."""
    return f"{'classes' if inspect.isclass(obj) else 'functions'}/{name}.md"


def build() -> dict[str, str]:
    pages: dict[str, str] = {}
    index = ["# Python API manual", "",
             "One page per public function and class exported by `mboostrs`, "
             "generated from the installed package by "
             "`tools/generate_api_manual.py`. Each page gives the live "
             "signature, the parameters, return value and errors from the "
             "docstring, and a link to the source lines. Class pages document "
             "every public method and property. The task-oriented "
             "[API reference](../appendix/api_reference.ipynb) groups the same "
             "names by purpose.", "",
             "Frequently used entries: "
             "[`gamboost`](functions/gamboost.md), [`glmboost`](functions/glmboost.md), "
             "[`GAMBoostModel.predict`](classes/GAMBoostModel.md#gamboostmodel-predict), "
             "[`LSSModel.predict`](classes/LSSModel.md#lssmodel-predict), "
             "[`CvRiskResult`](classes/CvRiskResult.md), [`FoldScheme`](classes/FoldScheme.md).",
             "", "## Functions", "", "| Name | Summary |", "|---|---|"]
    classes, funcs = [], []
    for name, obj in public_symbols():
        rel = page_path(name, obj)
        pages[rel] = page_for(name, obj)
        d = (inspect.getdoc(obj) or "").strip()
        summary = d.splitlines()[0].replace("|", "\\|") if d else ""
        (classes if inspect.isclass(obj) else funcs).append(
            f"| [`{name}`]({rel}) | {summary} |")
    index += funcs + ["", "## Classes", "", "| Name | Summary |", "|---|---|"]
    index += classes
    pages["index.md"] = "\n".join(index) + "\n"
    return pages


def audit() -> int:
    """List public symbols and members whose docstring lacks parts."""
    missing = 0
    def check(label, obj, params_expected=True):
        nonlocal missing
        head, secs = split_doc(obj.__doc__ if isinstance(obj, property)
                               else inspect.getdoc(obj))
        problems = []
        if not head:
            problems.append("no docstring")
        try:
            sig = inspect.signature(obj.fget if isinstance(obj, property) else obj)
            names = [p for p in sig.parameters if p not in ("self", "cls") and not p.startswith("_")
                     and not p.startswith("*")]
        except (TypeError, ValueError):
            names = []
        if names and params_expected and "Parameters" not in secs \
                and not (inspect.isclass(obj) and "Attributes" in secs):
            problems.append(f"no Parameters section ({len(names)} args)")
        elif "Parameters" in secs:
            documented = {h.split(" : ")[0].replace("*", "").strip()
                          for h, _ in parse_entries(secs["Parameters"])}
            flat = {x.strip() for d in documented for x in d.split(",")}
            gap = [n for n in names if n not in flat]
            if gap:
                problems.append("undocumented: " + ", ".join(gap))
        if problems:
            missing += 1
            print(f"{label}: " + "; ".join(problems))
    for name, obj in public_symbols():
        check(name, obj)
        if inspect.isclass(obj):
            for mname, m, _ in public_members(obj):
                if not isinstance(m, property):
                    check(f"{name}.{mname}", m)
    print(f"{missing} entries with gaps")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    if args.audit:
        return audit()
    pages = build()
    if args.check:
        stale = [n for n, t in pages.items()
                 if not (OUT / n).exists() or (OUT / n).read_text() != t]
        extra = [p.relative_to(OUT).as_posix() for p in OUT.rglob("*.md")
                 if p.relative_to(OUT).as_posix() not in pages]
        if stale or extra:
            print("stale API manual pages:", ", ".join(stale + extra))
            return 1
        print(f"{len(pages)} pages current")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.rglob("*.md"):
        if old.relative_to(OUT).as_posix() not in pages:
            old.unlink()
    for n, t in pages.items():
        (OUT / n).parent.mkdir(parents=True, exist_ok=True)
        (OUT / n).write_text(t)
    print(f"wrote {len(pages)} pages to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
