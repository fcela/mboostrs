"""Regressions in the PDF patcher for MyST-generated TeX."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _patcher():
    path = Path(__file__).resolve().parents[1] / "docs/book/patch_tex.py"
    spec = spec_from_file_location("book_patch_tex", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_arrow_adjacent_to_words_and_minted_source():
    patcher = _patcher()
    source = (
        r"preexplore\rightarrowreason $x \rightarrow y$ "
        r"\begin{minted}[frame=single]{python}print('a\rightarrowb')\end{minted}"
    )
    result = patcher.ensure_text_arrows(source)
    assert r"preexplore\ensuremath{\rightarrow}reason" in result
    assert r"$x \ensuremath{\rightarrow} y$" in result
    assert r"print('a\rightarrowb')" in result
    assert patcher.ensure_text_arrows(result) == result


def test_rule_only_tail_is_not_padded_into_a_table_row():
    patcher = _patcher()
    body = "Variable & role & statement \\\\\n\\hline\nvalue & role & statement \\\\\n\\hline\n\\bottomrule\n"
    result = patcher.pad_short_header_rows(body, 3)
    assert result.endswith("\\hline\n\\bottomrule\n")
    assert " &  & \\bottomrule" not in result


def _generated_table(patcher, body, n_cols):
    colspec = patcher._colspec([1 / n_cols] * n_cols)
    return f"\\begin{{tabular}}{{{colspec}}}{body}\\end{{tabular}}"


@pytest.mark.parametrize("n_rows, environment", [(2, "tabular"), (30, "longtable")])
def test_wide_count_rank_table_falls_back_to_proportional_columns(n_rows, environment):
    patcher = _patcher()
    values = [
        r"1.488 $\pm$ 0.012", r"1.682 $\pm$ 0.022",
        "0.908", "0.009", "2.371", "0", "0",
    ]
    method = "supplied fixed central quantile"
    row = " & ".join([method, *values]) + " \\\\\n\\hline\n"
    body = (
        "Method & Width & Count size & Coverage & Mean MSE & Interval score & Unbounded & Failed"
        " \\\\\n\\hline\n" + row * n_rows
    )
    assert patcher._mixed_layout(body, 8) is None

    result = patcher.repair_generated_tables(_generated_table(patcher, body, 8))

    spec = patcher._colspec(patcher._column_shares(body, 8))
    assert f"\\begin{{{environment}}}{{{spec}}}" in result
    assert "tabularx" not in result
    assert r"\scriptsize\setlength{\tabcolsep}{3pt}" in result
    assert result.count(row) == n_rows
    assert result.count(method) == n_rows
    for value in values[:5]:
        assert result.count(value) == n_rows
    assert (r"\endhead" in result) == (environment == "longtable")
    # Every proportional column has positive content width after padding
    # at the book's normal font size.
    assert all(
        share * patcher._LINE_EMS > 1.1
        for share in patcher._column_shares(body, 8)
    )


@pytest.mark.parametrize("n_cols", [4, 8, 16])
@pytest.mark.parametrize("n_long", [1, 2])
def test_mixed_layout_reserves_width_for_every_long_column(n_cols, n_long):
    patcher = _patcher()
    row = " & ".join(
        ["a method name that wraps"] * n_long
        + ["1234567890123456"] * (n_cols - n_long)
    )
    body = " & ".join(["Heading"] * n_cols) + " \\\\\n\\hline\n" + row + " \\\\\n"
    mixed = patcher._mixed_layout(body, n_cols)
    if mixed is not None:
        widths, small = mixed
        line_ems = patcher._LINE_EMS * (11 / 9 if small else 1.0)
        remaining = (
            line_ems - sum(w for w in widths if w)
            - patcher._padding_ems(n_cols, small)
        )
        assert remaining >= 6.0
    else:
        assert n_cols >= 8


def test_all_short_columns_keep_existing_layout():
    patcher = _patcher()
    body = " & ".join(["Value"] * 8) + " \\\\\n\\hline\n" + " & ".join(["1"] * 8) + " \\\\\n"
    assert patcher._mixed_layout(body, 8) is None
    result = patcher.repair_generated_tables(_generated_table(patcher, body, 8))
    assert r"\begin{tabular}{llllllll}" in result


def test_mixed_layout_keeps_fitting_short_columns():
    patcher = _patcher()
    body = (
        "Method & Score & Count & Rank \\\\\n\\hline\n"
        "a method name that wraps & 0.123 & 12 & 1 \\\\\n"
    )
    assert patcher._mixed_layout(body, 4) is not None
    result = patcher.repair_generated_tables(_generated_table(patcher, body, 4))
    assert r"\begin{tabularx}{\linewidth}" in result


def test_headings_beside_a_spanner_keep_their_column_positions():
    patcher = _patcher()
    body = (
        r"Method & \multicolumn{3}{c}{Prediction sets} & Mean MSE & Interval score & Unbounded & Failed \\" "\n"
        r"\hline & Width & Count size & Coverage & & & & \\" "\n"
        r"\hline a method name that wraps & 1.488 & 1.682 & 0.908 & 0.009 & 2.371 & 0 & 0 \\" "\n"
    )
    shares = patcher._column_shares(body, 8)
    # These headings occur after a three-column spanner, above zero counts.
    assert shares[6] > shares[7] > 0.06
    without_count_headings = body.replace("Unbounded & Failed", " & ")
    assert shares[6] > patcher._column_shares(without_count_headings, 8)[6]
    # Spanner text itself must not inflate the first column it covers.
    changed_spanner = body.replace("Prediction sets", "An extremely long heading over three columns")
    assert patcher._column_shares(changed_spanner, 8) == shares


# Captured from the count-rank calibration chapter's rendered GT artifact list.
_COUNT_RANK_ARTIFACT_TABLE = r"""\begin{tabular}{p{\dimexpr \linewidth-2\tabcolsep}}
\toprule
Bundled evidence \\
Protocols, complete raw records, summaries and validation \\
Artifact \\
\hline
AUDIT\_ADDENDUM.md \\
INITIAL\_ATTEMPTS.json \\
INPUT\_VALIDATION.json \\
INTEGRATION\_SCORE\_REPLAY.json \\
PROMOTION\_DECISION.json \\
PROTOCOL.md \\
R\_PARITY\_VALIDATION.json \\
R\_PARITY\_VALIDATION\_MEASURED.json \\
SCORE\_REPLAY\_VALIDATION.json \\
SOURCE\_MANIFEST.json \\
VALIDATION.md \\
confirmation\_audit\_gate.json \\
confirmation\_audit\_high.jsonl.metadata.json \\
confirmation\_audit\_high.jsonl.xz \\
confirmation\_audit\_low.jsonl.metadata.json \\
confirmation\_audit\_low.jsonl.xz \\
confirmation\_audit\_original.jsonl.metadata.json \\
confirmation\_audit\_original.jsonl.xz \\
confirmation\_audit\_overdispersed.jsonl.metadata.json \\
confirmation\_audit\_overdispersed.jsonl.xz \\
confirmation\_audit\_summary.json \\
confirmation\_audit\_zero\_inflated.jsonl.metadata.json \\
confirmation\_audit\_zero\_inflated.jsonl.xz \\
confirmation\_gate.json \\
confirmation\_high.jsonl.metadata.json \\
confirmation\_high.jsonl.xz \\
confirmation\_low.jsonl.metadata.json \\
confirmation\_low.jsonl.xz \\
confirmation\_original.jsonl.metadata.json \\
confirmation\_original.jsonl.xz \\
confirmation\_overdispersed.jsonl.metadata.json \\
confirmation\_overdispersed.jsonl.xz \\
confirmation\_summary.json \\
confirmation\_zero\_inflated.jsonl.metadata.json \\
confirmation\_zero\_inflated.jsonl.xz \\
development\_gate.json \\
development\_high.jsonl.metadata.json \\
development\_high.jsonl.xz \\
development\_low.jsonl.metadata.json \\
development\_low.jsonl.xz \\
development\_original.jsonl.metadata.json \\
development\_original.jsonl.xz \\
development\_overdispersed.jsonl.metadata.json \\
development\_overdispersed.jsonl.xz \\
development\_summary.json \\
development\_zero\_inflated.jsonl.metadata.json \\
development\_zero\_inflated.jsonl.xz \\
initial\_development\_high.jsonl.metadata.json \\
initial\_development\_high.jsonl.xz \\
initial\_development\_high.log \\
initial\_development\_low.jsonl.metadata.json \\
initial\_development\_low.jsonl.xz \\
initial\_development\_low.log \\
initial\_development\_original.jsonl.metadata.json \\
initial\_development\_original.jsonl.xz \\
initial\_development\_original.log \\
initial\_development\_overdispersed.jsonl.metadata.json \\
initial\_development\_overdispersed.jsonl.xz \\
initial\_development\_overdispersed.log \\
initial\_development\_zero\_inflated.jsonl.metadata.json \\
initial\_development\_zero\_inflated.jsonl.xz \\
initial\_development\_zero\_inflated.log \\
XZ archives use strict JSON with reversible nonfinite locations; decode with benches.evidence\_float\_codec. \\
\bottomrule
\end{tabular}"""


@pytest.mark.parametrize("fraction", ["", "1.000"])
def test_full_width_artifact_table_can_split_across_pages(fraction):
    patcher = _patcher()
    source = _COUNT_RANK_ARTIFACT_TABLE.replace(
        r"\dimexpr \linewidth", rf"\dimexpr {fraction}\linewidth", 1
    )
    result = patcher.repair_generated_tables(source)
    assert r"\begin{longtable}" in result
    assert r"\begin{tabular}" not in result
    original_body = source.split("\n", 1)[1].rsplit(r"\end{tabular}", 1)[0]
    result_body = result.split("\n", 1)[1].rsplit(r"\end{longtable}", 1)[0]
    assert result_body.replace("\n" + r"\endhead", "") == original_body
    assert result.count(r"\endhead") == 1
    assert patcher.repair_generated_tables(result) == result


# This native API signature exceeds a page at the book's normal font size.
_API_OVERSIZED_SIGNATURE_ROW = r"""\texttt{cvrisk\_gamboost} & \texttt{cvrisk\_gamboost} --- \texttt{cvrisk\_gamboost(scalar\_x: Any, scalar\_names: Any, scalar\_lambda: Any, scalar\_center: Any, scalar\_df: Any, bbs\_x: Any, bbs\_names: Any, bbs\_knots: Any, bbs\_degree: Any, bbs\_differences: Any, bbs\_df: Any, bbs\_lambda: Any, bbs\_center: Any, bbs\_cyclic: Any, bbs\_boundary\_knots\_lo: Any, bbs\_boundary\_knots\_hi: Any, bbs\_deriv: Any, bmono\_x: Any, bmono\_names: Any, bmono\_knots: Any, bmono\_degree: Any, bmono\_differences: Any, bmono\_df: Any, bmono\_constraint: Any, bmono\_type: Any, bmono\_lambda2: Any, bmono\_niter: Any, bmono\_center: Any, btree\_x: Any, btree\_names: Any, btree\_minbucket: Any, btree\_minsplit: Any, btree\_maxdepth: Any, y: Any, weights: Any, mstop: Any, nu: Any, family\_name: Any, family\_params: Any, folds: Any, judge: Any = None, offset: Any = None, scalar\_by: Any = None, bbs\_by: Any = None, bmono\_by: Any = None, btree\_by: Any = None, bmrf\_levels: Any = None, bmrf\_names: Any = None, bmrf\_k: Any = None, bmrf\_df: Any = None, bmrf\_lambda: Any = None, bmrf\_center: Any = None, bmrf\_by: Any = None, bspatial\_x1: Any = None, bspatial\_x2: Any = None, bspatial\_names: Any = None, bspatial\_knots: Any = None, bspatial\_degree: Any = None, bspatial\_differences: Any = None, bspatial\_df: Any = None, bspatial\_lambda: Any = None, bspatial\_center: Any = None, bspatial\_by: Any = None, bkernel\_x: Any = None, bkernel\_names: Any = None, bkernel\_bandwidth: Any = None, bkernel\_df: Any = None, bkernel\_kernel: Any = None, bkernel\_center: Any = None, bkernel\_by: Any = None, brandom\_x: Any = None, brandom\_names: Any = None, brandom\_n\_levels: Any = None, brandom\_df: Any = None, brandom\_lambda: Any = None, brandom\_center: Any = None, brandom\_by: Any = None, bmono\_C: Any = None, buser\_X: Any = None, buser\_K: Any = None, buser\_names: Any = None, buser\_df: Any = None, buser\_lambda: Any = None, buser\_center: Any = None, buser\_by: Any = None, risk: Any = "inbag", trace: Any = False, parallel: Any = True, fold\_label: Any = "bootstrap", step: Any = "fixed", step\_iterations: Any = 4) -\textgreater  Any:} \\"""


@pytest.mark.parametrize("n_rows, environment", [(1, "tabular"), (2, "longtable")])
def test_oversized_api_signature_row_uses_a_smaller_font_without_truncation(n_rows, environment):
    patcher = _patcher()
    body = (
        "\n" + r"\toprule Name & What it is \\" + "\n" + r"\hline" + "\n"
        + (_API_OVERSIZED_SIGNATURE_ROW + "\n") * n_rows + r"\bottomrule" + "\n"
    )
    result = patcher.repair_generated_tables(_generated_table(patcher, body, 2))
    assert result.startswith("{" + r"\footnotesize" + "\n")
    assert result.count(_API_OVERSIZED_SIGNATURE_ROW) == n_rows
    assert f"\\begin{{{environment}}}" in result
    assert result.endswith("}")
