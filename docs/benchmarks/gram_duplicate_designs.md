# Gram route: duplicate designs share their Gram blocks

Issue #301. The Gaussian Gram route (`try_fit_gaussian_gram`) computed every
pair block (j, w) with a full weighted cross-Gram, including the blocks of a
learner whose design is bit-identical to an earlier learner's. For a pair of
identical dense bspatial learners the route paid three O(n·p²) passes: the
first learner's diagonal (partly offset by the #276 stashed self-Gram), the
cross block, and the second learner's diagonal.

## Change

Before the pair computation, the route now finds, for each learner w, the
first earlier learner s whose design matrix and row weights are both
bit-identical to w's (element-wise `to_bits()` comparison, with a
pointer-equality fast path). Pairs (j, w) with j < w where w is such a
duplicate are not recomputed. After the remaining pairs are computed, each
skipped block is filled from the source's blocks: for j ≤ s it is the
computed upper block `g[j][s]` itself (the shared diagonal when j = s);
for s < j < w it is the transpose of the computed upper block `g[s][j]`,
materialized in standard layout exactly as the route's lower-triangle
fill does — `g[j][s]` at that point is still the empty lower-triangle
placeholder, and copying it was the first implementation's bug (below);
the diagonal `g[w][w]` is the source's diagonal `g[s][s]`. The lower
triangle is then filled by the existing materialized transpose. The
penalty is not part of the match: two learners with the same design and
different penalties (a `bbs` and a `bmono` on one variable, for example)
are duplicates for Gram purposes and keep their own penalties.

Bit-identity is a correctness requirement: the
R-parity contract guarded by `gaussian_gram_sparse_twins_stay_tied`
depends on duplicate learners keeping exact selection ties.

The regression test,
`gaussian_gram_duplicate_designs_share_blocks_and_stay_tied`, has two
parts. The first fits two and three copies of one bspatial learner
through the forced Gram route and asserts that all selections stay on
learner 0 and that the fitted values are bit-equal (`to_bits()`) across
the pair and trio fits. The second fits two `bbs` learners with the same
design and different penalties (df 0.05 and 50.0), where the duplicate's
weaker penalty should win selections: it asserts the Gram route's
selected sequence equals the general path's and the coefficients agree
to 1e-8. The second part exists because the first implementation of the
fill copied the still-empty lower-triangle block for learners between
the source and the duplicate, leaving 0×0 blocks that the tie scenario
never touches (its duplicates are never selected) but a
different-penalty duplicate hits immediately — the Python suite's
variable-importance tests caught it as a native panic
(`ndarray: inputs 0 × 0, 24 × 1 ... are not compatible for matrix
multiplication`) on a `bbs(x1) + bols(x2) + bbs(x3) + bmono(x1)` fit.
The test fails against that first fill and passes against the corrected
one; both directions were run.

## Protocol

Release build. A temporary example harness called the profiled fit entry
point on synthetic data: n = 10,000 rows, xorshift seed 13579,
y = sin(6·x1)·cos(5·x2), unit weights, mstop = 10, nu = 0.5. Two arms: a
single bspatial learner (knots = 20, degree = 3, p = 576 basis columns) and
two identical copies of it (1,152 basis columns). Median of 15 repetitions
after 3 warm-ups. Timers are the fit report's setup, cross-Gram, and wall
times. Baseline and fixed code were measured in the same session on the same
machine.

## Results

Duplicate pair (1,152 basis columns):

| timer | before | after | change |
|---|---|---|---|
| setup | 297.43 ms | 30.48 ms | −89.7% |
| cross-Gram | 295.83 ms | 7.94 ms | −97.3% |
| wall | 317.56 ms | 49.67 ms | −84.4% |

Single learner (576 basis columns), unchanged by the fix: setup 3.78 ms
before, 4.09 ms after; cross-Gram 3.16 ms before.

All 11 `gaussian_gram` tests pass, including the twins guard test and the
new duplicate-design regression test.

## Residual

The duplicate pair's setup still exceeds the single learner's by about
26 ms. That remainder is the per-block GramMatVec CSR conversion of the
1,152-column Gram matrix plus the duplicate right-hand-side projections:
O(p²) and O(n·p) work, not the O(n·p²) pass this issue targets. It is out
of scope here.

## Provenance

The issue's figures (bspatial preparation 45 ms, diagonal pass ≈165 ms of
≈210 ms total Gram-route setup) were measured on the issue author's M2
host. The numbers above were measured in a Linux sandbox with different
hardware; absolute times are not comparable across the two machines. The
claim this record supports is the same-session before/after comparison on
one machine: the duplicate-design O(n·p²) passes are gone, and fit results
are bit-identical.
