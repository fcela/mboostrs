# Empty prediction sets

Instrument clarification `nonmonotone-spread/1.1`; the nomination rule, processes,
seeds, partitions and every estimator setting remain those of the precommitted
protocol. This addendum is committed before the corrected scoring run.

The initial scoring code asserted lower <= upper for every test row. CQR can
produce an empty set when its calibration correction shrinks a narrow interval
past zero. This is a valid possible output of that method, not a fitting failure.
The original instrument incorrectly rejected such outcomes; its complete initial
records and runner are retained. No failure is removed from those records.

For the corrected run, coverage uses lower <= y <= upper (false for empty sets),
and width is max(upper-lower, 0), the length of the prediction set. Record the
number of empty sets and maximum endpoint crossing on every draw. The reported
interval score is the sum of the two scaled pinball losses,
(upper-lower) + 20*max(lower-y, 0) + 20*max(y-upper, 0). It equals the usual 90%
interval score for ordered bounds and remains a proper two-quantile score for
crossed bounds. It uses the signed endpoint difference in its first term;
substituting zero there would change the scoring rule.

Rerun every method and both mboostrs versions on the same original fresh seeds,
not merely the formerly rejected rows. Keep the initial and final phases
separate. This repairs the measurement instrument; it does not revise a model
rule after looking at test performance or create an independent confirmation
batch. Both initial attempts and corrected measurements remain visible.
