# Line-search instrumentation

Version2 incorrectly supplied step= to compare_specifications, whose signature
does not accept it. Every applicable line-search comparison failed before
fitting. Preserve all240version2records and their tracebacks. The120screen400
records are valid; incomplete line-search results cannot be averaged only over
successful inapplicable Gaussian cases. No line-search performance conclusion
is drawn from those failures.

Version3 leaves compare_specifications unchanged and supplies the existing
step= argument to its underlying evaluate_specification, which forwards it to
the public core-backed cv_risk evaluator. It also applies the same step to mean
tuning and final fits. Record every actual evaluator/fit control. The declared
algorithm, rates, caps, inputs and acceptance criteria are unchanged. Run only
the repaired line-search variant, on the same DEVELOPMENT inputs, in separate
pilot_v3 files. No historical file is overwritten. Reserved confirmation seeds
remain untouched. A production version would explicitly plumb the argument;
these global wrappers are isolated research instrumentation.
