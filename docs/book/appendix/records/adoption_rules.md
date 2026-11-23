(record-adoption-rules)=
# Specification-search adoption gates and budget arbitration

This record summarizes the empirical evaluations of adoption rules inside `autospec` and `discover`, supporting [](../../benchmarks/pair_form_stopping.ipynb) and [](../../benchmarks/all_column_boundary.ipynb).

## Single-holdout adoption gates

When `autospec` receives a single supplied holdout rather than multi-fold cross-validation, the fold-difference vector has length one and the paired fold standard error is undefined ($z = 0$ or `NaN`). Under a bare 2% aggregate risk margin without per-row uncertainty checks, selection on reused holdout rows admits noise terms through winner's-curse bias:

- **Residual tree, shape probe, and pair-form picker:** On the three-split breast-cancer single-holdout battery (273 train, 91 validation, 114 test; budget 32), bare-margin adoption produced mean test log loss 0.2049 (driven by draw 2 validating at 0.0322 and testing at 0.4219). Requiring a paired per-row test ($z \ge 2.0$ computed on per-row held-out losses of training-refitted models) reduced mean test log loss to 0.1179.
- **Coordinate-surface step:** On 40 additive Gaussian draws ($n = 364$, 91 holdout rows) whose two coordinate columns were pure noise, the unguarded coordinate-surface step adopted a spatial or tree term in 4 of 40 draws (margins +2.14% to +6.16%, fold $z = 0$). Evaluated on 20,000 fresh test rows, 3 of the 4 adoptions significantly worsened out-of-sample MSE (per-row $z = -3.61, -2.68, -10.74$). Applying the same paired per-row $z \ge 2.0$ requirement eliminated those false adoptions.
- **Distributional (`LSS`) adoption:** The distributional validators (`_validate_lss` and `compare_finished_routes`) previously waived the $2\text{ SE}$ requirement whenever `_paired_se` returned `NaN` on a single fold, letting flexible LSS families adopt on a bare 2% margin. Replacing the waiver with an explicit refusal when standard errors are non-finite prevents single-holdout distributional overfitting.

A constructive audit across 28 decision points in the automatic search also tested whether two remaining sites (`_held_out_pairs` and `_joint_tree_step`) failed at $\ge 4/40$ draws on additive Gaussian panels ($n = 364$); neither triggered the preset failure threshold, so no separate judging layer was built.

## Residual-tree adoption policy

An evaluation of enabling `residual_tree="auto"` by default on California housing (five random splits) rejected automatic promotion: because the coordinate-surface step already adopts `btree(longitude, latitude, maxdepth=3)`, the default route reached a mean MSE ratio of 1.057 against the best tree library without a residual tree. Enabling the residual tree caused it to be adopted on all five seeds under spatial-cell grouped judging folds (+11% to +18% judged gain), yet random-row test MSE worsened on 4 of 5 seeds because spatial and random-row generalization disagreed.

## Joint-tree gate, fit reservation, and budget arbitration

- **Coordinate-name gate vs. open availability:** On non-coordinate panels (`protein`, `bike_iid`, `diamonds`), restricting the final joint-tree step to datasets with recognized coordinate column names prevented it from running. Even after lifting that gate in diagnostic runs, `bike_iid` starved the joint-tree step at small budgets (budget 32), because earlier stages spent more than 23 fits while the joint-tree step required $3 \times (\text{folds} + 2) = 9$ fits.
- **Fit reservation and coordinate-surface arbitration:** Reserving fits early for the joint-tree step solved starvation on low-budget runs (e.g., budget 40 on coordinate data: signal MSE 0.435 with reservation vs. 9.075 without), but at intermediate budgets (60–70) the reservation starved the coordinate-surface step (cost $6 \times \text{folds} + 2$), raising fresh-sample MSE on an additive-plus-spatial fixture from 0.112 to 0.478 (signal MSE 0.070 vs. 0.435). Arbitrating the reservation—releasing reserved fits to the coordinate-surface step whenever the unreserved remainder cannot pay for it but the combined budget can—preserves both the low-budget joint-tree recovery and the mid-budget spatial-plus-additive fit.

## Whole-model check against simple baselines

At the end of a positive-budget search, `autospec` scores the final searched specification against two prespecified simple models on the shared folds: an all-linear model (`bols` for every univariate smooth) and the starting additive specification, replacing the searched model only when a simple rival is better by more than $1$ paired fold SE (never on a single fold). On multi-fold designs this prevents the search from finishing behind a simpler baseline; on single-holdout designs the check records the diagnostic scores without triggering a replacement.

**Audit artifacts.** Full per-draw logs, constructive-test protocols, and validation summaries are archived under `docs/book/benchmarks/` (`residual_tree_policy.md`, `single_holdout_adoption_gates.md`, `coordinate_surface_single_holdout.md`, `lss_single_holdout_se.md`, `joint_tree_gate.md`, `joint_tree_reservation_eval.md`, `joint_tree_reservation_arbitration.md`, `judging_constructive_test.md`, `whole_model_eval.md`, `all_column_boundary/VALIDATION.md`, and `pair_form_stopping/VALIDATION.md`).
