# Python API manual

One page per public function and class exported by `mboostrs`, generated from the installed package by `tools/generate_api_manual.py`. Each page gives the live signature, the parameters, return value and errors from the docstring, and a link to the source lines. Class pages document every public method and property. The task-oriented [API reference](../appendix/api_reference.ipynb) groups the same names by purpose.

Frequently used entries: [`gamboost`](functions/gamboost.md), [`glmboost`](functions/glmboost.md), [`GAMBoostModel.predict`](classes/GAMBoostModel.md#gamboostmodel-predict), [`LSSModel.predict`](classes/LSSModel.md#lssmodel-predict), [`CvRiskResult`](classes/CvRiskResult.md), [`FoldScheme`](classes/FoldScheme.md).

## Functions

| Name | Summary |
|---|---|
| [`accumulated_local_effect`](functions/accumulated_local_effect.md) | Estimate a centered univariate accumulated local effect. |
| [`acf_residuals`](functions/acf_residuals.md) | Compute residual autocorrelation and autocorrelation of selected powers. |
| [`adaptive_conformal_stream`](functions/adaptive_conformal_stream.md) | Online Adaptive Conformal Inference (Gibbs & Candès, 2021). |
| [`agent_guide`](functions/agent_guide.md) | Return the agent skill text, a topic section, or a file path. |
| [`aipw`](functions/aipw.md) | Cross-fitted augmented inverse-probability estimate for binary treatment. |
| [`api_card`](functions/api_card.md) | A bounded card for one function, method or class. |
| [`auto_policy`](functions/auto_policy.md) | Choose a specification with measured comparisons and a workflow budget. |
| [`autospec`](functions/autospec.md) | Guess a complete boosting specification for ``data``. |
| [`average_elasticity`](functions/average_elasticity.md) | Average a finite elasticity or semi-elasticity of response prediction. |
| [`average_marginal_effect`](functions/average_marginal_effect.md) | Average row-specific finite derivatives of the full fitted prediction. |
| [`blocked_folds`](functions/blocked_folds.md) | Split rows into contiguous held-out blocks. |
| [`boost_control`](functions/boost_control.md) | The control record a fitting function takes. |
| [`build_info`](functions/build_info.md) | Return immutable compile-time identity for the loaded Rust extension. |
| [`calibration_frame`](functions/calibration_frame.md) | Compare observed and fitted means within bins. |
| [`capabilities`](functions/capabilities.md) | Return family, learner, and Python-method capability tables. |
| [`causal_trap_report`](functions/causal_trap_report.md) | Check a declared causal design against the data before estimating it. |
| [`centile_fan_data`](functions/centile_fan_data.md) | Observed responses and fitted centiles ordered by one numeric covariate. |
| [`check_declarations`](functions/check_declarations.md) | Test each declaration of a specification against the data. |
| [`ci_test`](functions/ci_test.md) | Test whether ``x`` and ``y`` are independent given ``given``. |
| [`classify_columns`](functions/classify_columns.md) | Describe each predictor and name the learner suggested for it. |
| [`cluster_residual_check`](functions/cluster_residual_check.md) | Measure the share of response-residual variance between clusters. |
| [`coef_all`](functions/coef_all.md) | Return one coefficient row per learner, boosting step and component. |
| [`collect_warnings`](functions/collect_warnings.md) | Show each repeated warning once, with a count, when the block ends. |
| [`column_screen`](functions/column_screen.md) | Scatterplot diagnostics of the response against each column. |
| [`compare_models`](functions/compare_models.md) | Compare models across multiple metrics. |
| [`compare_specifications`](functions/compare_specifications.md) | Score several right-hand sides on the same folds. |
| [`compare_summaries`](functions/compare_summaries.md) | Put several fitted models side by side as one :class:`Summary`. |
| [`compare_to_r`](functions/compare_to_r.md) | Compare a mboostrs fit against R mboost. |
| [`conformal_interval`](functions/conformal_interval.md) | Distribution-free prediction intervals for a fitted standard model. |
| [`confounding_benchmark`](functions/confounding_benchmark.md) | Benchmark sensitivity by omitting one measured covariate at a time. |
| [`confounding_sensitivity`](functions/confounding_sensitivity.md) | Bias grid for one omitted baseline confounder under an additive model. |
| [`coverage`](functions/coverage.md) | Compute empirical coverage of prediction intervals. |
| [`cv_risk`](functions/cv_risk.md) | Out-of-fold risk of a right-hand side. |
| [`cv_weights`](functions/cv_weights.md) | Generate cross-validation fold weights. |
| [`cvrisk_gamboostLSS`](functions/cvrisk_gamboostLSS.md) | Cross-validated risk for a mixed structured ``gamboostLSS`` model. |
| [`cvrisk_glmboostLSS`](functions/cvrisk_glmboostLSS.md) | Cross-validated risk for a linear GAMLSS over a grid of per-parameter |
| [`density_overlay_data`](functions/density_overlay_data.md) | Observed and average fitted response distributions on one grid. |
| [`deselect`](functions/deselect.md) | Drop base learners with a small share of the risk reduction, then refit. |
| [`deviance_increment`](functions/deviance_increment.md) | Per-observation deviance contributions. |
| [`deviance_profile`](functions/deviance_profile.md) | Bin joint deviance contributions along a numeric covariate. |
| [`diagonal_grid`](functions/diagonal_grid.md) | A grid whose points give every distribution parameter the same budget. |
| [`digest`](functions/digest.md) | Return a concise, deterministic summary card of a fitted mboostrs object. |
| [`discover`](functions/discover.md) | Run the rule-driven analysis of ``response`` on ``data``. |
| [`distributional_contrast`](functions/distributional_contrast.md) | Contrast a fitted LSS mean, quantile, or exceedance probability. |
| [`dump_result_json`](functions/dump_result_json.md) | Serialize a result dictionary or JSON data with reversible nonfinite values. |
| [`effect_model`](functions/effect_model.md) | Estimate the conditional average treatment effect ``tau(x)``. |
| [`effect_modifiers`](functions/effect_modifiers.md) | Test whether the effect changes along each covariate, using the AIPW scores. |
| [`elicit_causal_spec`](functions/elicit_causal_spec.md) | Interview a judgment layer for a causal design and validate it. |
| [`equivalence_sets`](functions/equivalence_sets.md) | Learners whose design spaces overlap at or above ``threshold``. |
| [`estimate_cost`](functions/estimate_cost.md) | Count the fits of a call and time one fit, without running the call. |
| [`evaluate_distribution`](functions/evaluate_distribution.md) | Evaluate a fitted scalar-response LSS distribution on supplied data. |
| [`evaluate_specification`](functions/evaluate_specification.md) | Evaluate one specification and retain its held-out evidence. |
| [`evaluate_survival`](functions/evaluate_survival.md) | Brier score and time-dependent AUC of a CoxPH model at fixed horizons. |
| [`expand_formula`](functions/expand_formula.md) | Resolve ``.`` and ``- term`` against the data, returning a plain formula. |
| [`expand_interactions`](functions/expand_interactions.md) | Rewrite ``a %X% b`` terms into buser references. |
| [`expand_pairs`](functions/expand_pairs.md) | Interaction terms for ``pairs``, or for every pair when none are given. |
| [`family_capabilities`](functions/family_capabilities.md) | Return family support and evidence rows for both frontends. |
| [`family_set`](functions/family_set.md) | Return the candidate LSS family names for one response support. |
| [`fit`](functions/fit.md) | Fit one model and return it with the trail of how it was chosen. |
| [`gaic_table`](functions/gaic_table.md) | Compare same-sample fits using an explicitly named stopping criterion. |
| [`gam`](functions/gam.md) | Fit a penalized GAM by REML from a formula, with no boosting. |
| [`gam_refit`](functions/gam_refit.md) | Refit a boosted model's terms as one penalized GAM with REML smoothing. |
| [`gamboost`](functions/gamboost.md) | Fit a component-wise boosted additive model. |
| [`gamboostLSS`](functions/gamboostLSS.md) | Fit a distributional additive model with linear, spline, and structured learners. |
| [`gamlss_refit`](functions/gamlss_refit.md) | Refit a distributional model's terms as one penalized likelihood with |
| [`glmboost`](functions/glmboost.md) | Fit a component-wise boosted linear model. |
| [`glmboostLSS`](functions/glmboostLSS.md) | Fit a linear GAMLSS by component-wise boosting. |
| [`grouped_folds`](functions/grouped_folds.md) | K-fold splits that keep each group entirely on one side. |
| [`heteroscedasticity`](functions/heteroscedasticity.md) | Measure how residual spread changes across fitted values. |
| [`holdout`](functions/holdout.md) | Split a frame into a search part and a test part. |
| [`imputation_diagnostics`](functions/imputation_diagnostics.md) | Report the R-squared of each model-based predictor fill, by column. |
| [`influence_check`](functions/influence_check.md) | Use leverage and deletion refits to assess changes associated with individual rows. |
| [`insights`](functions/insights.md) | Run screen -> fit -> diagnose -> propose; return ranked findings. |
| [`install_skill`](functions/install_skill.md) | Copy the skill, its topics and examples to ``dest/analyzing-data-with-mboostrs/``. |
| [`interval_metrics`](functions/interval_metrics.md) | Weighted coverage, width, and proper central-interval score. |
| [`ipc_weights`](functions/ipc_weights.md) | Inverse probability of censoring weights (mboost ``IPCweights``). |
| [`iv_effect`](functions/iv_effect.md) | Cross-fitted partially linear instrumental-variable estimate for a binary treatment. |
| [`known_group_folds`](functions/known_group_folds.md) | K-fold splits in which every held-out row has a group-mate in training. |
| [`label_evidence`](functions/label_evidence.md) | Read a whole-number column against the response; see `LabelEvidence`. |
| [`learner_capabilities`](functions/learner_capabilities.md) | Return base-learner support and evidence rows for both frontends. |
| [`learner_design`](functions/learner_design.md) | Build a reusable design and penalty for one base-learner term. |
| [`llm_describe`](functions/llm_describe.md) | Send the model's description to a language model and return the reply. |
| [`llm_explain_prompt`](functions/llm_explain_prompt.md) | Render a fitted model as a prompt for a language model. |
| [`load_model`](functions/load_model.md) | Read a model artifact written by :func:`save_model`. |
| [`load_result_json`](functions/load_result_json.md) | Decode a :func:`dump_result_json` document into plain Python result data. |
| [`loo_folds`](functions/loo_folds.md) | Leave-one-out: fold ``i`` holds out row ``i`` alone (``n`` folds). |
| [`make_grid`](functions/make_grid.md) | Build a per-parameter ``mstop`` grid for distributional cross-validation. |
| [`make_grid_lss`](functions/make_grid_lss.md) | Per-parameter 1-D mstop grid, mirroring gamboostLSS::make.grid's marginal |
| [`method_capabilities`](functions/method_capabilities.md) | Return public method support for ``frontend='python'`` or ``'r'``. |
| [`modeler_loop`](functions/modeler_loop.md) | Run a measured specification search with optional LLM proposals. |
| [`negative_control`](functions/negative_control.md) | Estimate a prespecified negative-control exposure or outcome effect. |
| [`open_search`](functions/open_search.md) | Open a training-only session for human or external-agent tool choice. |
| [`outlier_frame`](functions/outlier_frame.md) | Observations ranked by absolute residual, largest first. |
| [`overlap_report`](functions/overlap_report.md) | Summarize where the estimated propensity leaves the overlap bounds. |
| [`pair_term_for`](functions/pair_term_for.md) | The learner that carries an interaction between two columns. |
| [`parse_formula`](functions/parse_formula.md) | Split a linear formula into its response and predictor names. |
| [`parse_gamboost_formula`](functions/parse_gamboost_formula.md) | Parse an additive formula into its response and learner objects. |
| [`pit`](functions/pit.md) | Probability integral transform for LSS models with CDF support. |
| [`predict_all`](functions/predict_all.md) | Return a table of fitted values (and response for in-sample). |
| [`prediction_contrast`](functions/prediction_contrast.md) | Average a full-model prediction change over the supplied rows. |
| [`prediction_profile`](functions/prediction_profile.md) | Weighted partial dependence and optional individual conditional curves. |
| [`probe`](functions/probe.md) | Select terms by the first entry of a permuted shadow, then refit them. |
| [`profile_mstop`](functions/profile_mstop.md) | Evaluate a model at different mstop values. |
| [`provenance`](functions/provenance.md) | Return build, environment and optional data identity metadata for a result. |
| [`q_stats`](functions/q_stats.md) | Binned quantile-residual moment summaries. |
| [`quantile_boost`](functions/quantile_boost.md) | Simultaneously CV-tune and fit multiple conditional quantile models in one Rust pass. |
| [`quantile_residuals`](functions/quantile_residuals.md) | Randomized quantile residuals. |
| [`r_squared`](functions/r_squared.md) | Generalized R-squared for a distributional model, as ``gamlss::Rsq``. |
| [`rank_by_effect`](functions/rank_by_effect.md) | Rank new rows by estimated effect, using only rows with overlap. |
| [`rearrange_quantiles`](functions/rearrange_quantiles.md) | Predict several quantile or expectile fits and sort each row across levels. |
| [`reference_gap`](functions/reference_gap.md) | Compare a specification with a tree reference and rank candidate revisions. |
| [`repeated_kfold_folds`](functions/repeated_kfold_folds.md) | ``repeats`` independent k-fold partitions, side by side. |
| [`report`](functions/report.md) | Render a ``Fit`` (or ``DecisionTrail``) as text, markdown or JSON. |
| [`reproduce`](functions/reproduce.md) | Reproduce a recorded :func:`fit` and its discovery report in one call. |
| [`residual_frame`](functions/residual_frame.md) | Diagnostic residual table for any model. |
| [`revision_checks`](functions/revision_checks.md) | Run the cluster, time and influence checks and report their revisions. |
| [`rolling_folds`](functions/rolling_folds.md) | Rolling-origin splits: train on the past, judge on what comes next. |
| [`same_family_gap`](functions/same_family_gap.md) | Compare a linear boost at its cross-validated step with the same-family GLM. |
| [`save_model`](functions/save_model.md) | Write `model` to `path` as a schema-versioned JSON artifact. |
| [`scale_absorption`](functions/scale_absorption.md) | Test whether a fitted LSS scale effect survives a more flexible mean. |
| [`scan_lss_families`](functions/scan_lss_families.md) | Scan candidate LSS families and rank by a metric. |
| [`screen_discontinuities`](functions/screen_discontinuities.md) | Screen for axis-aligned and diagonal step discontinuities. |
| [`screen_interactions`](functions/screen_interactions.md) | Rank candidate pairs by the residual variation they explain out of sample. |
| [`select_capacity`](functions/select_capacity.md) | Extend a formula with interaction capacity only where it pays. |
| [`selection_report`](functions/selection_report.md) | Equivalence sets, substitution sets and deselection, read together. |
| [`sensitivity`](functions/sensitivity.md) | How much the fitted function moves across each variable's range. |
| [`specification_curve`](functions/specification_curve.md) | Score the specifications a search touched on one shared fold partition. |
| [`split_refit`](functions/split_refit.md) | Choose terms on one random half of the rows and estimate them by REML on |
| [`stabsel`](functions/stabsel.md) | Stability selection for mboostrs models. |
| [`stabsel_gamboostLSS`](functions/stabsel_gamboostLSS.md) | Stability selection for a mixed structured ``gamboostLSS`` model. |
| [`stabsel_glmboostLSS`](functions/stabsel_glmboostLSS.md) | Stability selection for a linear GAMLSS (R's stabsel.mboostLSS). |
| [`stratified_folds`](functions/stratified_folds.md) | K-fold splits that preserve the class proportions of ``y``. |
| [`substitution_sets`](functions/substitution_sets.md) | Learners that stability selection picks in turn rather than together. |
| [`suggest_specifications`](functions/suggest_specifications.md) | Rank candidate changes by their expected held-out gain. |
| [`suggest_terms`](functions/suggest_terms.md) | The main-effect right-hand side suggested for ``data``. |
| [`term_report`](functions/term_report.md) | Refit a specification without each term, and each term in other shapes. |
| [`time_residual_check`](functions/time_residual_check.md) | Test whether residuals are correlated across ordered time points. |
| [`tune_mstop`](functions/tune_mstop.md) | Cross-validate the step count, doubling the grid while the risk still falls. |
| [`validate`](functions/validate.md) | The issues that ``predict(newdata)`` would meet, before it runs. |
| [`validation_capabilities`](functions/validation_capabilities.md) | Return the explicit validation ledger behind advertised capabilities. |
| [`weighted_median`](functions/weighted_median.md) | The weighted median, matching ``gamboostLSS:::weighted.median``. |
| [`worm_data`](functions/worm_data.md) | Return worm-plot deviation data. |

## Classes

| Name | Summary |
|---|---|
| [`AIPWResult`](classes/AIPWResult.md) | Cross-fitted AIPW estimate with reconstructible row contributions. |
| [`ALE1DResult`](classes/ALE1DResult.md) | A centered univariate accumulated-local-effect curve and its support. |
| [`AUC`](classes/AUC.md) | Smoothed 1 - AUC ranking loss (mboost::AUC()). |
| [`AdaExp`](classes/AdaExp.md) | Exponential loss, ``exp(-(2y - 1) f)`` — the AdaBoost loss. |
| [`AdaptiveConformalStream`](classes/AdaptiveConformalStream.md) | Online Adaptive Conformal Inference stream (Gibbs & Candès, 2021). |
| [`AicResult`](classes/AicResult.md) | An information-criterion path: the value at each iteration, the |
| [`Assumption`](classes/Assumption.md) | One causal statement with its provenance, never inferred from data. |
| [`AssumptionLedger`](classes/AssumptionLedger.md) | The causal statements a run depends on, tagged by who vouched for them. |
| [`BCCGLSS`](classes/BCCGLSS.md) | Box-Cox-Cole-Green family: ``mu`` (log link, median), ``sigma`` |
| [`BCPELSS`](classes/BCPELSS.md) | Box-Cox power-exponential family with four fitted parameters. |
| [`BCTLSS`](classes/BCTLSS.md) | Box-Cox t family with ``mu``, ``sigma``, ``nu``, and ``tau``. |
| [`BLPProjection`](classes/BLPProjection.md) | A linear projection of the effect on covariates, usable as a ranking source. |
| [`Bbs`](classes/Bbs.md) | Penalized B-spline (P-spline) base-learner (R `bbs`). |
| [`Bconstrained`](classes/Bconstrained.md) | Constrained P-spline base-learner (R `bconstrained`). |
| [`BetaLSS`](classes/BetaLSS.md) | Beta location-precision family: ``mu`` (logit link, mean in (0,1)), |
| [`Binomial`](classes/Binomial.md) | Binomial log-likelihood with a logit link, for a two-class response. |
| [`BinomialLSS`](classes/BinomialLSS.md) | Binomial location family: ``mu`` (logit link, probability). |
| [`Bkernel`](classes/Bkernel.md) | Kernel base-learner (R `bkernel`). |
| [`Bmono`](classes/Bmono.md) | Monotone-constrained P-spline learner (R `bmono`). |
| [`Bmrf`](classes/Bmrf.md) | Markov Random Field base-learner (R `bmrf`). |
| [`Bns`](classes/Bns.md) | Ridge-penalized B-spline base-learner (mboostrs extension). |
| [`Bols`](classes/Bols.md) | Ordinary least-squares base-learner (R `bols`). |
| [`BoostControl`](classes/BoostControl.md) | How long to boost, and how far each step goes. |
| [`Brad`](classes/Brad.md) | Radial-basis base-learner (R `brad`). |
| [`Brandom`](classes/Brandom.md) | Random-effects base-learner (R `brandom`). |
| [`Bspatial`](classes/Bspatial.md) | Tensor-product P-spline base-learner (R `bspatial`). |
| [`Btree`](classes/Btree.md) | Tree base-learner (R `btree`). Fits decision stumps by default. |
| [`Buser`](classes/Buser.md) | User-supplied base-learner (R `buser`). |
| [`CITest`](classes/CITest.md) | A test of ``x`` independent of ``y`` given the columns in ``given``. |
| [`CapacitySelection`](classes/CapacitySelection.md) | The capacity ladder :func:`select_capacity` measured, and what it kept. |
| [`CausalDraft`](classes/CausalDraft.md) | A proposed causal design: a spec a person has not yet confirmed. |
| [`CausalSpec`](classes/CausalSpec.md) | A binary-treatment causal question and its adjustment design. |
| [`Cindex`](classes/Cindex.md) | Uno's IPCW-weighted concordance ("C-index") loss (mboost::Cindex()). |
| [`ColumnKind`](classes/ColumnKind.md) | What a column looks like, the learner that follows, and what else it |
| [`CompareReport`](classes/CompareReport.md) | Structured comparison report between Python and R. |
| [`CompareSpec`](classes/CompareSpec.md) | Specification for a single model comparison. |
| [`ConformalInterval`](classes/ConformalInterval.md) | Prediction intervals with their calibration evidence attached. |
| [`ConformalReliabilityWarning`](classes/ConformalReliabilityWarning.md) | Warning emitted when conformal reliability diagnostics detect risk to coverage. |
| [`ContrastResult`](classes/ContrastResult.md) | A standardized full-model prediction contrast and its row contributions. |
| [`CostBudget`](classes/CostBudget.md) | Limits on a procedure's fit count, run time and memory. ``None`` means no limit on that field. |
| [`CoxPH`](classes/CoxPH.md) | Cox partial likelihood for right-censored survival data. |
| [`CvReport`](classes/CvReport.md) | Summary of a cross-validation run for standard and distributional models. |
| [`CvRiskResult`](classes/CvRiskResult.md) | Cross-validated risk along the boosting path. |
| [`DecisionTrail`](classes/DecisionTrail.md) | A builder for the record in section 5 of the design note. |
| [`DeclarationCheck`](classes/DeclarationCheck.md) | The test of one declaration against the fit. |
| [`Deselection`](classes/Deselection.md) | The terms :func:`deselect` kept and dropped, and the refitted model. |
| [`DirichletLSS`](classes/DirichletLSS.md) | Dirichlet family for compositional responses: ``K`` concentration |
| [`DiscontinuityHit`](classes/DiscontinuityHit.md) | One evaluated boundary from :func:`screen_discontinuities`. |
| [`DiscoveryResult`](classes/DiscoveryResult.md) | What :func:`discover` found, with the decisions that produced it. |
| [`DistributionEvaluationResult`](classes/DistributionEvaluationResult.md) | Row-level distribution scores and aggregate predictive diagnostics. |
| [`EffectContrast`](classes/EffectContrast.md) | The average change in a linear predictor when covariates move, with its interval. |
| [`EffectModel`](classes/EffectModel.md) | A conditional average treatment effect ``tau(x)``, fitted by the |
| [`EffectModifiers`](classes/EffectModifiers.md) | Per-covariate tests of effect modification, with a joint projection. |
| [`ElasticityResult`](classes/ElasticityResult.md) | Average and row-specific finite log-response derivatives. |
| [`EquivalenceSets`](classes/EquivalenceSets.md) | Groups of learners whose design spaces overlap at or above a threshold. |
| [`EvaluationResult`](classes/EvaluationResult.md) | Held-out predictions, scoring semantics and provenance for one specification. |
| [`ExecutionReport`](classes/ExecutionReport.md) | Versioned summary of the numerical route used by a fitted model. |
| [`Expectile`](classes/Expectile.md) | Asymmetric squared error loss for expectile regression. |
| [`Finding`](classes/Finding.md) | One thing the discovery pipeline found, and what to do about it. |
| [`Fit`](classes/Fit.md) | A fitted model wrapped with the trail that produced it. |
| [`FitCache`](classes/FitCache.md) | Memo for evaluated specifications within a Python process. |
| [`FoldScheme`](classes/FoldScheme.md) | An ``(n, B)`` in-bag weight matrix, and what it means. |
| [`GAMBoostModel`](classes/GAMBoostModel.md) | Fitted generalized additive boosting model. |
| [`GLMBoostModel`](classes/GLMBoostModel.md) | Fitted gradient boosting model. |
| [`GamRefit`](classes/GamRefit.md) | A jointly penalized refit of a set of terms, with Bayesian intervals. |
| [`GamlssRefit`](classes/GamlssRefit.md) | A penalized refit of a distributional model's terms, all parameters |
| [`GammaLSS`](classes/GammaLSS.md) | Gamma location-shape family: ``mu`` (log link, mean), ``sigma`` (log link, |
| [`GammaReg`](classes/GammaReg.md) | Gamma log-likelihood with a log link, for a positive continuous response. |
| [`GaussReg`](classes/GaussReg.md) | Squared error loss with an identity link, without R's `mboost` equivalent. |
| [`Gaussian`](classes/Gaussian.md) | Squared error loss, ``0.5 * (y - f)**2``, with an identity link. |
| [`GaussianKnownVarLSS`](classes/GaussianKnownVarLSS.md) | Normal location-scale family for outcomes that carry a known sampling |
| [`GaussianLSS`](classes/GaussianLSS.md) | Normal location-scale family: ``mu`` (identity link), ``sigma`` (log link). |
| [`Gehan`](classes/Gehan.md) | Gehan rank loss for AFT survival models (mboost::Gehan()). |
| [`GeometricLSS`](classes/GeometricLSS.md) | :math:`\text{Geom}(\pi)` geometric location family: ``mu`` (log link, |
| [`GumbelLSS`](classes/GumbelLSS.md) | Gumbel location-scale family for maxima — the standard extreme-value |
| [`Hinge`](classes/Hinge.md) | Adaptive hinge base-learner that estimates a piecewise-linear effect. |
| [`Hinge`](classes/Hinge.md) | Adaptive hinge base-learner that estimates a piecewise-linear effect. |
| [`HingeLoss`](classes/HingeLoss.md) | Hinge loss, ``max(0, 1 - y f)`` — the support vector loss. |
| [`Huber`](classes/Huber.md) | Huber's robust loss: quadratic near zero, linear in the tails. |
| [`Hurdle`](classes/Hurdle.md) | The count part of a hurdle model: a zero-truncated negative binomial. |
| [`IVResult`](classes/IVResult.md) | A cross-fitted instrumental-variable estimate, its interval and its first-stage diagnostics. |
| [`InfluenceFinding`](classes/InfluenceFinding.md) | The influence check, with the per-row measures and exact deletion refits. |
| [`InverseGaussianLSS`](classes/InverseGaussianLSS.md) | Inverse Gaussian family: ``mu`` (log link, mean), ``sigma`` (log link, |
| [`JSULSS`](classes/JSULSS.md) | Johnson's SU family: ``mu`` (identity link, location), ``sigma`` |
| [`LLMJudge`](classes/LLMJudge.md) | A judge backed by any OpenAI-compatible chat endpoint. |
| [`LSSModel`](classes/LSSModel.md) | Fitted gamboostLSS/glmboostLSS model (one predictor per parameter). |
| [`LabelEvidence`](classes/LabelEvidence.md) | Whether a whole-number column's values act as labels or as a measurement. |
| [`Laplace`](classes/Laplace.md) | Absolute error loss, ``\|y - f\|``, with an identity link. |
| [`LearnerDesign`](classes/LearnerDesign.md) | A fitted design/penalty recipe for ``buser`` and inspection. |
| [`LogLogLSS`](classes/LogLogLSS.md) | Log-logistic AFT family. Takes a right-censored ``Surv(time, status)`` |
| [`LogNormalLSS`](classes/LogNormalLSS.md) | Log-normal AFT family. Matches R ``gamboostLSS::LogNormalLSS()`` for a |
| [`LogisticLSS`](classes/LogisticLSS.md) | Logistic family: ``mu`` (identity link, location), ``sigma`` (log link, |
| [`Loglog`](classes/Loglog.md) | Log-logistic accelerated-failure-time model for right-censored data. |
| [`Lognormal`](classes/Lognormal.md) | Log-normal accelerated-failure-time model for right-censored data. |
| [`LssCvRiskResult`](classes/LssCvRiskResult.md) | Cross-validated LSS risk over a per-parameter mstop grid. |
| [`LssStabSelResult`](classes/LssStabSelResult.md) | Stability-selection result for combined LSS learner/parameter effects. |
| [`MarginalEffectResult`](classes/MarginalEffectResult.md) | Average and row-specific finite derivatives of the full prediction. |
| [`MboostrsError`](classes/MboostrsError.md) | Base class for every mboostrs error, carrying error attributes. |
| [`MboostrsKeyError`](classes/MboostrsKeyError.md) | A missing map key, such as a column the model needs but the data lacks. |
| [`MboostrsRuntimeError`](classes/MboostrsRuntimeError.md) | A numerical or solver failure in the Rust engine. |
| [`MboostrsValueError`](classes/MboostrsValueError.md) | A failed validation: a bad argument, shape, parameter, or response. |
| [`ModelerResult`](classes/ModelerResult.md) | Model and specification selected by the loop, with the decision record. |
| [`MstopTuning`](classes/MstopTuning.md) | A cross-validated step count, and whether the grid grew to find it. |
| [`MultiQuantileModel`](classes/MultiQuantileModel.md) | Simultaneously fitted conditional quantile models sharing one learner basis. |
| [`Multinomial`](classes/Multinomial.md) | Multinomial logistic likelihood for an unordered K-class response. |
| [`NBinomial`](classes/NBinomial.md) | Negative binomial log-likelihood with a log link, for overdispersed counts. |
| [`NBinomialLSS`](classes/NBinomialLSS.md) | Negative-binomial family: ``mu`` (log link, mean), ``sigma`` (log link, |
| [`NegativeBinomialLSS`](classes/NegativeBinomialLSS.md) | Negative-binomial location-scale family: ``mu`` (log link, mean), ``sigma`` |
| [`NegativeControlResult`](classes/NegativeControlResult.md) | A prespecified negative-control estimate and its expected null value. |
| [`OrderedCategoryWarning`](classes/OrderedCategoryWarning.md) | An ordered categorical predictor was fitted as unordered groups. |
| [`OverlapRanking`](classes/OverlapRanking.md) | The top group of rows by estimated effect, ranked inside the overlap region. |
| [`OverlapReport`](classes/OverlapReport.md) | Rows whose estimated propensity lies outside ``bounds``, where they |
| [`PIGLSS`](classes/PIGLSS.md) | Poisson-inverse-Gaussian family: ``mu`` (log link, mean), ``sigma`` |
| [`PairedTest`](classes/PairedTest.md) | One rung's decision: the paired fold comparison behind an adoption. |
| [`Pareto2LSS`](classes/Pareto2LSS.md) | Pareto type-2 (Lomax) family: ``mu`` (log link, scale), ``sigma`` (log |
| [`PathRow`](classes/PathRow.md) | One learner update in a fitted boosting path. |
| [`Poisson`](classes/Poisson.md) | Poisson log-likelihood with a log link, for counts. |
| [`PoissonLSS`](classes/PoissonLSS.md) | :math:`P(\mu)` Poisson location family: ``mu`` (log link, mean). |
| [`PolicyResult`](classes/PolicyResult.md) | Specification selected by the policy, with evidence for each step. |
| [`PredictionProfileResult`](classes/PredictionProfileResult.md) | A weighted partial-dependence curve with optional ICE predictions. |
| [`Probing`](classes/Probing.md) | Result of :func:`probe`: selected terms, the first shadow, and its entry iteration. |
| [`PropOdds`](classes/PropOdds.md) | Proportional-odds cumulative model for an ordered categorical response. |
| [`Quantile`](classes/Quantile.md) | Check (pinball) loss for quantile regression. |
| [`RearrangedQuantiles`](classes/RearrangedQuantiles.md) | Sorted predictions at several levels and the crossings found before sorting. |
| [`ReferenceGapResult`](classes/ReferenceGapResult.md) | Result of comparing a specification with a tree reference, including candidate revisions. |
| [`ReproductionResult`](classes/ReproductionResult.md) | Outcome of replaying a saved :class:`DecisionTrail` on a dataset. |
| [`RevisionFinding`](classes/RevisionFinding.md) | One residual check: what it found, the evidence, and the revision. |
| [`RevisionReport`](classes/RevisionReport.md) | The three checks run together, in the order cluster, time, influence. |
| [`SameFamilyGap`](classes/SameFamilyGap.md) | Log-likelihood gap between a boost and the same-family GLM. |
| [`ScaleAbsorption`](classes/ScaleAbsorption.md) | Result of :func:`scale_absorption`. |
| [`ScreenedPair`](classes/ScreenedPair.md) | One candidate interaction and the mean squared error it removed. |
| [`SearchEvidence`](classes/SearchEvidence.md) | One action's result and its position in the search's audit trail. |
| [`SearchSession`](classes/SearchSession.md) | A persistent training-only search whose next diagnostic is caller-chosen. |
| [`SelectionReport`](classes/SelectionReport.md) | Equivalence sets, substitution sets, and the deselected learners they explain. |
| [`SpecificationCurve`](classes/SpecificationCurve.md) | The candidate specifications of a search, scored on one fold partition. |
| [`SpecificationJudgment`](classes/SpecificationJudgment.md) | Validated editable proposal; predictive evidence decides adoption. |
| [`Spent`](classes/Spent.md) | Fit, time and memory charges recorded during a procedure. |
| [`SplitRefitResult`](classes/SplitRefitResult.md) | The output of :func:`split_refit`: the aggregate, and the per-repeat records. |
| [`StabSelResult`](classes/StabSelResult.md) | Stability-selection frequencies, selected learners, and error bound. |
| [`StoppingBudget`](classes/StoppingBudget.md) | How far a fitted model was boosted, in the same shape for every model. |
| [`StudentTLSS`](classes/StudentTLSS.md) | Student's t family: ``mu`` (identity link), ``sigma`` (log link, scale), |
| [`SubstitutionSets`](classes/SubstitutionSets.md) | Substitute pairs and the groups they form, from a stability selection run. |
| [`Suggestion`](classes/Suggestion.md) | One change worth trying, and the evidence for it. |
| [`Summary`](classes/Summary.md) | A model summary that prints as text and displays as HTML. |
| [`SummaryTable`](classes/SummaryTable.md) | A titled table of rows, rendered as text or HTML. |
| [`SurvivalEvaluationResult`](classes/SurvivalEvaluationResult.md) | Fixed-horizon Brier score and time-dependent AUC of a CoxPH model. |
| [`TermReport`](classes/TermReport.md) | Held-out contribution and alternative-shape scores for each term. |
| [`Transcript`](classes/Transcript.md) | One recorded step per line of the search, in order. |
| [`TrapFinding`](classes/TrapFinding.md) | One design problem a causal claim would rest on. |
| [`TrapReport`](classes/TrapReport.md) | Findings from checking a declared causal design against the supplied data. |
| [`Weibull`](classes/Weibull.md) | Weibull accelerated-failure-time model for right-censored data. |
| [`WeibullLSS`](classes/WeibullLSS.md) | Weibull AFT family. Takes a right-censored ``Surv(time, status)`` |
| [`ZINBLSS`](classes/ZINBLSS.md) | Zero-inflated negative-binomial (type I) family: ``mu`` (log link, mean), |
| [`ZIPoLSS`](classes/ZIPoLSS.md) | Zero-inflated Poisson family: ``mu`` (log link, Poisson mean), ``sigma`` |
| [`ZeroAdjustedGammaLSS`](classes/ZeroAdjustedGammaLSS.md) | Zero-adjusted Gamma: ``mu`` (log, mean of the positive part), ``sigma`` |
| [`ZeroAdjustedPoissonLSS`](classes/ZeroAdjustedPoissonLSS.md) | Zero-adjusted Poisson family: ``mu`` (log link, Poisson mean), ``sigma`` |
