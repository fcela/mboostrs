# Ten-column boundary representation

Instrument all-column-boundary/1, frozen before measurement on main5c68ea5.
Retest the actual task in #608 rather than extrapolating a five-column supplied
pair result. No learner engine or default changes in this study. A btree term
already fits successive gradient trees. This study compares predictor
representations and selection workflows for adequate prediction.

Generate2000 rows with10 independent uniform[-1,1] columns x0..x9,
Y=3*sign(x0*x1)+x2+Normal(0,.5^2). Use the original70/30 split with
sklearn train_test_split random_state0 and retain actual row identities.
Development seeds6087100–6087109, confirmation6088100–6088149. Each complete
draw is independent; no test result selects a setting or method. Standardize
RMSE by the test-response population SD (ddof0), as the original task did.
Also report response and latent-mean MSE, all predictions, source/native
identities, model choices, costs and failures. No successful-only averages.

Methods are fixed now:
- original hybrid:all10 default bbs terms plus btree(all10,maxdepth6),500
  iterations,nu.1; additive-only:the same smooths without the tree;
- supplied pair:btree(x0,x1,maxdepth3,selection=sse)+bols(x2),500 iterations;
- supplied pair with all10 default bbs main terms,500 iterations;
- supplied tuned pair:the same supplied pair+linear formula, five-fold
  tune_mstop at nu.1,start100,max500,seed0. Its family is Gaussian;
- data-only:ordinary discover on all10 observed columns,seed0, no supplied
  family/forms, default400 cap. Extract public mu if it adopts GaussianLSS;
- original fixed competitors:CatBoost200trees,depth6,rate.05,random_seed0;
  LightGBM200trees,max_depth-1,num_leaves31,rate.05,random_state0. One thread
  per method and no competitor tuning.

CatBoost is the prespecified primary comparator. Retain LightGBM separately,
with a Bonferroni-adjusted two-comparator upper bound reported in addition.
The target is <=1% mean paired NRMSE excess, with the point criterion and
one-sided95% Student-t upper-limit criterion distinguished. A target claim
for the supplied information track uses the SUPPLIED TUNED method fixed above;
the data-only track uses DISCOVER fixed above. No confirmation winner among
the diagnostic representations becomes the target method after measurement.
Preserve every failed method/draw and the original hybrid even if it loses.

Report development before confirmation, then freeze runner/source identity
before reading any confirmation draw. These results can support a bounded
task result and correct the obsolete ensemble explanation. They cannot show
that the original all-column formula itself improved, establish universal
tree capacity, or resolve Gaussian mean-versus-scale attribution (#629).
