<!-- Written by the orchestrator from the agent's returned text: the agent did not write REPORT.md. -->
Final model: statsmodels QuantReg at q = 0.1, 0.5, 0.9 on log1p(pm2.5), back-transformed with expm1 and clipped at 0. Basis: patsy bs(TEMP, DEWP, PRES, log_Iws), df=4, cubic, exterior knots fixed from covariate ranges of the train and test feature files. Also C(cbwd), C(hour), C(month), log_Is, log_Ir, and winter-interaction terms bs(...):winter and C(cbwd):winter. 75 columns after pruning aliased columns.

Tuning: 18 configurations, fit 2010-2012, validate 2013. Score: mean pinball over q = 0.1/0.5/0.9 on the raw scale. Best: log1p target, df=4, winter interactions, validation pinball 14.775. Identity-target configurations 15.7 to 16.0. Tuning fit time 797 s.

Refit on 2010-2013: 12.45 s wall-clock. Predictions sorted per row, so quantiles do not cross.

Friction: (1) patsy bs() raises NotImplementedError for points outside the outer knots (validation and test covariates); fixed with bounds from feature-file ranges. (2) Winter-interaction spline columns plus month dummies are aliased; QuantReg hits max_iter=5000 (IterationLimitWarning, about 100 s per fit); fixed by pivoted-QR pruning, which the library does not do itself. (3) patsy.dmatrix with an outcome term fails with "encountered outcome variables for a model that does not expect them" when building the prediction design.
