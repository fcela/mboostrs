# Validation

The executed notebook and focused strict MyST HTML build both pass on the integrated main768519f branch. The rendered page contains its great_tables outputs and resolved computed prose. Source notebooks retain no execution outputs. Repository-wide cell and documentation checks pass (187 ToC pages at validation).

The development and confirmation validators check all 480 complete records, archive hashes, disjoint original 1400/600 partitions, method coverage and independently recompute response MSE, latent-mean MSE and NRMSE to 1e-12. Neither the estimator nor its settings changed after reading confirmation. The original hybrid remains in every comparison.

Run `PYTHONPATH=python:. .venv/bin/python benches/summarize_all_column_boundary.py --phase confirmation` and the corresponding development command from the repository root. Measurement source/native identities remain in the original metadata; newer integrated documentation code does not replace those identities. Memory is not measured; shared-hardware elapsed times are descriptive workflow costs.
