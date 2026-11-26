# Distributional prediction shootout

Runner for the study in
[docs/benchmarks/distributional_shootout](../../docs/benchmarks/distributional_shootout/PROTOCOL.md):
17 methods that predict a whole conditional distribution, scored by CRPS,
calibration and interval coverage on 48 planted-truth task instances and the
35 OpenML-CTR23 regression datasets.

The competitors need their own venv: NGBoost depends on lifelines, which
pins pandas below 3.0, and the repository venv uses pandas 3. The venv reads
mboostrs from `../../python` through a `.pth` file, so it uses the extension
built in the repository.

```sh
cd benches/distributional_shootout
uv venv --python 3.12 .venv
echo "$(cd ../../python && pwd)" > .venv/lib/python3.12/site-packages/mboostrs.pth
uv pip install --python .venv/bin/python -r requirements.txt
# DisTreebution's Rust trees publish only a Linux x86_64 wheel; build from source:
git clone https://github.com/quentin-duchemin/distreebution-rs /tmp/distreebution-rs
git -C /tmp/distreebution-rs checkout 08c73bad12b22b8389631a36da4b9c808f6a6657
V="$PWD/.venv"
(cd /tmp/distreebution-rs/distreebu_rs && "$V/bin/maturin" build --release -i "$V/bin/python")
uv pip install --python .venv/bin/python /tmp/distreebution-rs/distreebu_rs/target/wheels/distreebu_rs-0.3.0-*.whl

.venv/bin/python -I tasks.py                               # OpenML-CTR23 -> data/
.venv/bin/python -I run_all.py .venv/bin/python 8 results.jsonl
.venv/bin/python -I analyze.py results.jsonl ../../docs/benchmarks/distributional_shootout
```

`run_all.py` runs each (task, seed, arm) in its own process with all thread
pools set to one, skips jobs already in the results file, and kills a job
after 3,600 seconds. One job by hand:
`.venv/bin/python -I run_one.py synth S1_hetero_gauss@500 0 hessboost.tuned`.

Files: `tasks.py` (generators with their true quantiles, CTR23 download and
splits), `arms.py` (one function per method), `metrics.py` (scores),
`run_one.py`, `run_all.py`, `analyze.py`.
