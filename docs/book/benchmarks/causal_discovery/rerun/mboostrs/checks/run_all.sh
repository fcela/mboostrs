#!/bin/bash
set -e
cd "docs/book/benchmarks/causal_discovery/rerun/mboostrs"
source ".venv/bin/activate"
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2
rm -rf dev_out checks/null_seed5 checks/null_seed6 checks/small_1500
python run.py --train "docs/book/benchmarks/causal_discovery/data/dev/train.parquet" --test "docs/book/benchmarks/causal_discovery/data/dev/test.parquet" --out dev_out --seed 1 2>/dev/null
python run.py --train <scratch>/scratch_null/train_null5.parquet --test "docs/book/benchmarks/causal_discovery/data/dev/test.parquet" --out checks/null_seed5 --seed 1 2>/dev/null
python run.py --train <scratch>/scratch_null/train_null6.parquet --test "docs/book/benchmarks/causal_discovery/data/dev/test.parquet" --out checks/null_seed6 --seed 1 2>/dev/null
python run.py --train <scratch>/scratch_small/train1500.parquet --test "docs/book/benchmarks/causal_discovery/data/dev/test.parquet" --out checks/small_1500 --seed 2 2>/dev/null
echo ALLDONE
