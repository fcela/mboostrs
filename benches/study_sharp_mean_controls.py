"""Frozen development controls for mean-error attribution to scale (#629)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ.update({k: '1' for k in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})

import mboostrs as mb
import numpy as np
import polars as pl
from mboostrs._output import json_safe

from benches.study_discovery_calibration import partition, score
from benches.study_evidence import data_digest, evidence_document

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'docs/book/benchmarks/sharp_mean_controls'
PANELS = ('axis_constant', 'axis_varying', 'shifted_constant', 'shifted_varying',
          'smooth_constant', 'smooth_absolute', 'smooth_bell', 'smooth_monotone',
          'null_constant', 'heavy_tail')


def draw(panel, seed, n):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 5))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    sharp = panel.startswith(('axis', 'shifted'))
    a, b = (.25, -.2) if panel.startswith('shifted') else (0, 0)
    mean = (3*np.sign(x1-a)*np.sign(x2-b)+x3 if sharp else
            np.zeros(n) if panel == 'null_constant' else np.sin(2*np.pi*x1)+.5*x2)
    sd = np.full(n, .5 if sharp else 1.)
    if panel.endswith('varying') or panel == 'smooth_absolute':
        sd = .3+1.5*np.abs(x1)
    elif panel == 'smooth_bell':
        sd = .3+1.5*np.exp(-8*x1**2)
    elif panel == 'smooth_monotone':
        sd = np.exp(.7*x1)
    noise = rng.standard_t(3, n)/np.sqrt(3) if panel == 'heavy_tail' else rng.normal(size=n)
    frame = pl.DataFrame({name: x[:, i] for i, name in enumerate(('x1', 'x2', 'x3', 'n0', 'n1'))})
    return frame.with_columns(pl.Series('y', mean+sd*noise)), mean, sd


def fitted(method, panel, train, seed):
    if method.startswith('discovery'):
        result = mb.discover(train, 'y', spread=method == 'discovery_spread', seed=seed)
        return result.model, dict(formula=result.formula, family=result.family,
            spread_formula=result.spread_formula, mstop=result.mstop,
            log=result.log, provenance=result.provenance)
    mu = ('btree(x1, x2, maxdepth=3, selection=sse) + bols(x3)'
          if panel.startswith(('axis', 'shifted')) else '1' if panel == 'null_constant'
          else 'bbs(x1, df=4) + bols(x2)')
    sigma = 'bbs(x1, df=4)' if method == 'supplied_varying' else '1'
    model = mb.gamboostLSS({'mu': 'y ~ '+mu, 'sigma': 'y ~ '+sigma}, train,
        family=mb.GaussianLSS(), control=mb.boost_control(100))
    return model, dict(formula=mu, spread_formula=sigma, mstop=100)


def oracle_diagnostic(train, mean):
    """Remove the known mean only for this labelled diagnostic control."""
    residual = train['y'].to_numpy()-mean
    oracle_frame = train.with_columns(pl.Series('y', residual))
    oracle = mb.gamboost('y ~ 1', oracle_frame, control=mb.boost_control(1))
    return mb.heteroscedasticity(oracle, oracle_frame)


def run():
    output = DIRECTORY/'development.jsonl.gz'
    journal = DIRECTORY/'development.jsonl'
    metadata = DIRECTORY/'development.metadata.json'
    if any(p.exists() for p in (output, journal, metadata)):
        raise FileExistsError('Refusing to overwrite sharp-mean controls')
    document = evidence_document({'id': 'sharp-mean-controls/1', 'phase': 'development',
        'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'seeds': list(range(6293100, 6293110)), 'test_seed_offset': 100000}, [Path(__file__)])
    source = Path(mb.__file__).parent
    document['loaded_sources'] = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ('discover.py', 'explore.py', 'conformal.py')}
    document.pop('results')
    metadata.write_text(json.dumps(document, indent=2)+'\n')
    count = 0
    with journal.open('x') as stream:
        def write(row):
            nonlocal count
            stream.write(json.dumps(json_safe(row), allow_nan=False)+'\n')
            stream.flush()
            count += 1

        for panel in PANELS:
            for seed in range(6293100, 6293110):
                frame, mean, _ = draw(panel, seed, 1000)
                test, truth, sd = draw(panel, seed+100000, 2000)
                train, calibration, parts = partition(frame, seed)
                base = dict(panel=panel, seed=seed, partition=parts,
                    input_digest=data_digest([frame.to_numpy(), test.to_numpy(), mean, truth]),
                    test_response=test['y'].to_list(), true_mean=truth.tolist())
                for method in ('discovery_spread', 'discovery_constant', 'supplied_constant', 'supplied_varying'):
                    started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        try:
                            model, details = fitted(method, panel, train, seed)
                            band = mb.conformal_interval(model, test.drop('y'), calibration=calibration,
                                                         extrapolation='clip')
                            point = band.fitted_mean
                            a, b = (.25, -.2) if panel.startswith('shifted') else (0, 0)
                            boundary = ((np.abs(test['x1'].to_numpy()-a) < .1) |
                                        (np.abs(test['x2'].to_numpy()-b) < .1))
                            row = {**base, 'method': method, 'status': 'ok', 'details': details,
                                **score(band.lower, band.upper, point, test, sd),
                                'true_mean_mse': float(np.mean((point-truth)**2)),
                                'boundary_coverage': float(band.contains(test['y'])[boundary].mean()),
                                'predictions': dict(lower=band.lower.tolist(), upper=band.upper.tolist(),
                                                    point=point.tolist())}
                        except Exception:
                            row = {**base, 'method': method, 'status': 'failed', 'traceback': traceback.format_exc()}
                    row.update(elapsed=time.perf_counter()-started, warnings=[str(w.message) for w in caught])
                    write(row)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        diagnostic = oracle_diagnostic(train, mean[parts['selection_rows']])
                        row = {**base, 'method': 'oracle_residual', 'status': 'ok', 'diagnostic': diagnostic}
                    except Exception:
                        row = {**base, 'method': 'oracle_residual', 'status': 'failed', 'traceback': traceback.format_exc()}
                write({**row, 'warnings': [str(w.message) for w in caught]})
                print(json.dumps(dict(panel=panel, seed=seed, records=count)), flush=True)
    with journal.open('rb') as raw, output.open('xb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', mtime=0) as zipped:
            import shutil
            shutil.copyfileobj(raw, zipped)
    document.update(records=count, compressed_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    metadata.write_text(json.dumps(document, indent=2)+'\n')
    journal.unlink()


if __name__ == '__main__':
    run()
