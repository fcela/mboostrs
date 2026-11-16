"""Prospective mean-path comparison; settings in the committed protocol."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import warnings

os.environ.update({k: '1' for k in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})

import mboostrs as mb
import numpy as np
import polars as pl
from mboostrs._output import json_safe

from benches.study_discovery_calibration import fit_band, partition
from benches.study_poisson_conformal import CONTROLS, SCORES, predicted_record
from benches.study_evidence import data_digest, evidence_document

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'docs/book/benchmarks/poisson_tuning_consistency'
PANELS = ('low', 'original', 'high', 'overdispersed', 'zero_inflated')


def draw(panel, seed, n=1000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-1, 1, (2, n))
    multiplier = {'low': .2, 'original': 1}.get(panel, 10)
    mean = multiplier*np.exp(.3+.6*x1-.4*x2)
    if panel == 'overdispersed':
        y = rng.negative_binomial(5, 5/(5+mean))
        variance = mean+mean**2/5
    elif panel == 'zero_inflated':
        y = (rng.uniform(size=n) >= .3)*rng.poisson(mean/.7)
        variance = mean+.3/.7*mean**2
    else:
        y, variance = rng.poisson(mean), mean
    return pl.DataFrame({'x1': x1, 'x2': x2, 'y': y}), mean, np.sqrt(variance)


@contextmanager
def observe_tuning(trace):
    """Record the actual CV rate/curves without modifying any tuning argument."""
    module = importlib.import_module('mboostrs.discover')
    original = module.tune_mstop

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        trace.append(dict(formula=args[0], nu=kwargs.get('nu', .1),
            rows=len(args[1]), mstop=result.mstop, largest=result.largest,
            risk=np.asarray(result.risk).tolist(), fold_scores=np.asarray(result.fold_scores).tolist(),
            at_boundary=bool(result.at_boundary)))
        return result

    module.tune_mstop = record
    try:
        yield
    finally:
        module.tune_mstop = original


def fit_track(train, seed, track):
    """Only proper-training rows reach selection; calibration occurs afterward."""
    if track.endswith('_discovery'):
        trace = []
        started = time.perf_counter()
        with observe_tuning(trace):
            result = mb.discover(train, 'y', seed=seed)
        model = result.model
        return model, dict(family=result.family, formula=result.formula,
            mstop=result.mstop, nu=model._control.nu, tuning=trace,
            log=result.log, provenance=result.provenance, fit_elapsed=time.perf_counter()-started)
    started = time.perf_counter()
    if track == 'supplied_fixed':
        model = mb.gamboost('y ~ bols(x1) + bols(x2)', train,
                            family=mb.Poisson(), control=mb.boost_control(100))
        return model, dict(formula='y ~ bols(x1) + bols(x2)',
            family='Poisson', mstop=100, nu=.1, fit_elapsed=time.perf_counter()-started)
    tuned = mb.tune_mstop('y ~ bols(x1) + bols(x2)', train,
        family=mb.Poisson(), nu=.02, folds=5, seed=seed, start=100, max_mstop=400)
    return tuned.model, dict(formula='y ~ bols(x1) + bols(x2)',
        family='Poisson', mstop=tuned.mstop, nu=.02, risk=np.asarray(tuned.risk).tolist(),
        fit_elapsed=time.perf_counter()-started)


def run(phase, variant, output, panels=PANELS, only_seeds=None):
    journal = output.with_suffix('.jsonl')
    meta_path = output.with_suffix('.metadata.json')
    if any(p.exists() for p in (output, journal, meta_path)):
        raise FileExistsError(f'Refusing to replace evidence: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = range(6493100, 6493110) if phase == 'development' else range(6494100, 6494150)
    if only_seeds is not None:
        # Re-records chosen protocol draws; the summarizer never treats them as a full phase.
        if not set(only_seeds) <= set(seeds):
            raise ValueError(f'--seeds must be {phase} protocol seeds')
        seeds = [seed for seed in seeds if seed in set(only_seeds)]
    document = evidence_document({'id': 'poisson-tuning-consistency/1', 'phase': phase,
        'variant': variant, 'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'seeds': list(seeds), 'test_seed_offset': 100000},
        [Path(__file__), ROOT/'benches/study_discovery_calibration.py',
         ROOT/'benches/study_poisson_conformal.py'])
    package = Path(mb.__file__).parent
    document['loaded_sources'] = {name: hashlib.sha256((package/name).read_bytes()).hexdigest()
        for name in ('discover.py', 'explore.py', 'conformal.py')}
    document['runtime_source_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=package.parents[1], text=True).strip()
    document['panels'] = list(panels)
    document.pop('results')
    meta_path.write_text(json.dumps(document, indent=2)+'\n')
    count = 0
    with journal.open('x') as stream:
        def write(row):
            nonlocal count
            stream.write(json.dumps(json_safe(row), allow_nan=False)+'\n')
            stream.flush()
            count += 1

        for panel in panels:
            for seed in seeds:
                frame, _, _ = draw(panel, seed)
                test, truth, sd = draw(panel, seed+100000)
                train, calibration, parts = partition(frame, seed)
                base = dict(panel=panel, seed=seed, variant=variant,
                    input_digest=data_digest([frame.to_numpy(), test.to_numpy(), truth]),
                    partition=parts, test_response=test['y'].to_list(), true_mean=truth.tolist(),
                    n_fit=train.height, n_calibration=calibration.height)
                # A fit failure produces every scheduled score row, never fewer draws.
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter('always')
                    for track in ([variant+'_discovery'] if variant == 'baseline' else
                                  [variant+'_discovery', 'supplied_fixed', 'supplied_tuned']):
                        try:
                            model, details = fit_track(train, seed, track)
                            point = model.predict(test.drop('y'), type='response', extrapolation='clip')
                            if isinstance(point, dict):
                                raise ValueError('selected LSS mean is unsupported by standard count scores')
                        except Exception:
                            failure = traceback.format_exc()
                            for score in SCORES:
                                write({**base, 'method': track+'_'+score, 'status': 'failed', 'traceback': failure})
                            continue
                        for score in SCORES:
                            started = time.perf_counter()
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                try:
                                    band = mb.conformal_interval(model, test.drop('y'), calibration=calibration,
                                        score=score, level=.9, extrapolation='clip')
                                    row = {**base, 'method': track+'_'+score, 'status': 'ok',
                                        'details': {**details, 'quantile': band.quantile,
                                            'fit_warnings': [str(w.message) for w in fit_warnings]},
                                        'true_mean_mse': float(np.mean((np.asarray(point)-truth)**2)),
                                        **predicted_record(band.lower, band.upper, point, test, sd)}
                                except Exception:
                                    row = {**base, 'method': track+'_'+score, 'status': 'failed',
                                           'traceback': traceback.format_exc()}
                            row.update(calibration_elapsed=time.perf_counter()-started,
                                       warnings=[str(w.message) for w in caught])
                            write(row)
                if variant == 'corrected':
                    for method in CONTROLS:
                        started = time.perf_counter()
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            try:
                                lo, hi, point, details = fit_band(method, 'poisson', seed, frame, test)
                                assert details['partition'] == parts
                                row = {**base, 'method': method, 'status': 'ok', 'details': details,
                                    'true_mean_mse': float(np.mean((np.asarray(point)-truth)**2)),
                                    **predicted_record(lo, hi, point, test, sd)}
                            except Exception:
                                row = {**base, 'method': method, 'status': 'failed',
                                       'traceback': traceback.format_exc()}
                        row.update(elapsed=time.perf_counter()-started, warnings=[str(w.message) for w in caught])
                        write(row)
                print(json.dumps(dict(phase=phase, variant=variant, panel=panel, seed=seed, records=count)), flush=True)
    with journal.open('rb') as raw, output.open('xb') as destination:
        with gzip.GzipFile(fileobj=destination, mode='wb', mtime=0) as zipped:
            import shutil
            shutil.copyfileobj(raw, zipped)
    document.update(records=count, compressed_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    meta_path.write_text(json.dumps(document, indent=2)+'\n')
    journal.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('development', 'confirmation'), required=True)
    p.add_argument('--variant', choices=('baseline', 'corrected'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panels', nargs='+', choices=PANELS, default=PANELS)
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='record only these protocol seeds (default: every seed of the phase)')
    args = p.parse_args()
    run(args.phase, args.variant, args.output, args.panels, args.seeds)
