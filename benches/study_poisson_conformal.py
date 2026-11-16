"""Frozen count-support audit and independent Poisson score study (#636)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ.update({key: '1' for key in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})

import mboostrs as mb
import numpy as np
import polars as pl

from benches.study_discovery_calibration import draw as old_draw, fit_band, partition, score as old_score
from benches.study_evidence import data_digest, evidence_document

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'docs/book/benchmarks/poisson_conformal'
SCORES = ('absolute', 'poisson_normalized', 'poisson_quantile')
CONTROLS = ('catboost_split', 'catboost_cqr', 'lightgbm_split', 'lightgbm_cqr', 'sklearn_cqr')


def count_metrics(lower, upper, point, test, sd):
    lower, upper, point = (np.asarray(a, dtype=float).ravel() for a in (lower, upper, point))
    values = test['y'].to_numpy()
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError('count-support audit requires nonnegative integer responses')
    result = old_score(lower, upper, point, test, sd)
    lo, hi = np.maximum(np.ceil(lower), 0), np.floor(upper)
    count = np.maximum(hi-lo+1, 0)
    original_covered = (lower <= values) & (values <= upper)
    count_covered = (lo <= values) & (values <= hi)
    if not np.array_equal(original_covered, count_covered):
        raise AssertionError('support intersection changed count coverage')
    result.update(count_size=float(count.mean()), count_span=float(np.maximum(hi-lo, 0).mean()),
                  support_length=float(np.maximum(upper-np.maximum(lower, 0), 0).mean()),
                  empty_count_sets=int(np.sum(count == 0)),
                  count_coverage_identical=True)
    return result


def draw(panel, seed):
    if panel == 'original':
        return old_draw('poisson', seed, 1000)
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-1, 1, (2, 1000))
    multiplier = {'low': .2, 'high': 10}[panel]
    mean = multiplier*np.exp(.3+.6*x1-.4*x2)
    return pl.DataFrame({'x1': x1, 'x2': x2, 'y': rng.poisson(mean)}), np.sqrt(mean)


def predicted_record(lower, upper, point, test, sd):
    arrays = [np.asarray(a, dtype=float).ravel() for a in (lower, upper, point)]
    return {**count_metrics(*arrays, test, sd),
            'predictions': dict(zip(('lower', 'upper', 'point'), [a.tolist() for a in arrays], strict=True))}


def snapshot_sources():
    package = Path(mb.__file__).parent
    return {name: hashlib.sha256((package/name).read_bytes()).hexdigest()
            for name in ('discover.py', 'conformal.py', 'explore.py')}


def run(phase, output):
    if output.exists() or output.with_suffix('.jsonl').exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {output}')
    document = evidence_document({'id': 'poisson-conformal/1', 'phase': phase,
        'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'phase': phase, 'independent_test_seed_offset': 50000},
        [Path(__file__), ROOT/'benches/study_discovery_calibration.py',
         ROOT/'benches/summarize_poisson_conformal.py'])
    document['loaded_sources'] = snapshot_sources()
    document.pop('results')
    journal = output.with_suffix('.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = output.with_suffix('.metadata.json')
    meta.write_text(json.dumps(document, indent=2)+'\n')
    if phase == 'audit':
        seeds, panels = range(6311000, 6311050), ('original',)
        previous = json.loads(gzip.decompress((ROOT/'docs/book/benchmarks/discovery_calibration/fixed.json.gz').read_bytes()))
        prior = {(r['seed'], r['method']): r for r in previous['results'] if r['panel'] == 'poisson'}
    else:
        seeds = range(6361000, 6361020) if phase == 'development' else range(6362000, 6362050)
        panels = ('low', 'original', 'high')
        prior = {}
    count = 0
    with journal.open('x') as stream:
        def write(row):
            nonlocal count
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            count += 1

        for panel in panels:
            for seed in seeds:
                frame, _ = draw(panel, seed)
                test, sd = draw(panel, seed+50000)
                train, calibration, parts = partition(frame, seed)
                inputs = data_digest([frame.to_numpy(), test.to_numpy()])

                def record(method, callback, panel=panel, seed=seed, inputs=inputs, parts=parts,
                           train=train, calibration=calibration, test=test, sd=sd):
                    row = dict(panel=panel, seed=seed, method=method, input_digest=inputs,
                               partition=parts, n_fit=train.height, n_calibration=calibration.height)
                    started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        try:
                            lo, hi, point, details = callback()
                            row.update(status='ok', details=details,
                                       **predicted_record(lo, hi, point, test, sd))
                            if (seed, method) in prior:
                                old = prior[seed, method]
                                row['prior_digest_matches'] = row['prediction_digest'] == old['prediction_digest']
                                row['prior_input_matches'] = inputs == old['input_digest']
                                row['prior_metric_differences'] = {key: row[key]-old[key] for key in
                                    ('coverage', 'width', 'interval_score', 'point_mse')}
                        except Exception:
                            row.update(status='failed', traceback=traceback.format_exc())
                    row.update(elapsed=time.perf_counter()-started, warnings=[str(w.message) for w in caught])
                    write(row)

                if phase == 'audit':
                    for method in ('discovery', 'known', *CONTROLS):
                        record(method, lambda method=method, seed=seed, frame=frame, test=test:
                               fit_band(method, 'poisson', seed, frame, test))
                else:
                    for track in ('known', 'discovery'):
                        started = time.perf_counter()
                        with warnings.catch_warnings(record=True) as fit_warnings:
                            warnings.simplefilter('always')
                            try:
                                if track == 'known':
                                    model = mb.gamboost('y ~ bols(x1) + bols(x2)', train,
                                        family=mb.Poisson(), control=mb.boost_control(100))
                                    details = dict(family='Poisson', formula='y ~ bols(x1) + bols(x2)', mstop=100)
                                else:
                                    selected = mb.discover(train, 'y', seed=seed)
                                    model = selected.model
                                    details = dict(family=selected.family, formula=selected.formula,
                                        spread_formula=selected.spread_formula, mstop=selected.mstop,
                                        provenance=selected.provenance, log=selected.log)
                                point = model.predict(test.drop('y'), type='response', extrapolation='clip')
                                if isinstance(point, dict):
                                    raise ValueError('selected LSS mean is unsupported by standard-model count scores')
                                details.update(fit_elapsed=time.perf_counter()-started,
                                    fit_warnings=[str(w.message) for w in fit_warnings])
                            except Exception:
                                failure = traceback.format_exc()
                                for score in SCORES:
                                    write(dict(panel=panel, seed=seed, method=track+'_'+score, status='failed',
                                        input_digest=inputs, partition=parts, traceback=failure))
                                continue
                        for score in SCORES:
                            def calibrated(score=score, model=model, test=test, calibration=calibration,
                                           point=point, details=details):
                                band = mb.conformal_interval(model, test.drop('y'), calibration=calibration,
                                    level=.9, score=score, extrapolation='clip')
                                np.testing.assert_array_equal(band.fitted_mean, point)
                                return band.lower, band.upper, point, {**details, 'score': score,
                                    'quantile': band.quantile, 'calibration_source': band.calibration_source}
                            record(track+'_'+score, calibrated)
                    for method in CONTROLS:
                        record(method, lambda method=method, seed=seed, frame=frame, test=test:
                               fit_band(method, 'poisson', seed, frame, test))
                progress = dict(phase=phase, panel=panel, seed=seed, records=count)
                output.with_suffix('.progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    with journal.open('rb') as raw, output.open('xb') as destination:
        with gzip.GzipFile(fileobj=destination, mode='wb', mtime=0) as compressed:
            import shutil
            shutil.copyfileobj(raw, compressed)
    document['records'] = count
    document['results_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    document['uncompressed_sha256'] = hashlib.sha256(journal.read_bytes()).hexdigest()
    meta.write_text(json.dumps(document, indent=2)+'\n')
    journal.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audit', 'development', 'confirmation'), required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.phase, args.output or DIRECTORY/(args.phase+'.jsonl.gz'))
