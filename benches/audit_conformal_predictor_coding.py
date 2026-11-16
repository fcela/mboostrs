"""Independent raw-pipeline references for the frozen issue #648 diagnostic."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np

import mboostrs as mb
from benches.study_evidence import DETERMINISTIC_UNCERTAINTY, data_digest, evidence_document, write_evidence

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'docs/book/benchmarks/conformal_predictor_coding'


def inputs(family, recipe, n=240):
    rng = np.random.default_rng(636)
    latent = rng.uniform(-1, 1, n)
    y = (.3+.6*latent+.2*rng.normal(size=n) if family == 'gaussian'
         else rng.poisson(np.exp(.3+.6*latent)))
    x = np.exp(latent) if recipe == 'log' else latent.copy()
    query = np.exp([-.6, 0., .6]) if recipe == 'log' else np.array([-.6, 0., .6])
    if recipe == 'missing':
        x[::13] = np.nan
        query[1] = np.nan
    return {'x': x, 'y': y, 'rowid': np.arange(n)}, {'x': query}


def subset(frame, indices):
    return {name: values[indices] for name, values in frame.items()}


def fit(frame, frontend, family, recipe, steps=60):
    term = {'bare': 'x', 'scale': 'scale(x)', 'log': 'log(x)', 'missing': 'scale(x)'}[recipe]
    fitter, rhs = ((mb.glmboost, term) if frontend == 'glm' else (mb.gamboost, f'bols({term})'))
    distribution = (mb.Gaussian() if family == 'gaussian' else mb.Poisson()) if isinstance(family, str) else family
    return fitter('y ~ '+rhs, frame, family=distribution, control=mb.boost_control(steps),
                  na_action='impute' if recipe == 'missing' else 'error')


def cv_record(model, result, folds, rowids):
    return dict(training_rowids=np.asarray(rowids, dtype=int).tolist(),
                family=type(model.family).__name__, budget=int(model.mstop),
                mode=result.preprocessing, optimum=int(result.optimal_mstop),
                fold_digest=data_digest([folds]), aggregate=result.aggregate.tolist())


def direct_refit(frame, frontend, family, recipe, budget, trace):
    if budget != 'tune':
        return fit(frame, frontend, family, recipe, budget)
    folds = mb.cv_weights(len(frame['y']), 'kfold', 5, seed=4)
    largest = 1600
    while True:
        model = fit(frame, frontend, family, recipe, largest)
        result = model.cvrisk(folds=folds, preprocessing='refit')
        trace.append(cv_record(model, result, folds, frame['rowid']))
        if not result.at_boundary() or largest >= 10000:
            return model[:max(int(result.optimal_mstop), 1)]
        largest = min(2*largest, 10000)


def reference(frame, query, frontend, family, recipe, method, budget, *, n_folds=4, seed=4):
    n = len(frame['y'])
    permutation = np.random.default_rng(seed).permutation(n)
    trace = []
    target = frame if query is None else query
    if method == 'cqr':
        calibration, training = permutation[:round(n*.5)], permutation[round(n*.5):]
        lower = direct_refit(subset(frame, training), frontend, mb.Quantile(tau=.05), recipe, budget, trace)
        upper = direct_refit(subset(frame, training), frontend, mb.Quantile(tau=.95), recipe, budget, trace)
        held = subset(frame, calibration)
        a, b = lower.predict(held, type='response'), upper.predict(held, type='response')
        scores = np.maximum(np.minimum(a, b)-held['y'], held['y']-np.maximum(a, b))
        correction = np.sort(scores)[int(np.ceil((len(scores)+1)*.9))-1]
        a, b = lower.predict(target, type='response'), upper.predict(target, type='response')
        lo, hi = np.minimum(a, b)-correction, np.maximum(a, b)+correction
        crossed = lo > hi
        middle = .5*(lo+hi)
        lo, hi = np.where(crossed, middle, lo), np.where(crossed, middle, hi)
        point = .5*(lo+hi)
        partitions = {'training_rows': training.tolist(), 'calibration_rows': calibration.tolist()}
        stops = [lower.mstop, upper.mstop]
        extra = dict(quantile=float(correction), n_training=len(training), n_calibration=len(calibration),
                     n_set_aside=0, guaranteed_coverage=.9)
    else:
        size = n//n_folds
        folds = permutation[:size*n_folds].reshape(n_folds, size)
        residuals, prediction, training_rows, stops = [], [], [], []
        for held in folds:
            training = np.setdiff1d(folds.ravel(), held)
            model = direct_refit(subset(frame, training), frontend, family, recipe, budget, trace)
            residuals.extend(np.abs(frame['y'][held]-model.predict(subset(frame, held), type='response')))
            prediction.extend([model.predict(target, type='response')]*len(held))
            training_rows.append(training.tolist())
            stops.append(model.mstop)
        residuals, prediction = np.asarray(residuals), np.asarray(prediction)
        count = len(residuals)
        lo_rank, hi_rank = int(np.floor((1-.9)*(count+1))), int(np.ceil(.9*(count+1)))
        lo = np.sort(prediction-residuals[:, None], axis=0)[lo_rank-1]
        hi = np.sort(prediction+residuals[:, None], axis=0)[hi_rank-1]
        full = fit(frame, frontend, family, recipe, max(60, budget) if isinstance(budget, int) else 60)
        point = full.predict(target, type='response', mstop=budget if isinstance(budget, int) else None)
        partitions = {'training_rows': training_rows, 'held_rows': folds.tolist()}
        correction = min(2*(1-1/n_folds)/(size+1), (1-n_folds/count)/(n_folds+1))
        extra = dict(quantile=None, n_training=count-size, n_calibration=count,
                     n_set_aside=n-count, guaranteed_coverage=1-2*(1-.9)-correction)
    return dict(lower=np.asarray(lo).tolist(), upper=np.asarray(hi).tolist(),
                predicted=np.asarray(point).tolist(), partitions=partitions, stops=stops, inner_cv=trace, **extra)


@contextmanager
def observe_inner_cv(trace):
    """Observe actual production CV boundaries; reference fitting uses no hooks."""
    classes = [importlib.import_module('mboostrs.'+name).__dict__[cls]
               for name, cls in (('glmboost', 'GLMBoostModel'), ('gamboost', 'GAMBoostModel'))]
    originals = {cls: cls.cvrisk for cls in classes}
    def observer(original):
        def call(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            trace.append(cv_record(model, result, kwargs['folds'], model._data['rowid']))
            return result
        return call
    try:
        for cls in classes:
            cls.cvrisk = observer(originals[cls])
        yield
    finally:
        for cls in classes:
            cls.cvrisk = originals[cls]


def cases():
    for frontend, family, method, recipe, target in itertools.product(
            ('glm', 'gam'), ('gaussian', 'poisson'), ('cv+', 'cqr'),
            ('bare', 'scale', 'log', 'missing'), ('query', 'training')):
        yield dict(frontend=frontend, family=family, method=method, recipe=recipe, target=target, budget=30)
    for frontend, method, recipe in itertools.product(('glm', 'gam'), ('cv+', 'cqr'), ('scale', 'missing')):
        yield dict(frontend=frontend, family='gaussian', method=method, recipe=recipe, target='query', budget='tune')


def run(output, phase):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic evidence: {output}')
    document = evidence_document({'id': 'conformal-predictor-coding/1', 'phase': phase,
        'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'unit': 'one deterministic fixture, no coverage experiment'}, [Path(__file__)],
        uncertainty=DETERMINISTIC_UNCERTAINTY)
    document['numerical_source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('python/mboostrs/conformal.py', 'python/mboostrs/_cv_preprocessing.py')}
    for config in cases():
        frame, query = inputs(config['family'], config['recipe'])
        query = None if config['target'] == 'training' else query
        row = {**config, 'input_digest': data_digest(list(frame.values()))}
        actual_trace = []
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            expected = reference(frame, query, config['frontend'], config['family'], config['recipe'],
                                 config['method'], config['budget'])
            row['reference'] = expected
            try:
                model = fit(frame, config['frontend'], config['family'], config['recipe'])
                with observe_inner_cv(actual_trace):
                    interval = mb.conformal_interval(model, query, level=.9, method=config['method'],
                        seed=4, n_folds=4, mstop=config['budget'], extrapolation='allow')
                row['observed'] = {name: np.asarray(getattr(interval, name)).tolist()
                                   for name in ('lower', 'upper', 'predicted')}
                row['max_absolute_error'] = max(float(np.max(np.abs(np.asarray(row['observed'][name])-expected[name])))
                                                 for name in row['observed'])
                row['observed_fields'] = {name: getattr(interval, name) for name in
                    ('quantile', 'n_training', 'n_calibration', 'n_set_aside', 'guaranteed_coverage')}
                row['status'] = 'ok'
            except Exception:
                row.update(status='failed', traceback=traceback.format_exc())
        row.update(inner_cv=actual_trace, elapsed=time.perf_counter()-started,
                   warnings=[str(w.message) for w in caught])
        document['results'].append(row)
        write_evidence(document, output)
        print(json.dumps({**config, 'status': row['status'], 'error': row.get('max_absolute_error')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'corrected'), required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.output or DIRECTORY/(args.phase+'.json'), args.phase)
