"""Independent original-response references for the frozen issue #652 study."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import traceback
import warnings

import numpy as np

import mboostrs as mb
from benches.audit_conformal_predictor_coding import observe_inner_cv, subset
from benches.study_evidence import DETERMINISTIC_UNCERTAINTY, data_digest, evidence_document, write_evidence

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/book/benchmarks/conformal_response_units'
TRANSFORMS = ('bare', 'center', 'scale', 'log', 'log1p', 'sqrt', 'exp')


def inputs(n=240):
    rng = np.random.default_rng(636)
    x = rng.uniform(-1, 1, n)
    return ({'x': x, 'y': np.exp(.3+.6*x+.2*rng.normal(size=n)),
             'rowid': np.arange(n)}, {'x': np.array([-.6, 0., .6])})


def fit(frame, frontend, transform, steps=60, family=None, predictor='x', **kwargs):
    lhs = 'y' if transform == 'bare' else f'{transform}(y)'
    fitter, rhs = ((mb.glmboost, predictor) if frontend == 'glm'
                   else (mb.gamboost, f'bols({predictor})'))
    return fitter(f'{lhs} ~ {rhs}', frame, family=family or mb.Gaussian(),
                  control=mb.boost_control(steps), **kwargs)


def original_prediction(model, train, query, transform, mstop=None):
    """Explicit inverse formulas, independent of production transform helpers."""
    z = model.predict(query, type='response', mstop=mstop)
    if transform == 'center':
        return z + train['y'].mean()
    if transform == 'scale':
        return z * train['y'].std(ddof=1) + train['y'].mean()
    if transform == 'log':
        return np.exp(z)
    if transform == 'log1p':
        return np.expm1(z)
    if transform == 'sqrt':
        return z**2
    if transform == 'exp':
        return np.log(z)
    return z


def direct_fit(frame, frontend, transform, budget, trace, family=None):
    if budget != 'tune':
        return fit(frame, frontend, transform, budget, family)
    folds = mb.cv_weights(len(frame['y']), 'kfold', 5, seed=4)
    largest = 1600
    while True:
        model = fit(frame, frontend, transform, largest, family)
        cv = model.cvrisk(folds=folds, preprocessing='refit')
        trace.append(dict(rowids=frame['rowid'].tolist(), budget=largest,
                          optimum=int(cv.optimal_mstop), aggregate=cv.aggregate.tolist()))
        if not cv.at_boundary() or largest >= 10000:
            return model[:max(int(cv.optimal_mstop), 1)]
        largest = min(2*largest, 10000)


def reference(frame, query, frontend, transform, method, budget, route='internal', n_folds=4):
    n = len(frame['y'])
    permutation = np.random.default_rng(4).permutation(n)
    target = frame if query is None else query
    trace, stops = [], []
    if method in ('split', 'cqr'):
        cal, train = permutation[:round(n*.5)], permutation[round(n*.5):]
        proper, held = subset(frame, train), subset(frame, cal)
        families = [mb.Quantile(tau=.05), mb.Quantile(tau=.95)] if method == 'cqr' else [mb.Gaussian()]
        models = [direct_fit(proper, frontend, transform, budget, trace, f) for f in families]
        cal_pred = [original_prediction(m, proper, held, transform) for m in models]
        pred = [original_prediction(m, proper, target, transform) for m in models]
        stops = [m.mstop for m in models]
        if method == 'split':
            scores = np.abs(held['y']-cal_pred[0])
        else:
            scores = np.maximum(np.minimum(*cal_pred)-held['y'], held['y']-np.maximum(*cal_pred))
        q = np.sort(scores)[int(np.ceil((len(scores)+1)*.9))-1]
        if method == 'split':
            point = pred[0]
            lo, hi = point-q, point+q
        else:
            lo, hi = np.minimum(*pred)-q, np.maximum(*pred)+q
            middle = .5*(lo+hi)
            crossed = lo > hi
            lo, hi = np.where(crossed, middle, lo), np.where(crossed, middle, hi)
            point = .5*(lo+hi)
        partitions = dict(training_rows=train.tolist(), calibration_rows=cal.tolist())
        if route == 'external' and query is None:
            # External newdata=None refers to the proper model's own training rows.
            point = original_prediction(models[0], proper, proper, transform)
            lo, hi = point-q, point+q
    else:
        folds = permutation[:n//n_folds*n_folds].reshape(n_folds, -1)
        residuals, predictions, train_rows = [], [], []
        for held in folds:
            train = np.setdiff1d(folds.ravel(), held)
            proper = subset(frame, train)
            model = direct_fit(proper, frontend, transform, budget, trace)
            residuals.extend(np.abs(frame['y'][held]-original_prediction(model, proper, subset(frame, held), transform)))
            predictions.extend([original_prediction(model, proper, target, transform)]*len(held))
            stops.append(model.mstop)
            train_rows.append(train.tolist())
        residuals, predictions = np.asarray(residuals), np.asarray(predictions)
        count = len(residuals)
        a, b = int(np.floor((1-.9)*(count+1))), int(np.ceil(.9*(count+1)))
        lo = np.sort(predictions-residuals[:, None], axis=0)[a-1]
        hi = np.sort(predictions+residuals[:, None], axis=0)[b-1]
        full = fit(frame, frontend, transform)
        point = original_prediction(full, frame, target, transform, budget if isinstance(budget, int) else None)
        partitions, q = dict(training_rows=train_rows, calibration_rows=folds.tolist()), None
    return dict(lower=np.asarray(lo).tolist(), upper=np.asarray(hi).tolist(),
                predicted=np.asarray(point).tolist(), quantile=None if q is None else float(q),
                partitions=partitions, stops=stops, inner_cv=trace)


def cases():
    for frontend, transform, method, target in itertools.product(
            ('glm', 'gam'), TRANSFORMS, ('split', 'cv+', 'cqr'), ('query', 'training')):
        yield dict(frontend=frontend, transform=transform, method=method, target=target,
                   budget=30, route='internal')
    for frontend, transform, target in itertools.product(('glm', 'gam'), TRANSFORMS, ('query', 'training')):
        yield dict(frontend=frontend, transform=transform, method='split', target=target,
                   budget=30, route='external')
    for frontend, method, transform in itertools.product(('glm', 'gam'), ('cqr', 'cv+'), ('bare', 'log')):
        yield dict(frontend=frontend, transform=transform, method=method, target='query',
                   budget='tune', route='internal')
    for frontend, method, transform in itertools.product(('glm', 'gam'), ('split', 'cv+', 'cqr'), ('center', 'scale')):
        yield dict(frontend=frontend, transform=transform, method=method, target='query',
                   budget='tune', route='internal', expected_refusal=True)


def run(output, phase):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {output}')
    document = evidence_document({'id': 'conformal-response-units/1', 'phase': phase,
        'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'unit': 'one deterministic original-response fixture; no coverage experiment'}, [Path(__file__)],
        uncertainty=DETERMINISTIC_UNCERTAINTY)
    document['numerical_source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('python/mboostrs/conformal.py', 'python/mboostrs/_cv_preprocessing.py',
                     'python/mboostrs/transforms.py')}
    for config in cases():
        frame, query = inputs()
        query = None if config['target'] == 'training' else query
        row = {**config, 'input_digest': data_digest(list(frame.values()))}
        trace = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if not config.get('expected_refusal'):
                row['reference'] = reference(frame, query, config['frontend'], config['transform'],
                    config['method'], config['budget'], config['route'])
            try:
                calibration = None
                if config['route'] == 'external':
                    p = np.random.default_rng(4).permutation(len(frame['y']))
                    calibration, frame = subset(frame, p[:120]), subset(frame, p[120:])
                model = fit(frame, config['frontend'], config['transform'])
                with observe_inner_cv(trace):
                    interval = mb.conformal_interval(model, query, method=config['method'], seed=4,
                        n_folds=4, mstop=config['budget'], calibration=calibration, extrapolation='allow')
                row['observed'] = {name: getattr(interval, name).tolist() for name in ('lower', 'upper', 'predicted')}
                errors = [float(np.max(np.abs(np.asarray(row['observed'][name])-row['reference'][name])))
                          for name in row['observed']]
                if interval.quantile is not None:
                    errors.append(abs(interval.quantile-row['reference']['quantile']))
                row.update(status='ok', max_absolute_error=max(errors), quantile=interval.quantile,
                           fitted_mean_available=interval.fitted_mean is not None)
            except Exception as error:
                row.update(status='refused' if config.get('expected_refusal') and
                           isinstance(error, NotImplementedError) and 'scale changes between folds' in str(error)
                           else 'failed', error=str(error), traceback=traceback.format_exc())
        row.update(inner_cv=trace, warnings=[str(w.message) for w in caught])
        document['results'].append(row)
        write_evidence(document, output)
        print(json.dumps({**config, 'status': row['status'], 'error': row.get('max_absolute_error')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'corrected'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.phase)
