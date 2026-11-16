"""Frozen whole-procedure calibration study for issue #631."""
from __future__ import annotations

import argparse
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
from mboostrs.cvrisk import cv_weights
import numpy as np
import polars as pl

from benches.study_evidence import data_digest, evidence_document, write_evidence

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/book/benchmarks/discovery_calibration'
PANELS = ('absolute', 'constant', 'null', 'sharp', 'small', 'correlated', 'poisson', 'heavy_tail')


def draw(panel, seed, n):
    rng = np.random.default_rng(seed)
    if panel == 'correlated':
        x1, z = rng.normal(size=(2, n))
        x2 = .9*x1 + np.sqrt(.19)*z
        mean, sd = .8*x1+.4*x2, np.full(n, .5)
    else:
        x1, x2 = rng.uniform(-1, 1, (2, n))
        mean = np.sin(2*np.pi*x1)+.5*x2
        sd = .3+1.5*np.abs(x1) if panel == 'absolute' else np.ones(n)
    columns = {'x1': x1, 'x2': x2}
    if panel == 'sharp':
        # Five independent columns: keep the first two identical to other uniform panels.
        x3, n0, n1 = rng.uniform(-1, 1, (3, n))
        columns.update(x3=x3, n0=n0, n1=n1)
        mean, sd = 3*np.sign(x1)*np.sign(x2)+x3, np.full(n, .5)
    if panel == 'null':
        mean = np.zeros(n)
    if panel == 'poisson':
        mean = np.exp(.3+.6*x1-.4*x2)
        sd = np.sqrt(mean)
        y = rng.poisson(mean)
    else:
        noise = rng.standard_t(3, n)/np.sqrt(3) if panel == 'heavy_tail' else rng.normal(size=n)
        y = mean+sd*noise
    return pl.DataFrame({**columns, 'y': y}), sd


def partition(frame, seed):
    weights = cv_weights(frame.height, type='kfold', B=2, seed=seed)[:, 0]
    selection, calibration = np.flatnonzero(weights > 0), np.flatnonzero(weights == 0)
    return frame[selection], frame[calibration], dict(selection_rows=selection.tolist(),
        calibration_rows=calibration.tolist(), digest=data_digest([selection, calibration]))


def score(lo, hi, point, test, sd):
    y = test['y'].to_numpy()
    lo, hi, point = (np.asarray(v, dtype=float).ravel() for v in (lo, hi, point))
    assert y.shape == lo.shape == hi.shape == point.shape
    assert np.all(np.isfinite(lo+hi+point))
    covered = (lo <= y) & (y <= hi)
    low, high = np.quantile(sd, [1/3, 2/3])
    return dict(coverage=float(covered.mean()), width=float(np.maximum(hi-lo, 0).mean()),
        interval_score=float(np.mean(hi-lo+20*np.maximum(lo-y, 0)+20*np.maximum(y-hi, 0))),
        point_mse=float(np.mean((point-y)**2)), coverage_low=float(covered[sd <= low].mean()),
        coverage_high=float(covered[sd >= high].mean()), empty_sets=int(np.sum(hi < lo)),
        prediction_digest=data_digest([lo, hi, point]))


def fit_band(method, panel, seed, frame, test):
    new = test.drop('y')
    if method == 'discovery':
        result = mb.discover(frame, 'y', newdata=new, seed=seed, level=.9)
        details = dict(family=result.family, formula=result.formula, spread_formula=result.spread_formula,
            mstop=result.mstop, log=result.log, provenance=result.provenance)
        if getattr(result, 'interval', None) is not None:
            details['interval'] = result.interval.to_dict()
        return result.predictions['lower'], result.predictions['upper'], result.predictions['mean'], details
    fit, calibration, parts = partition(frame, seed)
    if method == 'known':
        rhs = {'null': '1', 'sharp': 'btree(x1, x2, maxdepth=3, selection=sse) + bols(x3)',
               'correlated': 'bols(x1) + bols(x2)', 'poisson': 'bols(x1) + bols(x2)'}.get(
                   panel, 'bbs(x1, df=4) + bols(x2)')
        if panel == 'absolute':
            model = mb.gamboostLSS({'mu': 'y ~ '+rhs, 'sigma': 'y ~ bbs(x1, df=4)'}, fit,
                family=mb.GaussianLSS(), control=mb.boost_control(100))
        else:
            model = mb.gamboost('y ~ '+rhs, fit, family=mb.Poisson() if panel == 'poisson' else mb.Gaussian(),
                               control=mb.boost_control(100))
        band = mb.conformal_interval(model, new, level=.9, calibration=calibration,
                                    extrapolation='clip')
        return band.lower, band.upper, band.fitted_mean, {'partition': parts, 'interval': band.to_dict()}
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    x, xc, xt = (a.drop('y').to_numpy() for a in (fit, calibration, test))
    y, yc = fit['y'].to_numpy(), calibration['y'].to_numpy()
    brand, mode = method.split('_')
    def estimator(alpha=None):
        if brand == 'catboost':
            return CatBoostRegressor(iterations=500, depth=6, learning_rate=.05, thread_count=1,
                verbose=False, allow_writing_files=False, random_seed=seed,
                loss_function='RMSE' if alpha is None else f'Quantile:alpha={alpha}')
        if brand == 'lightgbm':
            return LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=.05, n_jobs=1,
                verbosity=-1, random_state=seed, objective='regression' if alpha is None else 'quantile',
                **({} if alpha is None else {'alpha': alpha}))
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05,
            random_state=seed, loss='quantile', alpha=alpha)
    if mode == 'split':
        model = estimator().fit(x, y)
        residual = np.abs(yc-model.predict(xc))
        lo = hi = model.predict(xt)
    else:
        low, high = estimator(.05).fit(x, y), estimator(.95).fit(x, y)
        residual = np.maximum(low.predict(xc)-yc, yc-high.predict(xc))
        lo, hi = low.predict(xt), high.predict(xt)
    q = np.sort(residual)[min(int(np.ceil((len(yc)+1)*.9)), len(yc))-1]
    return lo-q, hi+q, (lo+hi)/2, {'partition': parts, 'n_fit': len(y), 'n_calibration': len(yc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('baseline', 'fixed'), required=True)
    args = parser.parse_args()
    output = DIRECTORY/f'{args.phase}.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite measured evidence: {output}')
    document = evidence_document({'id': 'discovery-calibration/1', 'phase': args.phase,
        'sha256': hashlib.sha256((DIRECTORY/'PROTOCOL.md').read_bytes()).hexdigest()},
        {'seeds': list(range(6311000, 6311050)), 'independent_test_seed_offset': 50000}, [Path(__file__)])
    source = Path(mb.__file__).parent
    document['loaded_sources'] = {str(source/name): hashlib.sha256((source/name).read_bytes()).hexdigest()
                                  for name in ('discover.py', 'conformal.py', 'explore.py')}
    methods = ('discovery',) if args.phase == 'baseline' else (
        'discovery', 'known', 'catboost_split', 'catboost_cqr', 'lightgbm_split', 'lightgbm_cqr', 'sklearn_cqr')
    for panel in PANELS:
        for seed in range(6311000, 6311050):
            n = 80 if panel == 'small' else 800 if panel == 'sharp' else 1000
            frame, _ = draw(panel, seed, n)
            test, sd = draw(panel, seed+50000, 1000)
            for method in methods:
                row = dict(panel=panel, seed=seed, method=method, phase=args.phase,
                    input_digest=data_digest([frame.to_numpy(), test.to_numpy()]), n_development=n)
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        lo, hi, point, details = fit_band(method, panel, seed, frame, test)
                        row.update(status='ok', details=details, **score(lo, hi, point, test, sd))
                    except Exception:
                        row.update(status='failed', traceback=traceback.format_exc())
                row.update(elapsed=time.perf_counter()-start, warnings=[str(w.message) for w in caught])
                document['results'].append(row)
                write_evidence(document, output)
            print(json.dumps({'panel': panel, 'seed': seed, 'records': len(document['results'])}), flush=True)


if __name__ == '__main__':
    main()
