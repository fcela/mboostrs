"""Summaries and paired uncertainty for the frozen #636 score study."""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np

from benches.study_poisson_conformal import CONTROLS, DIRECTORY

METRICS = ('coverage', 'count_size', 'width', 'support_length', 'count_span', 'interval_score',
           'point_mse', 'coverage_low', 'coverage_high', 'empty_count_sets', 'elapsed')


def load(phase, directory=DIRECTORY):
    path = directory/(phase+'.jsonl.gz')
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['results_sha256']:
        raise ValueError(f'archive checksum mismatch: {path}')
    rows = [json.loads(line) for line in gzip.open(path, 'rt')]
    if len(rows) != metadata['records']:
        raise ValueError('archive record count mismatch')
    return rows, metadata


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()),
            'se': float(values.std(ddof=1)/np.sqrt(values.size)) if values.size > 1 else None}


def summarize(phase, rows):
    panels = sorted({row['panel'] for row in rows})
    summary = []
    for panel in panels:
        methods = sorted({row['method'] for row in rows if row['panel'] == panel})
        groups = {}
        for method in methods:
            group = [r for r in rows if r['panel'] == panel and r['method'] == method]
            ok = [r for r in group if r['status'] == 'ok']
            groups[method] = ok
            item = dict(phase=phase, panel=panel, method=method, n=len(ok), failures=len(group)-len(ok))
            for metric in METRICS:
                item[metric] = mean_se([r[metric] for r in ok]) if ok else None
            summary.append(item)
        eligible = [method for method in CONTROLS if groups.get(method) and
                    .88 <= np.mean([r['coverage'] for r in groups[method]]) <= .92]
        for item in summary:
            if item['panel'] != panel or item['phase'] != phase:
                continue
            for metric in ('count_size', 'width'):
                leader = min(eligible, key=lambda method: np.mean([r[metric] for r in groups[method]])) if eligible else None
                item[metric+'_leader'] = leader
                item[metric+'_gap_percent'] = None
                if leader is not None:
                    peer = {r['seed']: r for r in groups[leader]}
                    deltas = [100*(r[metric]/peer[r['seed']][metric]-1) for r in groups[item['method']]
                              if r['seed'] in peer and peer[r['seed']][metric] > 0]
                    if deltas:
                        item[metric+'_gap_percent'] = mean_se(deltas)
            item['coverage_eligible'] = bool(item['coverage'] and .88 <= item['coverage']['mean'] <= .92)
            if item['method'].startswith(('known_', 'discovery_')):
                track = item['method'].partition('_')[0]
                baseline = {r['seed']: r for r in groups[track+'_absolute']}
                item['paired_to_absolute'] = {}
                for metric in ('coverage', 'count_size', 'width', 'interval_score', 'point_mse'):
                    differences = [r[metric]-baseline[r['seed']][metric] for r in groups[item['method']]
                                   if r['seed'] in baseline]
                    item['paired_to_absolute'][metric] = mean_se(differences) if differences else None
    return summary


def validate(phase, rows):
    expected = {'audit': 350, 'development': 660, 'confirmation': 1650}[phase]
    assert len(rows) == expected, (phase, len(rows), expected)
    keys = [(r['panel'], r['seed'], r['method']) for r in rows]
    assert len(keys) == len(set(keys))
    draws = {(r['panel'], r['seed']) for r in rows}
    for panel, seed in draws:
        group = [r for r in rows if r['panel'] == panel and r['seed'] == seed]
        digests = {r['input_digest'] for r in group}
        partitions = {json.dumps(r['partition'], sort_keys=True) for r in group}
        assert len(digests) == len(partitions) == 1
        partition = group[0]['partition']
        fit, cal = set(partition['selection_rows']), set(partition['calibration_rows'])
        assert not fit.intersection(cal) and fit.union(cal) == set(range(1000))
        assert len(fit) == len(cal) == 500
        if phase != 'audit':
            for track in ('known', 'discovery'):
                variants = [r for r in group if r['method'].startswith(track+'_') and r['status'] == 'ok']
                assert all(r['predictions']['point'] == variants[0]['predictions']['point'] for r in variants)
        assert all(r['count_coverage_identical'] for r in group if r['status'] == 'ok')
    return dict(records=len(rows), panel_draws=len(draws),
                independent_seeds_per_panel={panel: len({seed for name, seed in draws if name == panel})
                                             for panel in sorted({name for name, _ in draws})},
                panels_share_streams=phase != 'audit',
                failures=sum(r['status'] != 'ok' for r in rows),
                matched_partitions=True, fixed_mean_across_scores=phase != 'audit',
                count_support_preserves_coverage=True)


def main():
    results, validation = [], {}
    for phase in ('audit', 'development', 'confirmation'):
        rows, _ = load(phase)
        validation[phase] = validate(phase, rows)
        results.extend(summarize(phase, rows))
        if phase == 'audit':
            ok = [r for r in rows if r['status'] == 'ok']
            validation[phase].update(
                prior_input_mismatches=sum(not r['prior_input_matches'] for r in ok),
                prior_prediction_digest_mismatches=sum(not r['prior_digest_matches'] for r in ok),
                maximum_prior_metric_difference=max(abs(v) for r in ok for v in r['prior_metric_differences'].values()))
    (DIRECTORY/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    (DIRECTORY/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
