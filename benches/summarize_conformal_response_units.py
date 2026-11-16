"""Verify complete issue #652 evidence and derive the book's result tables."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json

import numpy as np

from benches.audit_conformal_response_units import DIRECTORY, inputs

TOLERANCE = 1e-10


def load(phase):
    manifest = json.loads((DIRECTORY/'archives.json').read_text())
    compressed = (DIRECTORY/(phase+'.json.gz')).read_bytes()
    payload = gzip.decompress(compressed)
    assert hashlib.sha256(compressed).hexdigest() == manifest[phase]['compressed_sha256']
    assert hashlib.sha256(payload).hexdigest() == manifest[phase]['uncompressed_sha256']
    return json.loads(payload)


def validate(before, after):
    assert len(before['results']) == len(after['results']) == 132
    assert before['build_info'] == after['build_info']
    assert before['protocol']['sha256'] == after['protocol']['sha256']
    keys = ('frontend', 'transform', 'method', 'target', 'budget', 'route', 'input_digest')
    maximum, affine = 0., []
    data, _ = inputs()
    for old, new in zip(before['results'], after['results'], strict=True):
        assert all(old[k] == new[k] for k in keys)
        if new.get('expected_refusal'):
            assert old['status'] == new['status'] == 'refused'
            assert 'scale changes between folds' in new['error']
            continue
        assert old['reference'] == new['reference']
        assert new['status'] == 'ok'
        errors = [float(np.max(np.abs(np.asarray(new['observed'][name])-new['reference'][name])))
                  for name in ('lower', 'upper', 'predicted')]
        if new['quantile'] is not None:
            errors.append(abs(new['quantile']-new['reference']['quantile']))
        error = max(errors)
        assert error == new['max_absolute_error'] and error < TOLERANCE
        assert new['fitted_mean_available'] == (new['transform'] == 'bare')
        maximum = max(maximum, error)
        assert len(new['inner_cv']) == len(new['reference']['inner_cv'])
        for actual, expected in zip(new['inner_cv'], new['reference']['inner_cv'], strict=True):
            assert actual['mode'] == 'refit'
            assert actual['training_rowids'] == expected['rowids']
            assert actual['budget'] == expected['budget']
            a, b = np.asarray(actual['aggregate']), np.asarray(expected['aggregate'])
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            assert actual['optimum'] == int(np.argmin(a[1:]))+1
            assert expected['optimum'] == int(np.argmin(b[1:]))+1
        if (new['route'] == 'internal' and new['budget'] == 30 and
                new['transform'] in ('center', 'scale') and new['method'] in ('split', 'cqr')):
            proper_y = data['y'][new['reference']['partitions']['training_rows']]
            sd = proper_y.std(ddof=1) if new['transform'] == 'scale' else 1.
            mapped = {k: np.asarray(old['observed'][k])*sd+proper_y.mean()
                      for k in ('lower', 'upper', 'predicted')}
            endpoint_error = max(float(np.max(np.abs(mapped[k]-new['reference'][k])))
                                 for k in ('lower', 'upper'))
            affine.append({**{k: new[k] for k in keys[:-1]},
                           'baseline_after_affine_unit_conversion': endpoint_error,
                           'corrected_error': error})
    summaries = []
    for method, route in itertools.product(('split', 'cv+', 'cqr'), ('internal', 'external')):
        def select(doc, method=method, route=route):
            return [r for r in doc['results'] if r['method'] == method and r['route'] == route
                    and not r.get('expected_refusal')]
        old, new = select(before), select(after)
        if not new:
            continue
        summaries.append(dict(method=method, route=route, cases=len(new),
            baseline_failures=sum(r['status'] == 'failed' for r in old),
            baseline_disagreements=sum(r.get('max_absolute_error', 0) > TOLERANCE for r in old),
            baseline_max_error=max(r.get('max_absolute_error', 0) for r in old),
            corrected_failures=0, corrected_max_error=max(r['max_absolute_error'] for r in new)))
    facts = dict(total_cases=132, supported_cases=120, fixed_cases=112, tuned_cases=8,
        named_refusals=12, maximum_corrected_error=maximum, corrected_failures=0,
        baseline_failures=sum(r['status'] == 'failed' for r in before['results']),
        baseline_disagreements=sum(r.get('max_absolute_error', 0) > TOLERANCE for r in before['results']),
        baseline_max_error=max(r.get('max_absolute_error', 0) for r in before['results']),
        identical_reference_arrays=True, identical_native_binary=True,
        original_response_units=True, inner_risk_comparison_tolerance=1e-12,
        uncertainty='Not estimated: one deterministic fixture shared by every case.',
        metadata_note='The archived generic evidence helper uncertainty block is boilerplate; '
                      'the frozen protocol and this explicit scope govern interpretation. '
                      'No independent draws, Monte Carlo standard errors or timing comparison were run.')
    return summaries, affine, facts


def main():
    summaries, affine, facts = validate(load('baseline'), load('corrected'))
    for name, value in (('summary', summaries), ('affine_diagnostic', affine), ('validation', facts)):
        (DIRECTORY/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
