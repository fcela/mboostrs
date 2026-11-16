"""Validate and summarize the complete deterministic issue #648 archives."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json

import numpy as np

from benches.audit_conformal_predictor_coding import DIRECTORY

TOLERANCE = 1e-10


def load(phase):
    manifest = json.loads((DIRECTORY/'archives.json').read_text())
    path = DIRECTORY/(phase+'.json.gz')
    compressed = path.read_bytes()
    payload = gzip.decompress(compressed)
    assert hashlib.sha256(compressed).hexdigest() == manifest[phase]['compressed_sha256']
    assert hashlib.sha256(payload).hexdigest() == manifest[phase]['uncompressed_sha256']
    return json.loads(payload)


def validate(before, after):
    assert len(before['results']) == len(after['results']) == 72
    assert before['build_info'] == after['build_info']
    keys = ('frontend', 'family', 'method', 'recipe', 'target', 'budget', 'input_digest')
    summaries = []
    maximum = 0.
    different_stops = []
    for old, new in zip(before['results'], after['results'], strict=True):
        assert all(old[key] == new[key] for key in keys)
        assert old['reference'] == new['reference']
        assert new['status'] == 'ok'
        errors = []
        for name in ('lower', 'upper', 'predicted'):
            actual, expected = np.asarray(new['observed'][name]), np.asarray(new['reference'][name])
            errors.append(float(np.max(np.abs(actual-expected))))
        error = max(errors)
        assert error == new['max_absolute_error'] and error < TOLERANCE
        maximum = max(maximum, error)
        for name, actual in new['observed_fields'].items():
            expected = new['reference'][name]
            if actual is None:
                assert expected is None
            else:
                np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
        assert len(new['inner_cv']) == len(new['reference']['inner_cv'])
        for index, (actual, expected) in enumerate(zip(new['inner_cv'], new['reference']['inner_cv'], strict=True)):
            assert actual['mode'] == 'refit'
            for name in ('training_rowids', 'family', 'budget', 'mode', 'fold_digest'):
                assert actual[name] == expected[name]
            a, b = np.asarray(actual['aggregate']), np.asarray(expected['aggregate'])
            # Preserve the exact positive-grid argmin rule in both paths.
            # Equivalent flat curves may differ by roundoff (tracked in #653).
            assert actual['optimum'] == int(np.argmin(a[1:]))+1
            assert expected['optimum'] == int(np.argmin(b[1:]))+1
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            if actual['optimum'] != expected['optimum']:
                different_stops.append({**{name: new[name] for name in keys[:-1]}, 'stage': index,
                    'actual_stop': actual['optimum'], 'reference_stop': expected['optimum'],
                    'maximum_risk_difference': float(np.max(np.abs(a-b))),
                    'interval_error': error})
    for family, frontend, method, mode in itertools.product(
            ('gaussian', 'poisson'), ('glm', 'gam'), ('cv+', 'cqr'), ('fixed', 'tuned')):
        def selected(document, family=family, frontend=frontend, method=method, mode=mode):
            return [r for r in document['results'] if r['family'] == family and r['frontend'] == frontend
                    and r['method'] == method and (r['budget'] == 'tune') == (mode == 'tuned')]
        old, new = selected(before), selected(after)
        if not new:
            continue
        summaries.append(dict(family=family, frontend=frontend, method=method, budget=mode,
            cases=len(new), baseline_failures=sum(r['status'] != 'ok' for r in old),
            baseline_mismatches=sum(r.get('max_absolute_error', float('inf')) > TOLERANCE for r in old),
            baseline_max_error=max((r.get('max_absolute_error', 0) for r in old), default=0),
            corrected_max_error=max(r['max_absolute_error'] for r in new), corrected_mismatches=0))
    facts = dict(cases=72, fixed_cases=64, tuned_cases=8,
        baseline_failures=sum(r['status'] != 'ok' for r in before['results']),
        baseline_mismatches=sum(r.get('max_absolute_error', float('inf')) > TOLERANCE for r in before['results']),
        corrected_failures=0, corrected_mismatches=0, maximum_corrected_error=maximum,
        identical_references=True, identical_native_binary=True,
        input_hashes_match=True, actual_inner_folds_and_risk_curves_match_reference=True,
        baseline_inner_modes=sorted({s['mode'] for r in before['results'] for s in r['inner_cv']}),
        corrected_inner_modes=sorted({s['mode'] for r in after['results'] for s in r['inner_cv']}))
    facts['different_stop_labels_on_equivalent_curves'] = different_stops
    facts['inner_risk_tolerance'] = 1e-12
    return summaries, facts


def main():
    summary, facts = validate(load('baseline'), load('corrected'))
    (DIRECTORY/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (DIRECTORY/'validation.json').write_text(json.dumps(facts, indent=2)+'\n')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
