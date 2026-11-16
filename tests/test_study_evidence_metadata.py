"""Uncertainty descriptions must follow the actual evidence design (#655)."""
from benches import study_evidence


def test_simulation_default_and_deterministic_override(monkeypatch):
    import mboostrs as mb

    monkeypatch.setattr(study_evidence, 'machine_metadata', lambda: {
        'commit': 'fixture', 'working_tree': 'clean', 'platform': 'test',
        'versions': {}, 'threads': {}})
    monkeypatch.setattr(mb, 'build_info', lambda: {'build_id': 'test'})
    simulation = study_evidence.evidence_document({}, 5, [])
    assert simulation['uncertainty'] == {
        'unit': 'one independent training and test draw; pair methods within a draw',
        'summary': 'mean and Monte Carlo standard error across draws',
        'timing': 'elapsed whole-workflow costs, not controlled implementation speed ratios'}
    contract = dict(study_evidence.DETERMINISTIC_UNCERTAINTY)
    deterministic = study_evidence.evidence_document({}, {'fixtures': 1}, [], uncertainty=contract)
    assert deterministic['uncertainty'] == study_evidence.DETERMINISTIC_UNCERTAINTY
    assert 'no sampling standard error' in deterministic['uncertainty']['summary']
    contract['summary'] = 'changed after capture'
    assert deterministic['uncertainty']['summary'] != contract['summary']
    assert study_evidence.evidence_document({}, 1, [], uncertainty={})['uncertainty'] == {}


def test_deterministic_audits_supply_contract(monkeypatch, tmp_path):
    from benches import audit_conformal_predictor_coding, audit_conformal_response_units

    captured = []
    original = study_evidence.evidence_document

    def record(*args, **kwargs):
        captured.append(kwargs['uncertainty'])
        return original(*args, **kwargs)

    for module in (audit_conformal_predictor_coding, audit_conformal_response_units):
        monkeypatch.setattr(module, 'evidence_document', record)
        monkeypatch.setattr(module, 'cases', lambda: iter(()))
        module.run(tmp_path / (module.__name__ + '.json'), 'corrected')
    assert captured == [study_evidence.DETERMINISTIC_UNCERTAINTY]*2


def test_fixed_competitor_versions_are_recorded(monkeypatch):
    import mboostrs as mb

    monkeypatch.setattr(study_evidence, "machine_metadata", lambda: {
        "commit": "fixture", "working_tree": "clean", "platform": "test", "versions": {}, "threads": {}})
    monkeypatch.setattr(mb, "build_info", lambda: {"build_id": "test"})
    monkeypatch.setattr(study_evidence.importlib.metadata, "version", lambda name: name + "-fixture-version")
    document = study_evidence.evidence_document({}, 1, [])
    assert document["versions"]["catboost"] == "catboost-fixture-version"
    assert document["versions"]["lightgbm"] == "lightgbm-fixture-version"


def test_post_run_inventory_does_not_infer_a_version_from_changed_files(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from benches import inventory_benchmark_runtime as runtime

    (tmp_path / "model.py").write_text("value=1\n")
    distribution = SimpleNamespace(version="test", files=[__import__("pathlib").Path("model.py")],
                                   locate_file=lambda entry: tmp_path / entry)
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", lambda name: distribution)
    later_change = runtime.inventory_distribution("fixture", 0)
    assert later_change["inferred_measurement_version"] is None
    assert not later_change["inference_supported_by_unchanged_installation"]
    prior_change = runtime.inventory_distribution("fixture", float("inf"))
    assert prior_change["inferred_measurement_version"] == "test"
