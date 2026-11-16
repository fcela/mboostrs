"""Independent scoring checks and the information boundary of the experiment."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from benches import bench_distributional as study
from benches.study_evidence import gaussian_scores


def test_scores_match_density_and_integrated_cdf_oracles():
    y = np.array([-1.0, 0.3, 2.5])
    mu = np.array([0.0, 0.5, 1.0])
    sigma = np.array([0.7, 1.2, 2.0])
    scores = gaussian_scores(y, mu, sigma)
    np.testing.assert_allclose(scores["log_score"], -norm.logpdf(y, mu, sigma))

    # CRPS is independently integrated from its definition, not its closed form.
    for index, (observed, location, scale) in enumerate(zip(y, mu, sigma, strict=True)):
        left = quad(lambda x, m=location, s=scale: norm.cdf(x, m, s)**2,
                    -np.inf, observed)[0]
        right = quad(lambda x, m=location, s=scale: (1 - norm.cdf(x, m, s))**2,
                     observed, np.inf)[0]
        assert scores["crps"][index] == pytest.approx(left + right, abs=1e-9)


def test_invalid_predictions_cannot_become_successful_scores():
    with pytest.raises(ValueError, match="aligned"):
        gaussian_scores([1, 2], [1], [1, 1])
    for scale in ([1, 0], [1, -1], [1, np.nan]):
        with pytest.raises(ValueError, match="positive scale"):
            gaussian_scores([1, 2], [1, 2], scale)


def test_outer_responses_do_not_choose_the_configuration(monkeypatch):
    original_sample = study.sample

    def fake_fit(library, training, prediction_data, configuration, track, seed):
        # A deterministic surrogate makes the data boundary test inexpensive.
        # Its fit reads training outcomes but cannot use prediction outcomes.
        mean = training["y"].mean() + configuration
        return (np.full(len(prediction_data["y"]), mean),
                np.full(len(prediction_data["y"]), 1.0), {})

    monkeypatch.setattr(study, "fit_predict", fake_fit)
    baseline = study.run("mboostrs", "additive", "additive", 0)

    def changed_outer_outcomes(seed, n, scenario):
        frame, mean, sigma = original_sample(seed, n, scenario)
        if n == 2000:
            frame["y"] = frame["y"] + 100
        return frame, mean, sigma

    monkeypatch.setattr(study, "sample", changed_outer_outcomes)
    changed = study.run("mboostrs", "additive", "additive", 0)
    assert baseline["configuration"] == changed["configuration"]
    assert baseline["inner"] == changed["inner"]
    assert baseline["output_digest"] == changed["output_digest"]
    assert baseline["metrics"]["log_score"] != changed["metrics"]["log_score"]


def test_failed_inner_fold_disqualifies_configuration(monkeypatch):
    def fake_fit(library, training, prediction_data, configuration, track, seed):
        if configuration == 0:
            raise ValueError("singular candidate")
        return (np.zeros(len(prediction_data["y"])),
                np.ones(len(prediction_data["y"])), {})

    monkeypatch.setattr(study, "fit_predict", fake_fit)
    result = study.run("mboostrs", "additive", "additive", 0)
    assert result["configuration"] != 0
    assert result["inner"][0]["log_score"] is None
    assert len(result["inner"][0]["errors"]) == 3
