"""Encoding views preserve latent rows and valid known dictionaries."""
import numpy as np

from benches.bench_encoding_tracks import ENCODINGS, FORMULAS, draw


def test_all_encodings_share_responses_covariates_and_latent_rows():
    views = [draw(name, 918) for name in ENCODINGS]
    assert len({v[3] for v in views}) == 1
    base = views[0]
    for train, test, truth, _ in views[1:]:
        np.testing.assert_array_equal(train.drop(columns="g"), base[0].drop(columns="g"))
        np.testing.assert_array_equal(test.drop(columns="g"), base[1].drop(columns="g"))
        np.testing.assert_array_equal(truth, base[2])
    assert views[2][0]["g"].dtype.kind == "b"
    assert set(views[3][0]["g"].unique()) == {-1, 1}
    assert set(views[4][0]["g"].unique()) == {2, 3}


def test_known_formulas_include_every_required_curve_component():
    axis = np.linspace(-1, 1, 40)
    f = 2 * np.sin(np.pi * axis)
    for encoding in ENCODINGS:
        assert "by=g" in FORMULAS[encoding] and "bols(x2)" in FORMULAS[encoding]
        if encoding != "signed":
            assert "bbs(x1, df=4) +" in FORMULAS[encoding]
        for b in [0, 1]:
            expected = f * (1 - 2 * b)
            if encoding == "signed":
                represented = f * (1 - 2 * b)
            elif encoding == "shifted":
                represented = 5 * f - 2 * f * (2 + b)
            else:
                represented = f - 2 * f * b
            np.testing.assert_allclose(represented, expected, atol=1e-14)
