"""The r_call helper skips a parity test only for a missing R package, and lets
an error raised by the R call fail the test (issue #405)."""

import pytest

from conftest import r_call


def _rpy2_loads():
    try:
        import rpy2.robjects  # noqa: F401
    except Exception:
        return False
    return True


needs_r = pytest.mark.skipif(not _rpy2_loads(), reason="rpy2 cannot load R")


@needs_r
def test_an_error_inside_the_r_call_fails_the_test():
    import rpy2.robjects as ro
    from rpy2.rinterface_lib.embedded import RRuntimeError

    with pytest.raises(RRuntimeError, match="boom from R"):
        r_call(ro.r, 'stop("boom from R")', packages=("base",))


@needs_r
def test_a_missing_r_package_skips_the_test():
    with pytest.raises(pytest.skip.Exception, match="mboostrs_no_such_package"):
        r_call(lambda: None, packages=("mboostrs_no_such_package",))


@needs_r
def test_the_r_result_is_returned_when_the_packages_are_installed():
    import rpy2.robjects as ro

    assert r_call(ro.r, "1 + 1", packages=("base",))[0] == 2
