import pytest
import numpy as np
import itertools

from .. import imf

from ..imf import kroupa, chabrier2005

@pytest.mark.parametrize('massfunc', imf.massfunctions.keys())
def test_mmax(massfunc):
    """
    Regression test for issue #4
    """

    if (not hasattr(imf.get_massfunc(massfunc), 'mmin')):
        pytest.skip("{0} doesn't have mmin defined".format(massfunc))

    c = imf.make_cluster(10000, mmax=1, mmin=0.01, massfunc=massfunc)

    assert c.max() <= 1

@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.08, 0.4, 0.5, 1.0, 120)))
def test_kroupa_integral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")
    num = kroupa.integrate(mlow, mhigh, numerical=True)[0]
    anl = kroupa.integrate(mlow, mhigh, numerical=False)[0]

    np.testing.assert_almost_equal(num, anl)
    if num != 0:
        assert anl != 0

@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.08, 0.4, 0.5, 1.0, 120)))
def test_kroupa_mintegral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")
    num = kroupa.m_integrate(mlow, mhigh, numerical=True)[0]
    anl = kroupa.m_integrate(mlow, mhigh, numerical=False)[0]
    print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    np.testing.assert_almost_equal(num, anl)
    if num != 0:
        assert anl != 0


@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.033, 0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.05, 0.08, 0.4, 0.5, 1.0, 120)))
def test_chabrier_integral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")

    num = chabrier2005.integrate(mlow, mhigh, numerical=True)[0]
    anl = chabrier2005.integrate(mlow, mhigh, numerical=False)[0]

    print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    np.testing.assert_almost_equal(num, anl)

    #for mlow in (0.01, 0.08, 0.1, 0.5, 1.0):
    #    for mhigh in (0.02, 0.08, 0.4, 0.5, 1.0):
    #        try:
    #            num = chabrier2005.m_integrate(mlow, mhigh, numerical=True)[0]
    #            anl = chabrier2005.m_integrate(mlow, mhigh, numerical=False)[0]
    #        except ValueError:
    #            continue
    #        print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    #        np.testing.assert_almost_equal(num, anl)


def test_krouva_val():
    assert np.allclose(kroupa(1), 0.09143468328964573, rtol=1e-4, atol=1e-4)
    assert np.allclose(kroupa(0.05), 5.615132028768199, rtol=1e-3, atol=1e-3)
    assert np.allclose(kroupa(1.5), 0.03598330658344697, rtol=1e-4, atol=1e-4)
    assert np.allclose(kroupa(3), 0.007306881750306478, rtol=1e-4, atol=1e-4)

def test_make_cluster():
    cluster = imf.make_cluster(1000)
    assert np.abs(sum(cluster) - 1000 < 100)
