import pytest
import numpy as np

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

def test_kroupa_integral():
    for mlow in (0.01, 0.08, 0.1, 0.5, 1.0):
        for mhigh in (0.02, 0.08, 0.4, 0.5, 1.0):
            try:
                num = kroupa.integrate(mlow, mhigh, numerical=True)[0]
                anl = kroupa.integrate(mlow, mhigh, numerical=False)[0]
            except ValueError:
                continue
            np.testing.assert_almost_equal(num, anl)

    for mlow in (0.01, 0.08, 0.1, 0.5, 1.0):
        for mhigh in (0.02, 0.08, 0.4, 0.5, 1.0):
            try:
                num = kroupa.m_integrate(mlow, mhigh, numerical=True)[0]
                anl = kroupa.m_integrate(mlow, mhigh, numerical=False)[0]
            except ValueError:
                continue
            print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
            np.testing.assert_almost_equal(num, anl)


def test_chabrier_integral():
    for mlow in (0.033, 0.5, 1, 1.5, 3):
        for mhigh in (0.05, 0.5, 1, 1.5, 3.0):
            try:
                num = chabrier2005.integrate(mlow, mhigh, numerical=True)[0]
                anl = chabrier2005.integrate(mlow, mhigh, numerical=False)[0]
            except ValueError:
                continue
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

