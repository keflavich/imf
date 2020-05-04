import pytest
import numpy as np
import scipy.interpolate
from .. import imf
from .. import distributions as D


def test_lognorm():
    ln = D.LogNormal(1, 1)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    assert (np.abs(ln.ppf(ln.cdf(2))-2)<1e-5)
    # test that ppf is inverse of cdf

    for i in range(10):
        N = 100000
        mean = np.random.uniform(0.1,10)
        sig = np.random.uniform(0.1,10)
        ln2 = D.LogNormal(mean, sig)
        samp = ln2.rvs(N)
        # check that the means and sigmas are correct
        assert(np.abs(np.log(samp).mean()-np.log(mean))< 0.01*sig)
        assert(np.abs(np.log(samp).std()-sig)< 0.01*sig)

def test_broken_plaw():
    ln = D.BrokenPowerLaw([-2, -1.1, -3], [0.1, 1, 2, 100])
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    assert (np.abs(ln.ppf(ln.cdf(0.5))-0.5)<1e-5)
    assert (np.abs(ln.ppf(ln.cdf(1.5))-1.5)<1e-5)
    assert (np.abs(ln.ppf(ln.cdf(2.5))-2.5)<1e-5)


def test_distr():
    ln = D.TruncatedLogNormal(1, 1, 2, 3)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    assert (np.abs(ln.ppf(ln.cdf(2.5))-2.5)<1e-5)

    ln = D.PowerLaw(-2, 2, 6)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    assert (np.abs(ln.ppf(ln.cdf(3))-3)<1e-5)

def test_composite():
    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1, 1, 2, 3),
        D.PowerLaw(-2, 3, 4),
        D.TruncatedLogNormal(1, 1, 4, 5),
        D.PowerLaw(-2, 5, np.inf)
    ])
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    assert (np.abs(ln.ppf(ln.cdf(2.5))-2.5)<1e-5)
    assert (np.abs(ln.ppf(ln.cdf(3.5))-3.5)<1e-5)
    assert (np.abs(ln.ppf(ln.cdf(4.5))-4.5)<1e-5)
    assert (np.abs(ln.ppf(ln.cdf(5.5))-5.5)<1e-5)


def test_bounds():
    left, right = 1, 2
    tleft, tright = 0.5, 3
    ln = D.TruncatedLogNormal(1, 1, left, right)
    assert (ln.pdf(tleft) == 0)
    assert (ln.pdf(tright) == 0)
    assert (ln.cdf(tleft) == 0)
    assert (ln.cdf(tright) == 1)

    ln = D.PowerLaw(-3, left, right)
    assert (ln.pdf(tleft) == 0)
    assert (ln.pdf(tright) == 0)
    assert (ln.cdf(tleft) == 0)
    assert (ln.cdf(tright) == 1)

    ln = D.BrokenPowerLaw(
        [-2, -1.1, -3],
        [left, .6 * left + .3 * right, .3 * left + .6 * right, right])
    assert (ln.pdf(tleft) == 0)
    assert (ln.pdf(tright) == 0)
    assert (ln.cdf(tleft) == 0)
    assert (ln.cdf(tright) == 1)

    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1, 1, left, .75 * left + .25 * right),
        D.PowerLaw(-2, .75 * left + .25 * right, .5 * left + .5 * right),
        D.TruncatedLogNormal(1, 1, .5 * left + .5 * right,
                             .25 * left + .75 * right),
        D.PowerLaw(-2, .25 * left + .75 * right, right)
    ])
    assert (ln.pdf(tleft) == 0)
    assert (ln.pdf(tright) == 0)
    assert (ln.cdf(tleft) == 0)
    assert (ln.cdf(tright) == 1)


def test_integral():
    # test that the numerically integrated pdf is within 3 sigma of 1
    # for different kind of pdfs

    def checker(x):
        assert (np.abs(1 - x[0]) < 3 * x[1])

    left, right = 2, 3
    ln = D.TruncatedLogNormal(1, 1, left, right)
    checker(scipy.integrate.quad(lambda x: ln.pdf(x), left, right))

    ln = D.PowerLaw(-2, left, right)
    checker(scipy.integrate.quad(lambda x: ln.pdf(x), left, right))

    ln = D.BrokenPowerLaw(
        [-2, -1.1, -3],
        [left, .6 * left + .3 * right, .3 * left + .6 * right, right])
    checker(scipy.integrate.quad(lambda x: ln.pdf(x), left, right))

    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1, 1, left, .75 * left + .25 * right),
        D.PowerLaw(-2, .75 * left + .25 * right, .5 * left + .5 * right),
        D.TruncatedLogNormal(1, 1, .5 * left + .5 * right,
                             .25 * left + .75 * right),
        D.PowerLaw(-2, .25 * left + .75 * right, right)
    ])
    checker(scipy.integrate.quad(lambda x: ln.pdf(x), left, right))
