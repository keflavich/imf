import numpy as np
import scipy.interpolate
from .. import distributions as D

np.random.seed(1)


def sampltest(distr, left=None, right=None, bounds=None):

    # check that mean and stddev from the generated sample
    # match what we get from integrating the PDF

    def FF1(x):
        return distr.pdf(x) * x

    def FF2(x):
        return distr.pdf(x) * x**2

    if left is None:
        left = 0
    if right is None:
        right = np.inf
    if bounds is None:
        mom1, _ = scipy.integrate.quad(FF1, left, right)
        mom2, _ = scipy.integrate.quad(FF2, left, right)
    else:
        mom1, mom2 = 0, 0
        for curb in bounds:
            cmom1, _ = scipy.integrate.quad(FF1, curb[0], curb[1])
            cmom2, _ = scipy.integrate.quad(FF2, curb[0], curb[1])
            mom1 += cmom1
            mom2 += cmom2

    std = np.sqrt(mom2 - mom1**2)
    assert (mom2 > mom1**2)
    N = int(1e6)
    samps = distr.rvs(N)
    assert ((samps.mean() - mom1) < 5 * std / np.sqrt(N))
    assert ((samps.std() - std) < 20 * std / np.sqrt(2 * (N - 1)))


def ppftest(distr):
    # test that ppf is inverse of cdf
    xs = np.random.uniform(0, 1, size=100)
    eps = 1e-5
    assert (np.all(np.abs(distr.cdf(distr.ppf(xs)) - xs) < eps))
    # test on scalar
    assert (np.abs(distr.cdf(distr.ppf(xs[0])) - xs[0]) < eps)
    assert (np.isnan(distr.ppf(-0.1)))
    assert (np.isnan(distr.ppf(1.1)))


def test_lognorm():
    ln = D.LogNormal(1, 1)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    ppftest(ln)
    sampltest(ln)

    for i in range(10):
        N = 100000
        mean = np.random.uniform(0.1, 10)
        sig = np.random.uniform(0.1, 10)
        ln2 = D.LogNormal(mean, sig)
        samp = ln2.rvs(N)
        # check that the means and sigmas are correct
        assert (np.abs(np.log(samp).mean() - np.log(mean)) < 0.01 * sig)
        assert (np.abs(np.log(samp).std() - sig) < 0.01 * sig)


def test_broken_plaw():
    ln = D.BrokenPowerLaw([-2, -1.1, -3], [0.1, 1, 2, 100])
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    ppftest(ln)
    sampltest(ln, 0.05, 120, bounds=[[0.05, 1], [1, 2], [2, 120]])
    # test values in each range
    assert (np.abs(ln.ppf(ln.cdf(0.5)) - 0.5) < 1e-5)
    assert (np.abs(ln.ppf(ln.cdf(1.5)) - 1.5) < 1e-5)
    assert (np.abs(ln.ppf(ln.cdf(2.5)) - 2.5) < 1e-5)


def test_distr():
    ln = D.TruncatedLogNormal(1, 1, 2, 3)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    ppftest(ln)
    sampltest(ln, 1, 4)
    ln = D.PowerLaw(-2, 2, 6)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)
    ppftest(ln)
    sampltest(ln, 1, 7)


def test_composite():
    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1, 1, 2, 3),
        D.PowerLaw(-2, 3, 4),
        D.TruncatedLogNormal(1, 1, 4, 5),
        D.PowerLaw(-3.5, 5, np.inf)
    ])
    ln.pdf(2.5)
    ln.cdf(2.5)
    ln.rvs(1000)
    ppftest(ln)
    # test values in each break
    assert (np.abs(ln.ppf(ln.cdf(2.5)) - 2.5) < 1e-5)
    assert (np.abs(ln.ppf(ln.cdf(3.5)) - 3.5) < 1e-5)
    assert (np.abs(ln.ppf(ln.cdf(4.5)) - 4.5) < 1e-5)
    assert (np.abs(ln.ppf(ln.cdf(5.5)) - 5.5) < 1e-5)

    sampltest(ln, 1, np.inf, bounds=[[1, 3], [3, 4], [4, 5], [5, np.inf]])
    ln1 = D.CompositeDistribution([
        D.TruncatedLogNormal(1, 1, 2, 3),
        D.PowerLaw(-2, 3, 4),
    ])
    # check the exact edges work
    assert (ln1.cdf(4) == 1)
    assert (ln1.cdf(2) == 0)


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


def integralcheck(distr, left, x, val):
    I, EI = scipy.integrate.quad(lambda y: distr.pdf(y), left, x)
    assert (np.abs(val - I) < 1e-6)


def integralcheck_many(distr, left, right):
    integralcheck(distr, left, right, 1)
    N = 100
    xs = np.random.uniform(left, right, size=N)
    for x in xs:
        integralcheck(distr, left, x, distr.cdf(x))


def test_integral():
    # test that the numerically integrated pdf is within 3 sigma of 1
    # for different kind of pdfs

    left, right = 2, 3
    distrs = [
        D.TruncatedLogNormal(1, 1, left, right),
        D.PowerLaw(-2, left, right),
        D.BrokenPowerLaw(
            [-2, -1.1, -3],
            [left, .6 * left + .3 * right, .3 * left + .6 * right, right]),
        D.CompositeDistribution([
            D.TruncatedLogNormal(1, 1, left, .75 * left + .25 * right),
            D.PowerLaw(-2, .75 * left + .25 * right, .5 * left + .5 * right),
            D.TruncatedLogNormal(1, 1, .5 * left + .5 * right,
                                 .25 * left + .75 * right),
            D.PowerLaw(-2, .25 * left + .75 * right, right)
        ])
    ]
    for curd in distrs:
        integralcheck_many(curd, left, right)
