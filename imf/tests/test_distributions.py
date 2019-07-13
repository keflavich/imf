import pytest
import numpy as np
import scipy.interpolate
from .. import imf
from .. import distributions as D

def test_distr():
    ln = D.LogNormal(1,1)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)

    ln = D.TruncatedLogNormal(1,1,2,3)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)

    ln = D.PowerLaw(-2,2,6)
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)

    ln = D.BrokenPowerLaw([-2,-1.1,-3],[0.1,1,2,100])
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)

    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1,1,2,3),
        D.PowerLaw(-2,3,4),
        D.TruncatedLogNormal(1,1,4,5),
        D.PowerLaw(-2,5,np.inf)
        ])
    ln.pdf(1.)
    ln.cdf(1)
    ln.rvs(1000)

def test_bounds():
    left,right=1,2
    tleft, tright=0.5,3
    ln = D.TruncatedLogNormal(1,1,left,right)
    assert (ln.pdf(tleft)==0)
    assert (ln.pdf(tright)==0)
    assert (ln.cdf(tleft)==0)
    assert (ln.cdf(tright)==1)

    ln = D.PowerLaw(-3,left,right)
    assert (ln.pdf(tleft)==0)
    assert (ln.pdf(tright)==0)
    assert (ln.cdf(tleft)==0)
    assert (ln.cdf(tright)==1)

    ln = D.BrokenPowerLaw([-2,-1.1,-3],[left, .6 * left + .3 * right,
                                            .3 * left + .6 * right, right])
    assert (ln.pdf(tleft)==0)
    assert (ln.pdf(tright)==0)
    assert (ln.cdf(tleft)==0)
    assert (ln.cdf(tright)==1)

    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1,1,left,.75*left+.25*right),
        D.PowerLaw(-2,.75*left+.25*right,.5*left+.5*right),
        D.TruncatedLogNormal(1,1,.5*left+.5*right,.25*left+.75*right),
        D.PowerLaw(-2,.25*left+.75*right,right)
        ])
    assert (ln.pdf(tleft)==0)
    assert (ln.pdf(tright)==0)
    assert (ln.cdf(tleft)==0)
    assert (ln.cdf(tright)==1)


def test_integral():
    def checker(x):
        assert(np.abs(1-x[0])<3*x[1])
    left,right=2,3
    ln = D.TruncatedLogNormal(1,1,left,right)
    checker(scipy.integrate.quad(lambda x:ln.pdf(x),
                             left, right))

    ln = D.PowerLaw(-2,left, right)
    checker(scipy.integrate.quad(lambda x:ln.pdf(x),
                             left, right))
 
    ln = D.BrokenPowerLaw([-2,-1.1,-3],[left, .6 * left + .3 * right,
                                            .3 * left + .6 * right, right])
    checker(scipy.integrate.quad(lambda x:ln.pdf(x),
                             left, right))


    ln = D.CompositeDistribution([
        D.TruncatedLogNormal(1,1,left,.75*left+.25*right),
        D.PowerLaw(-2,.75*left+.25*right,.5*left+.5*right),
        D.TruncatedLogNormal(1,1,.5*left+.5*right,.25*left+.75*right),
        D.PowerLaw(-2,.25*left+.75*right,right)
        ])
    checker(scipy.integrate.quad(lambda x:ln.pdf(x),
                             left, right))
    
