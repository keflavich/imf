import pytest
import numpy as np
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
