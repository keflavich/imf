import numpy as np
import scipy.stats

class Distribution:
    """ The main class describing the distributions, to be inherited"""
    def __init__(self):
        self.m1=0 # edges of the support of the pdf
        self.m2=np.inf
        pass
    def pdf(self,x):
        """ Return the Probability density function"""
        pass
    def cdf(self, x):
        """ Cumulative distribtuion function """
        pass
    def rvs(self, N):
        """ Generate random sample """
        pass

class LogNormal(Distribution):
    def __init__(self, mu, sig):
        """
        Define the Lognormal with distribution
        ~ 1/x exp( -1/2 *(log(x/mu))^2/sig^2) """
        self.m1 = 0
        self.m2 = np.inf
        self.d = scipy.stats.lognorm(s=s, scale=mu)

    def pdf(self, x):
        return self.d.pdf(x)

    def cdf(self, x):
        return self.d.cdf(x)

    def rvs(self, N):
        return self.d.rvs(N)

class TruncatedLogNormal:
    def __init__(self, mu, sig, m1, m2):
        """ Standard log-normal but truncated in the interval m1,m2 """
        self.m1 = m1
        self.m2 = m2
        self.d = scipy.stats.lognorm(s=sig, scale=mu)
        self.norm = self.d.cdf(self.m2) - self.d.cdf(self.m1)

    def pdf(self, x):
        return self.d.pdf(x) * (x >= self.m1) * (x <= self.m2) / self.norm

    def cdf(self, x):
        return (self.d.cdf(x) - self.d.cdf(self.m1)) / self.norm

    def rvs(self, N):
        x = np.random.uniform(self.d.cdf(self.m1),self.d.cdf(self.m2),size=N)
        return self.d.ppf(x)

class PowerLaw(Distribution):
    def __init__(self, slope, m1, m2):
        """ Power law with slope slope in the interval m1,m2 """
        self.slope = slope
        self.m1 = m1
        self.m2 = m2
        assert(m1 < m2)
        assert(m1 > 0)
        assert(m1 != -1)

    def pdf(self, x):
        return x**self.slope * (self.slope + 1) / (
            self.m2**(self.slope + 1) -
            self.m1**(self.slope + 1)) * (x >= self.m1) * (x <= self.m2)

    def cdf(self, x):
        return (np.clip(x, self.m1, self.m2)**(self.slope + 1) -
                (self.m1**(self.slope + 1))) / (self.m2**(self.slope + 1) -
                                                self.m1**(self.slope + 1))
    def rvs(self, N):
        x = np.random.uniform(size=N)
        return(x * (self.m2**(self.slope+1)-self.m1**(self.slope+1))+self.m1**(self.slope+1))**(1./(self.slope+1))

class BrokenPowerLaw:
    def __init__(self, slopes, breaks):
        """
        Broken power-law with different slopes.

        Arguments:
        slopes: array
            Array of power-law slopes
        breaks: array
            Array of points/edges of powerlaw segments must be larger by one
            then the list of slopes
        """
        assert (len(slopes) == len(breaks) - 1)
        assert ((np.diff(breaks)>0).all())
        nsegm = len(slopes)
        pows = []
        for ii in range(nsegm):
            pows.append(PowerL(slopes[ii], breaks[ii], breaks[ii + 1]))
        weights = [1]
        for ii in range(1, nsegm):
            rat = pows[ii].pdf(breaks[ii]) / pows[ii - 1].pdf(breaks[i])
            weights.append(weights[-1] / rat)
        weights = np.array(weights)
        self.slopes = slopes
        self.breaks = breaks
        self.pows = pows
        self.weights = weights / np.sum(weights)
        self.nsegm = nsegm
        self.m1 = breaks[0]
        self.m2 = breaks[-1]

    def pdf(self, x):
        x1 = np.asarray(x)
        ret = x1 * 0.
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 > self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = self.weights[ii] * self.pows[ii].pdf(x1[xind])
        return ret

    def cdf(self, x):
        x1 = np.asarray(x)
        ret = x1 * 0.
        cums = np.r_[[0], np.cumsum(self.weights)]
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 > self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = cums[ii] + self.weights[ii] * self.pows[ii].cdf(
                    x1[xind])
        return ret
    def rvs(self, N):
        Ns = np.random.multinomial(N, self.weights)
        ret=[]
        for ii in range(self.nsegm):
            if Ns[ii]>0:
                ret.append(self.pows[ii].rvs(Ns[ii]))
        return np.concatenate(ret)

class CompositeDistribution(Distribution):
    def __init__(self, distrs):
        """ A Composite distribution that consists of several distributions
        that continuously join together

        Arguments:
        ----------
        distrs: list of Distributions
            The list of distributions. Their supports must not overlap
            and not have any gaps.

        Example:
        --------
        dd=distributions.CompositeDistribution([
          distributions.TruncatedLogNormal(0.3,0.3,0.08,1),
          distributions.PowerLaw(-2.55,1,np.inf)])
        dd.pdf(3)

        """
        nsegm = len(distrs)
        self.distrs = distrs
        weights = [1]
        breaks = [_.m1 for _ in self.distrs] + [self.distrs[-1].m2]

        self.m1 = breaks[0]
        self.m2 = breaks[-1]
        for ii in range(1,nsegm):
            assert(distrs[ii].m1==distrs[ii-1].m2)

        for ii in range(1, nsegm):
            rat = distrs[ii].pdf(breaks[ii]) / distrs[ii - 1].pdf(breaks[ii])
            weights.append(weights[-1] / rat)
        weights = np.array(weights)
        self.breaks = breaks
        self.weights = weights / np.sum(weights)
        self.nsegm = nsegm

    def pdf(self, x):
        x1 = np.asarray(x)
        ret = x1 * 0.
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 > self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = self.weights[ii] * self.distrs[ii].pdf(x1[xind])
        return ret

    def cdf(self, x):
        x1 = np.asarray(x)
        ret = x1 * 0.
        cums = np.r_[[0], np.cumsum(self.weights)]
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 > self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = cums[ii] + self.weights[ii] * self.distrs[ii].cdf(
                    x1[xind])
        return ret

    def rvs(self, N):
        Ns = np.random.multinomial(N, self.weights)
        ret=[]
        for ii in range(self.nsegm):
            if Ns[ii]>0:
                ret.append(self.distrs[ii].rvs(Ns[ii]))
        return np.concatenate(ret)
