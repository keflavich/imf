import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

class Distribution:
    """ The main class describing the distributions, to be inherited"""
    def __init__(self):
        self.m1 = 0  # edges of the support of the pdf
        self.m2 = np.inf
        pass

    def pdf(self, x):
        """ Return the Probability density function"""
        pass

    def cdf(self, x):
        """ Cumulative distribtuion function """
        pass

    def rvs(self, N):
        """ Generate random sample """
        pass

    def ppf(self, x):
        # inverse cdf
        raise RuntimeError('not implemented')
        pass


class LogNormal(Distribution):
    def __init__(self, mu, sig):
        """
        Define the Lognormal with distribution
        ~ 1/x exp( -1/2 *(log(x)-log(mu))^2/sig^2)
        I.e. the mean of log of the samples will be log(mu)
        and the stddev of log of the samples will be sig
        """
        self.m1 = 0
        self.m2 = np.inf
        self.d = scipy.stats.lognorm(s=sig, scale=mu)

    def pdf(self, x):
        return self.d.pdf(x)

    def cdf(self, x):
        return self.d.cdf(x)

    def rvs(self, N):
        return self.d.rvs(N)

    def ppf(self, x):
        return self.d.ppf(x)


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
        return (self.d.cdf(np.clip(x, self.m1, self.m2)) -
                self.d.cdf(self.m1)) / self.norm

    def rvs(self, N):
        x = np.random.uniform(0, 1, size=N)
        return self.ppf(x)

    def ppf(self, x0):
        x = np.asarray(x0)
        cut1 = self.d.cdf(self.m1)
        cut2 = self.d.cdf(self.m2)
        ret = self.d.ppf(x * (cut2 - cut1) + cut1)
        ret = np.asarray(ret)
        ret[(x < 0) | (x > 1)] = np.nan
        return ret


class PowerLaw(Distribution):
    def __init__(self, slope, m1, m2):
        """ Power law with slope slope in the interval m1,m2 """
        self.slope = slope
        self.m1 = float(m1)
        self.m2 = float(m2)
        assert (m1 < m2)
        assert (m1 > 0)
        assert (m1 != -1)

    def pdf(self, x):
        if self.slope == -1:
            return (x**self.slope / (np.log(self.m2 / self.m1)) *
                    (x >= self.m1) * (x <= self.m2))
        else:
            return x**self.slope * (self.slope + 1) / (
                self.m2**(self.slope + 1) -
                self.m1**(self.slope + 1)) * (x >= self.m1) * (x <= self.m2)

    def cdf(self, x):
        if self.slope == -1:
            raise RuntimeError('Not implemented')
        else:
            return (np.clip(x, self.m1, self.m2)**(self.slope + 1) -
                    (self.m1**(self.slope + 1))) / (self.m2**(self.slope + 1) -
                                                    self.m1**(self.slope + 1))

    def rvs(self, N):
        x = np.random.uniform(size=N)
        return self.ppf(x)

    def ppf(self, x0):
        x = np.asarray(x0)
        if self.slope == -1:
            ret = np.exp(x * np.log(self.m2 / self.m1)) * self.m1
        else:
            ret = (x *
                   (self.m2**(self.slope + 1) - self.m1**(self.slope + 1)) +
                   self.m1**(self.slope + 1))**(1. / (self.slope + 1))
        ret = np.asarray(ret)
        ret[(x < 0) | (x > 1)] = np.nan
        return ret


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
        self.slopes = slopes
        self.breaks = breaks
        self._calcpows()
        self._calcweights()

    @property
    def m1(self):
        return self.breaks[0]

    @m1.setter
    def m1(self, value):
        self.breaks[0] = value
        self._calcpows()
        self._calcweights()

    @property
    def m2(self):
        return self.breaks[-1]

    @m2.setter
    def m2(self, value):
        self.breaks[-1] = value
        self._calcpows()
        self._calcweights()

    def _calcpows(self):
        if not (len(self.slopes) == len(self.breaks) - 1):
            raise ValueError(
                'The length of array of slopes must be equal to length of ' +
                'array of break points minus 1')
        if not ((np.diff(self.breaks) > 0).all()):
            raise ValueError('Power law break-points must be monotonic')
        nsegm = len(self.slopes)
        pows = []
        for ii in range(nsegm):
            pows.append(
                PowerLaw(self.slopes[ii], self.breaks[ii],
                         self.breaks[ii + 1]))
        self.pows = pows
        self.nsegm = nsegm

    def _calcweights(self):
        nsegm = len(self.slopes)
        weights = [1]
        for ii in range(1, nsegm):
            rat = self.pows[ii].pdf(self.breaks[ii]) / self.pows[ii - 1].pdf(
                self.breaks[ii])
            weights.append(weights[-1] / rat)
        weights = np.array(weights)
        self.weights = weights / np.sum(weights)  # relative normalizations
        self.nsegm = nsegm

    def pdf(self, x):
        x1 = np.asarray(x)
        ret = np.atleast_1d(x1) * 0.
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 >= self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = self.weights[ii] * self.pows[ii].pdf(x1[xind])
        return ret.reshape(x1.shape)

    def cdf(self, x):
        x1 = np.asarray(x)
        ret = np.atleast_1d(x1) * 0.
        cums = np.r_[[0], np.cumsum(self.weights)]
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 >= self.breaks[ii])
            if np.any(xind):
                ret[xind] = cums[ii] + self.weights[ii] * self.pows[ii].cdf(
                    x1[xind])

        xind = x1 >= self.breaks[-1]

        if np.any(xind):
            ret[xind] = 1

        return ret.reshape(x1.shape)

    def rvs(self, N):
        Ns = np.random.multinomial(N, self.weights)
        ret = []
        for ii in range(self.nsegm):
            if Ns[ii] > 0:
                ret.append(self.pows[ii].rvs(Ns[ii]))
        return np.concatenate(ret)

    def ppf(self, x0):
        x = np.asarray(x0)
        x1 = np.atleast_1d(x)
        edges = np.r_[[0], np.cumsum(self.weights)]
        # edges of powerlaw in CDF scale from 0 to 1
        pos = np.digitize(x1, edges)  # bin positions, 1 is the leftmost
        pos = np.clip(pos, 1, self.nsegm)
        #  we can get zeros here if input is corrupt
        left = edges[pos - 1]
        w = self.weights[pos - 1]
        x2 = np.clip((x1 - left) / w, 0, 1)  # mapping to 0,1 on the segment

        # must force float b/c int dtypes can result in truncation
        ret = np.zeros_like(x1, dtype='float')
        for ii in range(x.size):
            ret[ii] = self.pows[pos[ii] - 1].ppf(x2[ii])

        isnan = (x1 < 0) | (x1 > 1)
        if any(isnan):
            ret[isnan] = np.nan
        return ret.reshape(x.shape)


class KoenConvolvedPowerLaw(Distribution):
    """Error-convolved power law.

    A power law over the mass range (m1,m2) with slope -(gamma-1) convolved with
    a normal distribution of width sigma, as described in Koen & Kondlo 2009.
    This implementation calculates the PDF and CDF of the distribution at npts
    evenly log-spaced points and interpolates between the results.

    Parameters
    ----------
    m1: float
        Lower mass bound of the power law.
    m2: float
        Upper mass bound of the power law.
    gamma: float
        Determines the slope of the power law in log space. Should be != 0.
    sigma: float
        Width of the Gaussian convolved with the power law. Should be > 0.
    npts: float
        Number of evenly log-spaced points at which the distribution 
        will be evaluated.
    """
    def __init__(self,m1,m2,gamma,sigma,npts):
        self.m1 = m1
        self.m2 = m2
        self.gamma = gamma
        self.sigma = sigma
        self.points = self._make_points(npts)
        self._pdf = self._pre_integrate(False)
        self._pdf_interpolator = PchipInterpolator(self.points,self._pdf)
        self._cdf = self._pre_integrate(True)
        self._cdf_interpolator = PchipInterpolator(self.points,self._cdf)
        self._ppf_interpolator = PchipInterpolator(self._cdf,self.points)

    def _make_points(self,n_pts):
        #points to interpolate between when calling the distribution
        infMax = ~np.isfinite(self.m2)
        if infMax:
            points = np.geomspace(self.m1,1000*self.m1,n_pts-1)
            points = np.append(points,np.inf)
        else:
            points = np.geomspace(self.m1,self.m2,n_pts)[:-1]
            extras = np.linspace(self.m2-3*self.sigma,self.m2,8)
            ext_inds = np.searchsorted(points,extras,side='left')
            points = np.append(points,extras[ext_inds == len(points)])
        return points

    def _mirror_steps(self):
        #Sub-intervals for the integration to capture small changes at both ends
        x = np.geomspace(self.m1,self.m2,100)
        mir_x = self.m2-(x[::-1]-self.m1)
        dx = x[1:]-x[:-1]
        cutoff = min(self.sigma,1) #set a ceiling dx of 1
        break1 = np.searchsorted(dx,cutoff)
        break2 = np.searchsorted(-dx[::-1],-cutoff)
        xpt = x[break1]
        mirxpt = mir_x[break2]
        x1, x2 = min(xpt,mirxpt), max(xpt,mirxpt)
        x = np.append(x[x < x1],np.linspace(x1,x2,
                                            int((x2-x1)/cutoff)))
        x = np.append(x,mir_x[mir_x > x2])
        return x

    def _integrand(self,x,y,integral_form):
        '''
        Implements equations (3) and (5) from KK09.
        '''
        if integral_form:
            #equation 5
            coef = (1 / (self.sigma * np.sqrt(2 * np.pi) * (
                self.m1**-self.gamma - self.m2**-self.gamma)))
            ret = ((self.m1**-self.gamma - x**-self.gamma) * np.exp(
            (-1 / 2) * ((y - x) / self.sigma)**2))
            return coef*ret
        else:
            #equation 3
            coef = (self.gamma / ((self.sigma * np.sqrt(2 * np.pi)) *
                                  ((self.m1**-self.gamma) - (self.m2**-self.gamma))))
            ret = (x**-(self.gamma + 1)) * np.exp(-.5 * ((y - x) / self.sigma)**2)
            return coef*ret

    def _pre_integrate(self,integral_form):
        steps = self._mirror_steps()
        results = []
        for pt in self.points:
            chunks = []
            for i in range(len(steps)-1):
                l,u = steps[i],steps[i+1]
                area = quad(self._integrand,l,u,args=(pt,integral_form))[0]
                chunks.append(area)
            if integral_form:
                results.append(np.sum(chunks)+
                               scipy.stats.norm.cdf((pt - self.m2) / self.sigma))
            else:
                results.append(np.sum(chunks))
        results = np.array(results)
        return results

    def pdf(self,x):
        ret = self._pdf_interpolator(x,extrapolate=False)
        return ret

    def cdf(self,x):
        ret = self._cdf_interpolator(x,extrapolate=False)
        return ret
        
    def rvs(self,N):
        samp = np.random.uniform(min(self._cdf),max(self._cdf),size=N)
        return self.ppf(samp)

    def ppf(self,x):
        ret = self._ppf_interpolator(x,extrapolate=False)
        return ret


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

        self.m1 = breaks[0]  # leftmost edge
        self.m2 = breaks[-1]  # rightmost edge

        # check that edges of intervals match
        for ii in range(1, nsegm):
            assert (distrs[ii].m1 == distrs[ii - 1].m2)

        for ii in range(1, nsegm):
            rat = distrs[ii].pdf(breaks[ii]) / distrs[ii - 1].pdf(breaks[ii])
            # relative normalization of next pdf to the previous one so they
            # join without a break
            weights.append(weights[-1] / rat)
        weights = np.array(weights)
        self.breaks = breaks
        self.weights = weights / np.sum(weights)
        # these are relative weights  of each pdf
        self.nsegm = nsegm

    def pdf(self, x):
        x1 = np.asarray(x)
        ret = np.atleast_1d(x1 * 0.)
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 >= self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = self.weights[ii] * self.distrs[ii].pdf(x1[xind])
        return ret.reshape(x1.shape)

    def cdf(self, x):
        x1 = np.asarray(x)
        ret = np.atleast_1d(x1 * 0.)
        cums = np.r_[[0], np.cumsum(self.weights)]
        for ii in range(self.nsegm):
            xind = (x1 < self.breaks[ii + 1]) & (x1 >= self.breaks[ii])
            if xind.sum() > 0:
                ret[xind] = cums[ii] + self.weights[ii] * self.distrs[ii].cdf(
                    x1[xind])
        xind = x1 >= self.breaks[-1]
        if xind.sum():
            ret[xind] = 1
        return ret.reshape(x1.shape)

    def rvs(self, N):
        Ns = np.random.multinomial(N, self.weights)
        ret = []
        for ii in range(self.nsegm):
            if Ns[ii] > 0:
                ret.append(self.distrs[ii].rvs(Ns[ii]))
        ret = np.concatenate(ret)
        ret = np.random.permutation(ret)  # permutation
        return ret

    def ppf(self, x0):
        x = np.asarray(x0)
        x1 = np.atleast_1d(x)
        edges = np.r_[[0], np.cumsum(self.weights)]
        pos = np.digitize(x1, edges)
        pos = np.clip(pos, 1, self.nsegm)  # if input is <0 or >1
        left = edges[pos - 1]
        w = self.weights[pos - 1]
        x2 = np.clip((x1 - left) / w, 0, 1)  # mapping to 0,1 on the segment
        ret = np.zeros_like(x1)
        for ii in range(x.size):
            ret[ii] = self.distrs[pos[ii] - 1].ppf(x2[ii])
        ret[(x1 < 0) | (x1 > 1)] = np.nan
        return ret.reshape(x.shape)
