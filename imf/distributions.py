import numpy as np
import scipy.stats
from scipy.integrate import quad,cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar
from scipy.special import hyp2f1

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
    a normal distribution of width sigma, as described in Koen & Kondlo (2009).
    This implementation calculates the PDF and CDF of the distribution at npts
    evenly log-spaced points and interpolates between the results.

    Arguments:
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


class PMF(Distribution):
    """Protostellar Mass Function.

    Creates a distribution for the Protostellar Mass Function (PMF) 
    corresponding to a supplied IMF and accretion history using the 
    formalism of McKee & Offner (2010).

    Arguments:
    ----------
    """
    def __init__(self,imf,m1,m2,
                 j,jf,scale_value,
                 n,tau):
        self.imf = imf
        self.m1 = m1
        self.m2 = m2
        self.j = j
        self.jf = jf
        self.scale_value = scale_value
        self.n = n
        self.tau = tau
        
        self._points = np.geomspace(min(self.m1,1e-3),self.m2,200)
        self._func_dict = None
        self._calculate('all')
        self._taper = False
        self._accelerating = False
        self._update_functions()

    def _make_interps(self,taper,accelerating):
        def pmf(mass,taper,accelerating):
            avg_time = self._average_time(taper,accelerating)

            def m_dot(mf,mass_):
                return self.scale_value * (mass_ / mf)**self.j * mf**self.jf

            def integrand(mf,mass_):
                if taper:
                    tf = self._tf(mf,taper)
                    def root_t(t,mf,mass_):
                        term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                        term2 = mass_**(1 - self.j) / self.scale_value / (1 - self.j) / mf**(self.jf - self.j)
                        prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                        prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                        return term1 - term2, prime_term1 - prime_term2

                    def taper_factor(mf,mass_):
                        sol = root_scalar(root_t,args=(mf,mass_),x0=0,fprime=True)
                        return 1 - (sol.root / tf)**self.n

                    t_factor = taper_factor(mf,mass_)
                    if accelerating:
                        tm = (1 - t_factor)**(1 / self.n) * tf
                        
                else:
                    t_factor = 1
                    if accelerating:
                        tm = mass_**(1 - self.j) / mf**(self.jf - self.j) / self.scale_value / (1 - self.j)
                a_factor = np.exp(-tm / self.tau / 1e6) if accelerating else 1
                
                return self.imf(mf) * mass_ / m_dot(mf,mass_) / t_factor * a_factor

            def integral(lolim,mass_,**kwargs):
                return scipy.integrate.quad(integrand,lolim,self.m2,args=(mass_),**kwargs)[0]

            ret = np.vectorize(integral)(np.where(self.m1 < mass, mass, self.m1),mass)
            return np.where(ret / avg_time > 0, ret / avg_time, 0) #ensure the PMF is always >= 0

        base = pmf(self._points,taper,accelerating)
        pdf = base / self._points
        cdf = cumulative_trapezoid(pdf,self._points,initial=0)
        cdf = np.concatenate((cdf,[max(cdf)]))
        cdf_points = np.concatenate(([min(self._points)],(self._points[1:]+self._points[:-1])/2,[self.m2]))
        zero_arg = np.argmin(np.diff(cdf))
        return (PchipInterpolator(self._points,pdf),
                PchipInterpolator(cdf_points,cdf),
                PchipInterpolator(cdf[:zero_arg+1],cdf_points[:zero_arg+1]))

    def _calculate(self,mode):
        not_ok = (self.j is None) | (self.jf is None) | (self.scale_value is None)
        if not_ok:
            raise ValueError('Cannot calculate a PMF without a history or all of (j, jf, scale_value)')
        else:
            pass

        keys = ['pdf','cdf','ppf']
        if mode == 'all':
            func_dict = {key: [] for key in keys}
            modes = [(0,0),(1,0),(0,1),(1,1)]
            for m in modes:
                interps = self._make_interps(*m)
                for i,key in enumerate(keys):
                    func_dict[key].append(interps[i])
            self._func_dict = func_dict
                
        elif mode == 'taper':
            modes = [(1,0),(1,1)]
            for i,m in enumerate(modes):
                interps = self._make_interps(*m)
                for j,key in enumerate(keys):
                    self._func_dict[key][2*i+1] = interps[j]

        elif mode == 'accelerating':
            modes = [(0,1),(1,1)]
            for i,m in enumerate(modes):
                interps = self._make_interps(*m)
                for j,key in enumerate(keys):
                    self._func_dict[key][i+2] = interps[j]
        
    def _pick_function(self,functype,taper,accelerating):
        return self._func_dict[functype][int(taper+2*accelerating)]

    def _update_functions(self):
        self._pdf = self._pick_function('pdf',self.taper,self.accelerating)
        self._cdf = self._pick_function('cdf',self.taper,self.accelerating)
        self._ppf = self._pick_function('ppf',self.taper,self.accelerating)

    def _tf(self,mf,taper):
        factor = (self.n + 1) / self.n if taper else 1
        tf1 = factor / (1 - self.j) / self.scale_value
        return tf1 * mf**(1 - self.jf)

    def _average_time(self,taper,accelerating):
        if accelerating:
            def accel_weight(mf,taper=False):
                return 1e6 * self.tau * (1 - np.exp(-self._tf(mf,taper=taper) / self.tau / 1e6))
            ret = self.imf.weight_average(accel_weight,taper)
        else:
            ret = self.imf.weight_average(self._tf,taper)
        return ret

    def pdf(self,x):
        return self._pdf(x,extrapolate=False)

    def cdf(self,x):
        return self._cdf(x,extrapolate=False)

    def ppf(self,x):
        return self._ppf(x,extrapolate=False)

    def rvs(self,N):
        samp = np.random.uniform(self.cdf(self.m1),self.cdf(self.m2),size=N)
        return self.ppf(samp)
    
    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self,x):
        self._taper = x
        self._update_functions()

    @property
    def accelerating(self):
        return self._accelerating

    @accelerating.setter
    def accelerating(self,x):
        self._accelerating = x
        self._update_functions()

        
class PMF_2C(PMF):
    """
    Two-component Protostellar Mass Function.

    Creates a distribution for the Protostellar Mass Function (PMF) 
    corresponding to a supplied IMF and two-component accretion history
    using the formalism of McKee & Offner (2010).

    Arguments:
    ---------- 
    """
    def __init__(self,imf,m1,m2,
                 j,jf,
                 R_mdot,m_is,
                 n,tau):
        self.imf = imf
        self.m1 = m1
        self.m2 = m2
        self.j = j
        self.jf = jf
        self.R_mdot = R_mdot
        self.m_is = m_is
        self.n = n
        self.tau = tau

        self._points = np.geomspace(min(self.m1,1e-3),self.m2,200)
        self._func_dict = None
        self._calculate('all')
        self._taper = False
        self._accelerating = False
        self._update_functions()

    def _make_interps(self,taper,accelerating):
        def pmf(mass,taper,accelerating):
            avg_time = self._average_time(taper,accelerating)

            def m_dot(mf,mass_):
                return self.m_is * np.sqrt(1 + self.R_mdot**2 *
                                           (mass_ / mf)**(2 * self.j) *
                                           mf**(2 * self.jf))

            def integrand(mf,mass_):
                def base_tm(mf,mass_):
                    return mass_ / self.m_is * hyp2f1(0.5,0.5/self.j,1+0.5/self.j,
                                                      -(self.R_mdot*(mass_/mf)**self.j*mf**self.jf)**2)
                
                if taper:
                    tf = self._tf(mf,taper)
                    def root_t(t,mf,mass_):
                        term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                        term2 = base_tm(mf,mass_)
                        prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                        prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                        return term1 - term2, prime_term1 - prime_term2

                    def taper_factor(mf,mass_):
                        sol = root_scalar(root_t,args=(mf,mass_),x0=0,fprime=True)
                        return 1 - (sol.root / tf)**self.n

                    t_factor = taper_factor(mf,mass_)
                    if accelerating:
                        tm = (1 - t_factor)**(1 / self.n) * tf
                else:
                    t_factor = 1
                    if accelerating:
                        tm = base_tm(mf,mass_)
                a_factor = np.exp(-tm / self.tau / 1e6) if accelerating else 1

                return self.imf(mf) * mass_ / m_dot(mf,mass_) / t_factor * a_factor

            def integral(lolim,mass_,**kwargs):
                return scipy.integrate.quad(integrand,lolim,self.m2,args=(mass_),**kwargs)[0]

            ret = np.vectorize(integral)(np.where(self.m1 < mass, mass, self.m1),mass)
            return np.where(ret / avg_time > 0, ret / avg_time, 0) #ensure the PMF is always >= 0

        base = pmf(self._points,taper,accelerating)
        pdf = base / self._points
        cdf = cumulative_trapezoid(pdf,self._points,initial=0)
        cdf = np.concatenate((cdf,[max(cdf)]))
        cdf_points = np.concatenate(([min(self._points)],(self._points[1:]+self._points[:-1])/2,[self.m2]))
        zero_arg = np.argmin(np.diff(cdf))
        return (PchipInterpolator(self._points,pdf),
                PchipInterpolator(cdf_points,cdf),
                PchipInterpolator(cdf[:zero_arg+1],cdf_points[:zero_arg+1]))

    def _calculate(self,mode):
        not_ok = (self.j is None) | (self.jf is None) | (self.R_mdot is None)
        if not_ok:
            raise ValueError('Cannot calculate a PMF without a history or all of (j, jf, R_mdot)')
        else:
            pass

        keys = ['pdf','cdf','ppf']
        if mode == 'all':
            func_dict = {key: [] for key in keys}
            modes = [(0,0),(1,0),(0,1),(1,1)]
            for m in modes:
                interps = self._make_interps(*m)
                for i,key in enumerate(keys):
                    func_dict[key].append(interps[i])
            self._func_dict = func_dict

        elif mode == 'taper':
            modes = [(1,0),(1,1)]
            for i,m in enumerate(modes):
                interps = self._make_interps(*m)
                for j,key in enumerate(keys):
                    self._func_dict[key][2*i+1] = interps[j]

        elif mode == 'accelerating':
            modes = [(0,1),(1,1)]
            for i,m in enumerate(modes):
                interps = self._make_interps(*m)
                for j,key in enumerate(keys):
                    self._func_dict[key][i+2] = interps[j]

    def _tf(self,mf,taper):
        factor = (self.n + 1) / self.n if taper else 1
        body = mf / self.m_is * hyp2f1(0.5,0.5/self.j,
                                       1+0.5/self.j,
                                       -(self.R_mdot * mf**self.jf)**2)
        return factor * body


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
