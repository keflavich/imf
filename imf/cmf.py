from __future__ import print_function
import numpy as np
from astropy import units as u
from astropy import constants
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from . import imf
from .distributions import Distribution
from .imf import MassFunction

class PN_CMF:
    """
    Core mass function derived from a population generated according to
    Padoan/Nordlund 2011 (http://adsabs.harvard.edu/abs/2011ApJ...741L..22P)

    Does not match their figures yet!
    """
    def __init__(self,mmin=None,mmax=None,
                 T0=10*u.K, L0=10*u.pc,
                 v0=4.9*u.km/u.s, rho0=2e-21*u.g/u.cm**3,
                 alpham1=1.35, eff=0.26, beta=0.4,
                 T_mean=7*u.K, mu=2.33):
        """
        Parameters
        ----------
        mmin : float
            Minimum permissible core mass
        mmax : float
            Maximum permissible core mass
        T0 : K (or equivalent)
            Mean temperature of the parent cloud (default = 10 K)
        L0 : pc (or equivalent)
            Cloud size (default = 10 pc)
        v0 : km s^-1 (or equivalent)
            RMS velocity of the parent cloud at R = 1 pc
            (default = 0.8 km s^-1)
        rho0 : g cm^-3 (or equivalent)
            Mean mass density of gas in the parent cloud
            (default = 2e-21 g cm^-3)
        alpham1 : float
            Determines the mass function (replace with mass function?)
        eff : float
            Efficiency of core formation. The core mass budget is
            eff * cloud mass (default = 0.26)
        beta : float
            Ratio of gas to magnetic pressure in postshock gas (default = 0.4)
        T_mean : K (or equivalent)
            Mean core temperature (default = 7 K)
        mu : float
            Mean molecular weight of gas (default = 2.33)
        """

        self._tcross = (L0 / v0).to(u.Myr)
        m0 = (4 / 3 * np.pi * L0**3 * rho0).to(u.M_sun) #total cloud mass

        # Mach number defined on largest scale (assumed)
        MS0 = (v0 / ((constants.k_B * T0 / (mean_mol_wt * constants.m_p))**0.5).to(u.km/u.s)).value

        # implementing eqn 1
        sigma_rho = MS0 / np.sqrt((1 + beta**-1)) / 2. #stdev of density [eqn 3]
        s = np.sqrt(np.log(1 + sigma_rho**2)) #lognormal shape

        n0 = (rho0 / constants.m_p / mean_mol_wt).to(u.cm**-3).value
        self._massfunc = imf.PadoanTF(mmin=mmin,mmax=mmax,
                                      T0=T0.value,n0=n0,
                                      sigma=s)
        self._maccr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                                       massfunc=self.massfunc,
                                       silent=True) * u.M_sun
        
        # loc is s**2/2 instead of -s**2/2 because this is the _mass-weighted_ log-normal density PDF
        # (this is only weakly hinted at in PN11 by the words "converted to mass fraction" just before eqn 1)
        # see e.g. Hopkins 2013 eqn 4 for the mass-weighted definitions
        ln_rho = scipy.stats.norm.rvs(loc=s**2/2,
                                      scale=s,
                                      size=len(self.maccr))
        x = np.exp(ln_rho)
        self._rho = x * rho0 #densities around each core
        
        self._birthdays = np.random.random(len(self.maccr)) * self.tcross #core birthdays (assuming flat formation over crossing time)

        a = (3 - (3 / alpham1)) / 2
        self._taccr = (self.tcross * sigma_rho**((4 - 4 * a) / (3 - 2 * a)) *
                       (self.maccr / m0)**((1 - a) / (3 - 2 * a))).to(u.Myr)

        c_s = ((constants.k_B * T_mean / (mean_mol_wt * constants.m_p))**0.5).to(u.km/u.s)
        self._mbe = (1.182 * c_s**3 / (constants.G**1.5 * self.rho**0.5)).to(u.M_sun)

        self.distr = PN(self.taccr,self.maccr,self.mbe,
                        self.rho,self.tcross,self.birthdays)

    def __call__(self,m,
                 tnow=1, #tnow is in # of crossing times
                 cores='prestellar',
                 integral_form=False,
                 visible_only=True,
                 return_masses=False):
        """
        Parameters
        ----------
        tnow : float
            Time at which to evaluate the CMF (in units of crossing times)
        visible_only : Bool
        
        """
        
        #core_types = ['prestellar','stellar','all']

        tbe = (self.taccr * (self.maccr / self.mbe)**(-1/3.)).to(u.s)
        tff = ((3 * np.pi / (32 * constants.G * self.rho))**0.5).to(u.s)
        mmax = (self.maccr * ((tbe + tff) / self.taccr)**3).to(u.M_sun)

        age = tnow * self.tcross - self.birthdays
        isBorn = age > 0
        isPrestellar = age < tbe + tff
        belowBE = self.maccr < self.mbe
        isStellar = np.logical_and(~isPrestellar,~belowBE)
        isForming = age < self.taccr

        m_f = np.vstack([mmax.value, self.maccr.value]).min(axis=0)*u.M_sun
        mnow = ((age / self.taccr)**3 * self.maccr).to(u.M_sun)
        mnow[mnow > self.maccr] = self.maccr[mnow > self.maccr]
        #mnow[mnow > m_f] = m_f[mnow > m_f]

	#We assume that cores that do not reach their BE mass are seen only during
        #their formation time, taccr

        if visible_only:
            cut = np.logical_and(isBorn,isForming)
            if cores == 'prestellar':
                cut = np.logical_and(cut,isPrestellar)
            elif cores == 'stellar':
                cut = np.logical_and(cut,isStellar)
        else:
            if cores == 'prestellar':
                cut = np.logical_and(isBorn,~isStellar)
            elif cores == 'stellar':
                cut = np.logical_and(isBorn,isStellar)
            else:
                cut = np.ones(len(mnow)).astype(int)

        core_masses = mnow[cut]

        edges = 10**np.histogram_bin_edges(np.log10(core_masses.value)) * u.M_sun
        hist,edges = np.histogram(core_masses,bins=edges)
        if return_masses:
            return hist,edges,core_masses
        else:
            return hist,edges

        #########
        self.distr.time = tnow
        self.distr.cores = cores

        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def will_collapse(self):
        return self.maccr > self.mbe / 2

    @property
    def tcross(self):
        return self._tcross
    
    @property
    def maccr(self):
        return self._maccr

    @property
    def rho(self):
        return self._rho

    @property
    def birthdays(self):
        return self._birthdays

    @property
    def taccr(self):
        return self._taccr

    @property
    def mbe(self):
        return self._mbe

    @property
    def massfunc(self):
        return self._massfunc
    
class PN(Distribution):

    def __init__(self,taccr,maccr,mbe,rho,
                 tcross,birthdays):
        
        self.tbe = (taccr * (maccr / mbe)**(-1/3.)).to(u.s)
        self.tff = ((3 * np.pi / (32 * constants.G * rho))**0.5).to(u.s)
        self.mmax = (maccr * ((tbe + tff) / taccr)**3).to(u.M_sun)
        self.belowBE = maccr < mbe
        self.m_f = np.vstack([mmax.value, maccr.value]).min(axis=0)*u.M_sun

        self.tcross = tcross
        self.birthdays = birthdays
        self.tnow = None

        keys = ['prestellar','stellar','all']
        self._func_dict = {key: None for key in keys}
        
    def _set_time(self,tnow):
        keys = ['prestellar','stellar','all']
        
        if tnow != self.tnow:
            age = tnow * self.tcross - self.birthdays
            isBorn = age > 0
            isPrestellar = age < tbe + tff
            isStellar = np.logical_and(~isPrestellar,~belowBE)
            isForming = age < self.taccr

            mnow = ((age / self.taccr)**3 * self.maccr).to(u.M_sun)
            mnow[mnow > self.maccr] = self.maccr[mnow > self.maccr]
            #mnow[mnow > m_f] = m_f[mnow > m_f]

            for key in keys:
                if visible_only:
                    cut = np.logical_and(isBorn,isForming)
                    if cores == 'prestellar':
                        cut = np.logical_and(cut,isPrestellar)
                    elif cores == 'stellar':
                        cut = np.logical_and(cut,isStellar)
                else:
                    if cores == 'prestellar':
                        cut = np.logical_and(isBorn,~isStellar)
                    elif cores == 'stellar':
                        cut = np.logical_and(isBorn,isStellar)
                    else:
                        cut = np.ones(len(mnow)).astype(int)

                core_masses = mnow[cut]

                edges = 10**np.histogram_bin_edges(np.log10(core_masses.value)) * u.M_sun
                hist,edges = np.histogram(core_masses,bins=edges)
            
                #construct function dictionary
                norm = np.trapezoid(N,x=self._points)
                cdf = cumulative_trapezoid(N/norm,self._points,initial=0)
                cdf = np.concatenate((cdf,[max(cdf)]))
                cdf_points = np.concatenate(([min(self._points)],
	                                     (self._points[1:]+self._points[:-1])/2,
                                             [self.m2]))
                zero_arg = np.argmin(np.diff(cdf))

                functions = [PchipInterpolator(self._points,N/norm),
                             PchipInterpolator(cdf_points,cdf),
                             PchipInterpolator(cdf[:zero_arg+1],cdf_points[:zero_arg+1])]
                self._func_dict[key] = functions
            
        else:
            pass
        
    def pdf(self,x):
        return self._pdf(x,extrapolate=False)

    def cdf(self,x):
        return self._cdf(x,extrapolate=False)

    def ppf(self,x):
        return self._ppf(x,extrapolate=False)

    def rvs(self,N):
        samp = np.random.uniform(self.cdf(self.m1),self.cdf(self.m2),size=N)
        return self.ppf(samp)

    def _pick_functions(self,cores):
        functions = self._func_dict[cores]
        return functions[0],functions[1],functions[2]

    def _update_functions(self):
        self._pdf, self._cdf, self._ppf = self._pick_functions(self.cores)

    @property
    def cores(self):
        return self._cores

    @cores.setter
    def cores(self,x):
        self._cores = x
        self._update_functions()


class HC_CMF(MassFunction):
    
    #add a sound speed parameter that overrides calculation
    def __init__(self,mmin=None,mmax=None,
                 clump_size=1*u.pc, n_cl=5, mu=2.33,
                 Cs0=0.2*u.km/u.s, T0=10*u.K,
                 v0=0.8*u.km/u.s, eta=None,
                 n_pow=3.8, b_forcing=0.5,
                 eos='isothermal',gamma1=0.7,
                 gamma2=1.1,rho_crit=1e-18*u.g/u.cm**3,m=3,
                 include_B=False,B0=10*u.uG,gammab=0.1,
                 time_dep=True):
        """
        Generalized core mass function following the formalism of
        Hennebelle/Chabrier 2008/2009/2013.
        
        Parameters
        ----------
        mmin : float
            Minimum permissible core mass
        mmax : float
            Maximum permissible core mass
        clump_size : pc (or equivalent)
            Radius of the parent clump (default = 1 pc)
        n_cl : float
            Clump density normalization at 1 pc. Number density is
            n_cl * 1e3 (default = 5)
        mu : float
            Mean molecular weight of gas (default = 2.33)
        Cs0 : km s^-1 (or equivalent)
            Average isothermal sound speed for a cloud with 
            number density 10^4 cm^-3 (default = 0.2 km s^-1)
        T0 : K (or equivalent)
            Mean temperature of the parent clump. Used to calculate
            sound speed if none is provided (default = 10 K)
        v0 : km s^-1 (or equivalent)
            RMS velocity of the parent clump at R = 1 pc 
            (default = 0.8 km s^-1)
        eta : None or float
            Exponent governing the behavior of dispersion velocity with scale
        n_pow : float
            Index of 3D velocity power spectrum. Used to derive eta 
            if no eta is provided (default = 3.8)
        b_forcing : float
            Forcing parameter of turbulence (default = 0.4)
        eos : str
            String specifying which equation of state to use for gas 
            in the parent clump. Accepts 'isothermal', 'polytropic', 
            and 'barotropic'; see papers for implementation details
            (default = 'isothermal')
        gamma1 : float
            Exponent in a non-isothermal EOS. The sole exponent in a 
            polytropic case and the lower-density exponent in a
            barotropic case. Only used if eos != 'isothermal' (default = 0.7)
        gamma2 : float
            High-density exponent in a barotropic EOS. Only used if 
            eos == 'barotropic' (default = 1.1)
        rho_crit : g cm^-3 (or equivalent)
            Critical density in a barotropic EOS (i.e. where the piecewise
            halves meet). Only used if eos == 'barotropic' 
            (default = 1e-18 g cm^-3) 
        m : float
            Exponent governing the combination of the piecewise components
            of a barotropic EOS; higher = less blending (default = 3)
        include_B : bool
            Whether or not to include support from a magnetic field in
            CMF calculation. (default = False)
        B0 : gauss (or equivalent)
            Mean magnetic field strength (default = 10 microgauss)
        gammab : float
            Exponent governing the relationship between magnetic field
            strength and gas density (default = 0.1)
        """
        
        if eta is None:
            eta = (n_pow - 3) / 2

        eos_types = ['isothermal','polytropic','barotropic']
        if eos not in eos_types:
            raise ValueError(f'EOS must be one of the following: {eos_types}')
        cs_mod = gamma1 if eos != 'isothermal' else 1
        
        #scale density according to Larson laws
        n0 = (n_cl * 1e3 * u.cm**-3) / (clump_size.to(u.pc).value)**0.7
        rho0 = (n0 * mu * constants.m_p).to(u.g/u.cm**3)
        
        #scale sound speed according to EOS
        if Cs0 is None:
            Cs0 = np.sqrt(constants.k_B * T / mu / constants.m_p)
        Cs = Cs0.to(u.km/u.s) * np.sqrt((n0.value / 1e4)**(cs_mod-1))

        self.distr = HC(mmin,mmax,
                        clump_size,rho0,Cs,
                        v0,eta,b_forcing,
                        eos,gamma1,gamma2,rho_crit,m,
                        include_B,B0,gammab)

        self.normfactor = 1
        
    def __call__(self,m,
                 integral_form=False,
                 time_dep=True):
        """
        time_dep : bool
            If true, use the time-dependent CMF of HC13; otherwise,
            use the time-independent form of HC08/09 (default = True) 
        """

        self.distr.time_dep = time_dep
        
        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    @property
    def mmin(self):
        return self.distr.m1

    @property
    def mmax(self):
        return self.distr.m2

    @property
    def clump_size(self):
        return self.distr.clump_size

    @property
    def rho0(self):
        return self.distr.rho0

    @property
    def Cs(self):
        return self.distr.Cs

    @property
    def v0(self):
        return self.distr.v0

    @property
    def eta(self):
        return self.distr.eta

    @property
    def b_forcing(self):
        return self.distr.b_forcing

    @property
    def eos(self):
        return self.distr.eos

    @property
    def gamma1(self):
        try:
            return self.distr.gamma1
        except(AttributeError):
            raise AttributeError("This object has no gamma1")

    @property
    def gamma2(self):
        try:
            return self.distr.gamma2
        except(AttributeError):
            raise AttributeError("This object has no gamma2")

    @property
    def rho_crit(self):
        try:
            return self.distr.rho_crit
        except(AttributeError):
            raise AttributeError("This object has no rho_crit")

    @property
    def rho_crit(self):
        try:
            return self.distr.rho_crit
        except(AttributeError):
            raise AttributeError("This object has no m")

    @property
    def B0(self):
        if not self.distr.include_B:
            raise AttributeError("This object has no B0")
        else:
            return self.distr.B0
        
    @property
    def gammab(self):
        if not self.distr.include_B:
            raise AttributeError("This object has no gammab")
        else:
            return self.distr.gammab
        
class HC(Distribution):

    def __init__(self, m1, m2,
                 clump_size, rho0, Cs,
                 v0, eta, b_forcing,
                 eos, gamma1, gamma2, rho_crit, m,
                 include_B, B0, gammab):

        self.m1 = m1
        self.m2 = m2

        self.clump_size = clump_size
        self.rho0 = rho0
        self.Cs = Cs

        self.v0 = v0
        self.eta = eta
        self.b_forcing = b_forcing

        self.eos = eos
        if self.eos != 'isothermal':
            self.gamma1 = gamma1
        if self.eos == 'barotropic':
            self.gamma2 = gamma2
            self.rho_crit = rho_crit
            self.m = m

        self.include_B = include_B
        self.B0 = B0
        self.gammab = gammab

        self._points = np.geomspace(self.m1,self.m2)
        keys = ['pdf','cdf','ppf']
        self._func_dict = {key: [] for key in keys}

        self._calculate()
    
    def _calculate(self):
        #use EOS to set thermal Cs (for Mj/Lj/Mstar)        
        cs_mod = 1 if self.eos == 'isothermal' else self.gamma1
        Cs = self.Cs# * cs_mod #papers suggest gamma should go here, but not in the HC IDL code

        #add support from magnetic field
        mag_coef = 1 if self.include_B else 0
        gauss = u.g**0.5 / u.cm**0.5 / u.s # define a custom gauss unit to work in cgs
        B0 = self.B0.to(u.G).value * gauss # transform to custom gauss
        Va_sq = (B0**2 / (24 * np.pi * self.rho0) / Cs**2).to(u.dimensionless_unscaled)

        #use EOS/support to define M and dM/dR
        #including D, the thermal and magnetic terms of M
        if self.eos == 'barotropic':
            Kcrit = ((self.rho_crit / self.rho0).decompose())**(self.gamma1-self.gamma2)
            def R_M(R_,M_):
                A = (M_ / R_**3)**((self.gamma1-1)*self.m) + Kcrit**self.m * (M_ / R_**3)**((self.gamma2-1)*self.m)
                return R_ * (A**(1/self.m) + Mstar**2 * R_**(2*self.eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*self.gammab-1)) - M_

            def D_funcs(rho_):
                A = rho_**((self.gamma1-1)*self.m) + Kcrit**self.m * rho_**((self.gamma2-1)*self.m)
                D = A**(1/self.m) + mag_coef * Va_sq * rho_**(2*self.gammab-1)
                dD = (A**(1/self.m-1) *
                      ((self.gamma1 - 1) * rho_**((self.gamma1-1)*self.m-1) +
                       Kcrit**self.m * (self.gamma2 - 1) * rho_**((self.gamma2-1)*self.m-1)) +
                      mag_coef * (2 * self.gammab - 1) * Va_sq * rho_**(2*self.gammab-2))
                return D, dD 
            
        else:
            def R_M(R_,M_):
                return R_ * ((M_ / R_**3)**(cs_mod-1) + Mstar**2 * R_**(2 * self.eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*self.gammab-1)) - M_

            def D_funcs(rho_):
                D = rho_**(cs_mod-1) + mag_coef * Va_sq * rho_**(2*self.gammab-1)
                dD = (cs_mod-1) * rho_**(cs_mod-2) + mag_coef * Va_sq * (2 * self.gammab - 1) * rho_**(2*self.gammab-2)
                return D, dD

        #equations 33/34 (specifying dM/dR) of HC13 hold for all relevant definitions of D
        def dM_dR(M_,R_):
            rho = M_ / R_**3
            D, dD = D_funcs(rho)
            B = D - 3 * rho * dD + (2 * self.eta + 1) * Mstar**2 * R_**(2*self.eta)
            C = 1 - R_**-2 * dD
            return B / C

        #root find for R
        def get_root(M_):
            return root_scalar(R_M,x0=1,args=(M_)).root
        
        #Jeans mass/length; leading coefficients are as in the IDL code
        aj = np.pi**2.5 / 6
        Mj = (aj * Cs**3 / np.sqrt(constants.G**3 * self.rho0)).to(u.M_sun) * cs_mod**1.5 #in the IDL code, gamma is explicit here
        Lj = ((np.pi**0.5 / 2)**(1/3) * Cs / np.sqrt(constants.G * self.rho0)).to(u.pc) * cs_mod**0.5
        Li = self.clump_size.to(u.pc) / Lj
        
        #Mach number
        Mstar = self.v0 / Cs * (Lj.value)**self.eta / np.sqrt(3)
        Mach = np.sqrt(3) * Mstar * Li**self.eta
        
        Mt = self._points / Mj.value
        Rt = np.vectorize(get_root)(Mt)
        delta = np.log(Mt / Rt**3)

        #calculate variance and CMF dP/dR term
        sig_0 = np.log(1 + self.b_forcing**2 * Mach**2)
        sig_sq = sig_0 * (1 - (Rt / Li)**(2*self.eta))
        dsigma_dR = -self.eta / np.sqrt(sig_sq) * (sig_0 - sig_sq) / Rt
        corr = dsigma_dR / np.sqrt(sig_sq) * (delta + sig_sq / 2)

        #determine maximum possible "core" mass given provided sizescale
        mmax = root_scalar(lambda md, rd : R_M(rd,md),x0=1,args=(Li)).root * Mj

        #calculate unscaled, dimensionless CMF
        dM = dM_dR(Mt,Rt)
        N = (1 / Mt / dM * ((3 / Rt - dM / Mt) + corr) /
             np.sqrt(2 * np.pi * sig_sq) *
             np.exp(-(delta - sig_sq / 2)**2 / 2 / sig_sq))

        #N *= self.clump_size.to(u.pc)**3
        #N = N.to(u.dimensionless_unscaled).value

        #get rid of impossible entries
        N[~np.isfinite(N)] = 0
        N *= self._points <= min(self.m2,mmax.value)

        #store time-independent PDF
        norm = np.trapezoid(N,x=self._points)
        cdf = cumulative_trapezoid(N/norm,self._points,initial=0)
        cdf = np.concatenate((cdf,[max(cdf)]))
        cdf_points = np.concatenate(([min(self._points)],
                                     (self._points[1:]+self._points[:-1])/2,
                                     [self.m2]))
        zero_arg = np.argmin(np.diff(cdf))

        self._func_dict['pdf'].append(PchipInterpolator(self._points,N/norm))
        self._func_dict['cdf'].append(PchipInterpolator(cdf_points,cdf))
        self._func_dict['ppf'].append(PchipInterpolator(cdf[:zero_arg+1],cdf_points[:zero_arg+1]))

        #store time-dependent PDF       
        alpha_ct = 3.7 # coefficient for crossing time, chosen to make phi ~ 3
        alpha_g = 0.6 # coefficient for self-gravitating gas in uniform density fluctuations
        phi_t = 2 * alpha_ct * np.sqrt(24 / np.pi**2 / alpha_g)
        N *= np.sqrt(np.exp(delta)) / phi_t

        norm = np.trapezoid(N,x=self._points)
        cdf = cumulative_trapezoid(N/norm,self._points,initial=0)
        cdf = np.concatenate((cdf,[max(cdf)]))
        cdf_points = np.concatenate(([min(self._points)],
                                     (self._points[1:]+self._points[:-1])/2,
                                     [self.m2]))
        zero_arg = np.argmin(np.diff(cdf))

        self._func_dict['pdf'].append(PchipInterpolator(self._points,N/norm))
        self._func_dict['cdf'].append(PchipInterpolator(cdf_points,cdf))
        self._func_dict['ppf'].append(PchipInterpolator(cdf[:zero_arg+1],cdf_points[:zero_arg+1]))
        
    def _pick_function(self,functype,time_dep):
        return self._func_dict[functype][int(time_dep)]

    def _update_functions(self):
        self._pdf = self._pick_function('pdf',self.time_dep)
        self._cdf = self._pick_function('cdf',self.time_dep)
        self._ppf = self._pick_function('ppf',self.time_dep)
        
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
    def time_dep(self):
        return self._time_dep

    @time_dep.setter
    def time_dep(self,x):
        self._time_dep = x
        self._update_functions()
