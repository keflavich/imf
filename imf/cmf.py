from __future__ import print_function
import numpy as np
from astropy import units as u
from astropy import constants
import scipy.stats
from scipy.optimize import root_scalar

from . import imf
from .distributions import Distribution
from .imf import MassFunction

class PN_CMF:
    """
    Padoan & Nordlund IMF - from http://adsabs.harvard.edu/abs/2011ApJ...741L..22P

    Parameters
    ----------

    tnow : float
        Time at which to evaluate the mass function in units of the
        crossing time


    Does not match their figures yet!
    """
    def __init__(self,mmin,mmax,
                 T0=10*u.K, L0=10*u.pc,
                 v0=4.9*u.km/u.s, rho0=2e-21*u.g/u.cm**3,
                 alpham1=1.35, eff=0.26, beta=0.4,
                 T_mean=7*u.K, mean_mol_wt=2.33):

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

    def __call__(self,
                 tnow=1, #tnow is in # of crossing times
                 visible_only=True,
                 cores='prestellar',
                 return_masses=False):
        
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

class HC(MassFunction):
    
    def init(self, sizescale=1*u.pc, n17=3.8, alpha_ct=0.75, mean_mol_wt=2.33,
             V0=0.8*u.km/u.s, meandens=5000*u.cm**-3, temperature=10*u.K,
             eta=0.45, b_forcing=0.4, Mach=6,
             eos='isothermal', include_B=False):
        """ Equation 21 of Hennebelle & Chabrier 2013
        
        Parameters
        ----------
        mass : np.array
            Masses at which to evaluate the PDF
        sizescale : pc equivalent
            The size of the clump (I think - extremely difficult to find this)
        n17 : float
            The "n" value in Equation 17, quoted to be 3.8 shortly afterward in the text
        alpha_ct : float
            "a dimensionless coefficient of the order of a few"
            I derived 0.75 from equation 9
        V0 : float
            u0 * 0.8 km/s according to the bottom of page 6, under eqn 36,
            which references eqn 16
        eta : None or float
            derived from Equation 17, but can be specified directly
        b_forcing : float
            The Forcing Parameter `b` from equation 4
        Mach : float
            Mach number
        """

        eos_types = ['isothermal','polytropic','barotropic']
        if eos not in eos_types:
            raise ValueError(f'EOS must be one of the following: {eos_types}')
        
        rho_bar = meandens * mean_mol_wt * constants.m_p

        c_s = ((constants.k_B * temperature /
                (mean_mol_wt*constants.m_p))**0.5).to(u.km/u.s)

        sigma = (np.log(1+b_forcing**2 * Mach**2))**0.5

        if eta is None:
            # eqn 17
            eta = (n17 - 3.)/2.

        alpha_g = 3/5. # for a uniform density fluctuation
        # eqn 9
        phit = 2 * alpha_ct * (24 / np.pi**2 / alpha_g)

        # dimensionless geometrical factor of the order of unity
        # For a sphere:
        aJ = np.pi**2.5 / 6
        # a geometrical factor, typically of the order of 4pi/3
        Cm = 4 * np.pi / 3

        # Jeans mass (eqn 13)
        MJ0 = (aJ / Cm * c_s**3 * constants.G**-1.5 * rho_bar**-0.5).to(u.M_sun)
    
        # Jeans length (eqn 14)
        lambdaJ0 = ((np.pi**1.5 / Cm)**(1./3) * c_s * (constants.G*rho_bar)**-0.5).to(u.pc)

        # relative Mach number (eqn 20)
        Mstar = (3**-0.5 * V0 / c_s * (lambdaJ0 / u.pc)**eta).to(u.dimensionless_unscaled)
    
        # Eqn 7 of Paper I
        # delta = np.log(rho/rho_bar
        # R = (mass/rho_bar)**(1/3.) * np.exp(-delta/3.) / lambdaJ0

        def R_side(R,M):
            return R * (1 + Mstar**2 * R**(2 * eta)) - M

        def get_root(M):
            return root_scalar(R_side,x0=1,args=(M)).root

        #Rtwiddle = (sizescale / lambdaJ0).to(u.dimensionless_unscaled)
        Rtwiddle = np.vectorize(get_root)(mass.value)
        
        Mtwiddle = (mass / MJ0).to(u.dimensionless_unscaled)

        #sigma *= 1 - (Rtwiddle * lambdaJ0 / sizescale)**(n17 - 3)

        # after eqn 21
        N0 = rho_bar / MJ0
        # PROBLEM: N0 is defined to be rho_bar / MJ0, but that is a dimensional
        # quantity with units cm^-3. This is a contradiction that means some
        # definition here is wrong.

        # eqn 21
        N = (2./phit * N0 * Rtwiddle**-6 *
             (1 + (1 - eta) * Mstar**2 * Rtwiddle**(2*eta)) /
             (1+(2*eta+1)*Mstar**2*Rtwiddle**(2*eta)) *
             (Mtwiddle / Rtwiddle**3)**(-1 - np.log(Mtwiddle/Rtwiddle**3) / 2 / sigma**2) *
             np.exp(-sigma**2/8.) / (2 * np.pi)**0.5 / sigma)

        return N

class HC(Distribution):

    def __init__(self):
        pass

    def calculate(self,M):
        #-(t0/tR) x rhobar/Mj/M x dR/dM x (ddelta/dR - abs(dsigma/dR/sigma*(delta+sigma^2/2))) x 1/sqrt(2pisigma^2) x exp(-delta^2/2sigma^2 + delta / 2 - sigma^2/8)
        #functions of EOS: M(R), dM/dR, Mj/Lj/Mstar (via sound speed)
        #functions of time dependence: tR

        #this block encompasses the changes based on EOS and sources of support
        #EOS sets thermal Cs for Mj/Lj/Mstar
        cs_mod = 1 if eos == 'isothermal' else gamma1
        Cs = ((constants.k_B * temperature * cs_mod /
               (mean_mol_wt*constants.m_p))**0.5).to(u.km/u.s)

        #EOS/support define M(R) and dM/dR
        mag_coef = 1 if include_B else 0
        Va_sq = B0**2 / 24 * np.pi * meandens / Cs**2
        rhobar = meandens * mean_mol_wt * constants.m_p

        #define M(R) and D (the thermal and magnetic terms of M)
        if eos == 'barotropic':
            Kcrit = ((rho_crit / meandens).decompose())**(gamma1-gamma2)
            def R_M(R_,M_):
                term1 = ((M_ / R_**3)**((gamma1-1)*m) + Kcrit**m * (M_ / R_**3)**((gamma2-1)*m))**(1/m)
                return term1 + Mstar**2 * R_**(2*eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*gammab-1)

            def D_funcs(rho_):
                D = ((rho_)**((gamma1-1)*m) + Kcrit**m * (rho_)**((gamma2-1)*m))**(1/m) + mag_coef * Va_sq * rho_**(2*gammab-1)
                dD = (((rho_)**((gamma1-1)*m) + Kcrit**m * (rho_)**((gamma2-1)*m))**(1/m-1) *
                      ((gamma1 - 1) * rho_**((gamma1-1)*m-1) + Kcrit**m * (gamma2 - 1) * rho_**((gamma2-1)*m-1)) +
                      mag_coef * (2 * gammab - 1) * Va_sq * rho_**(2*gammab-2)
                return D, dD 
            
        else:
            def R_M(R_,M_):
                return R_ * ((M_ / R_**3)**(cs_mod-1) + Mstar**2 * R_**(2 * eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*gammab-1)) - M_

            def D_funcs(rho_):
                D = rho_**(cs_mod-1) + mag_coef * Va_sq * rho_**(2*gammab-1)
                dD = (cs_mod-1) * rho_**(cs_mod-2) + mag_coef * Va_sq * (2 * gammab - 1) * rho_**(2*gammab-2)
                return D, dD

        #equations 33/34 (specifying dM/dR) of HC13 hold for all relevant definitions of D
        def dM_dR(M,R):
            rho = M / R**3
            D, dD = D_funcs(rho)
            B = D - 3 * rho * dD * (2 * eta + 1) * Mstar * R**(2*eta)
            C = 1 - R**-2 * dD
            return B / C

        def get_root(M):
	    return root_scalar(R_M,x0=1,args=(M)).root
        
        aj = np.pi**(5/2) / 6
        cm = 4 * np.pi / 3
        Mj = (aj / cm * Cs**3 / np.sqrt(constants.G**3 * rhobar)).to(u.M_sun)
        Lj = ((np.pi**(3/2) / cm)**(1/3) * Cs / np.sqrt(constants.G * rhobar)).to(u.pc)
        Mstar = V0 / Cs * (Lj / 1*u.pc)**eta / np.sqrt(3)

        Mt = M / Mj.value
        Rt = np.vectorize(get_root)(Mt)
        delta = np.log(Mt / Rt**3)

        #calculate variance and correction term (second term in HC13 equation 2)
        sig_0 = np.log(1 + b_forcing**2 * Mach**2)
        sig_sq = sig_0 * (1 - (Rt * Lj / sizescale)**(2*eta))
        dsigma_dR = -(2 * np.sqrt(sig_sq))**-1 * (sig_0 - sig_sq) / Rt
        corr = dsigma_dR / np.sqrt(sig_sq) * (delta + sig_sq / 2)

        N = (rhobar / Mj / np.sqrt(2 * np.pi * sig_sq) / dM_dR(Mt,Rt) *
             ((3 / Rt - dM_dR(Mt,Rt) / Mt) + corr) *
             np.exp(-delta**2 / 2 / sig_sq + delta / 2 - sig_sq / 8))

        if time_dep:
            alpha_ct = 3.7 # coefficient for crossing time, chosen to make phi ~ 3
            alpha_g = 0.6 # coefficient for self-gravitating gas in uniform density fluctuations
            phi_t = 2 * alpha_ct * np.sqrt(24 / np.pi**2 / alpha_g)

            N *= np.sqrt(np.exp(delta)) / phi_t
        
        return N * (M > self.m1) * (M < self.m2)

    def pdf(self,x):
        return self.calculate(x)

    def cdf(self,x):
        return quad(self.calculate,0,x)[0] #placeholder, needs to be vectorized still

    def ppf(self,x): #is there a way to do this without interpolation?
        return 0

    def rvs(self,N):
        samp = np.random.uniform(self.cdf(self.m1),self.cdf(self.m2),size=N)
        return self.ppf(samp)

