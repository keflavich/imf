import numpy as np
from astropy import units as u
from scipy.optimize import root_scalar

from imf.imf import get_massfunc, Schechter

expectedmass_cache = {}

def sample_mass_function(mtotal,
                         massfunc='kroupa',
                         tolerance=0.0,
                         sampling='random',
                         stop_criterion='nearest',
                         mmin=None,
                         mmax=None,
                         verbose=False,
                         silent=False,
                         **kwargs):
    """
    Sample masses from a mass function.

    Parameters
    ----------
    mtotal : float
        The target cluster mass, in solar masses.
    massfunc : str or MassFunction
        A mass function to use. Can be an existing ``MassFunction`` instance
        or ``'salpeter'``, ``'kroupa'``, or ``'chabrier'`` for default
        common forms (default = ``'kroupa'``)
    tolerance : float
        How close the sum of random samples must be to the requested 
        cluster mass; sampling stops once total sampled mass + ``tolerance``
        > ``mcluster``. It can be zero, but this does not guarantee that
        the final cluster mass will be exactly ``mcluster`` (default = 0)
    sampling: str
        Which sampling method to use. Can be ``'random'`` or ``'optimal'``
        (default = ``'random'``)
    stop_criterion : str
        The criterion to stop random sampling when the total cluster mass 
        is reached. Can be ``'nearest'``, ``'before'``, ``'after'``, or
        ``'sorted'``. Does not factor into optimal sampling (default = 
        ``'nearest'``)

    Other Parameters
    ----------------
    mmin: float
        If the provided mass function has no defined minimum, use this
        (default = ``None``)
    mmax: float
        If the provided mass function has no defined maximum, use this
        (default = ``None``)
    verbose: bool
        Whether to provide running commentary on the sampling (default = 
        ``False``)
    silent: bool
        Whether to suppress the final sampled cluster mass (default = 
        ``False``)
    """
    # use most common mass to guess needed number of samples
    # nsamp = mcluster / mostcommonmass[get_massfunc_name(massfunc)]
    # masses = inverse_imf(np.random.random(int(nsamp)), massfunc=massfunc, **kwargs)

    # mtot = masses.sum()
    # if verbose:
    #    print(("%i samples yielded a cluster mass of %g (%g requested)" %
    #          (nsamp, mtot, mcluster)))

    # catch wrong keywords early
    ok_samplings = ['random', 'optimal']
    ok_criteria = ['nearest', 'before', 'after', 'sorted']
    if not sampling in ok_samplings:
        raise ValueError("Sampling should be either 'random' or 'optimal' (see documentation)")
    if (sampling == 'random') and not stop_criterion in ok_criteria:
        raise ValueError("Stop criterion for random sampling should be 'nearest', 'before', 'after', or 'sorted' (see documentation)")

    if sampling == 'optimal':
        mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)
        masses, msamp = _opt_sample(mtotal, mfc, tolerance=tolerance)
        if verbose:
            print(f'Sampled {len(masses)} new masses.')
        if not silent:
            print(f'Total mass is {np.round(msamp, 3)} (limit was {int(mtotal)})')

    else:
        mtotal = u.Quantity(mtotal, u.M_sun).value

        mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)

        if (massfunc, mfc.mmin, mfc.mmax) in expectedmass_cache:
            expected_mass = expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)]
            assert expected_mass > 0
        else:
            expected_mass = mfc.m_integrate(mfc.mmin, mfc.mmax)[0]
            assert expected_mass > 0
            expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)] = expected_mass

        if verbose:
            print("Expected mass is {0:0.3f}".format(expected_mass))

        msamp = 0
        masses = []

        while msamp < mtotal + tolerance:
            # at least 1 sample, but potentially many more
            nsamp = int(np.ceil((mtotal + tolerance - msamp) / expected_mass))
            assert nsamp > 0
            newmasses = mfc.distr.rvs(nsamp)
            masses = np.concatenate([masses, newmasses])
            msamp = masses.sum()
            if verbose:
                print("Sampled %i new masses.  Total is now %g" %
                      (int(nsamp), msamp))

            if msamp >= mtotal + tolerance:  # don't force exact equality; that would yield infinite loop
                mcum = masses.cumsum()
                if stop_criterion == 'sorted':
                    masses = np.sort(masses)
                    if np.abs(masses[:-1].sum() - mtotal) < np.abs(masses.sum() -
                                                                     mtotal):
                        # if the most massive star makes the cluster a worse fit, reject it
                        # (this follows Krumholz+ 2015 appendix A1)
                        last_ind = len(masses) - 1
                    else:
                        last_ind = len(masses)
                else:
                    if stop_criterion == 'nearest':
                        # find the closest one, and use +1 to include it
                        last_ind = np.argmin(np.abs(mcum - mtotal)) + 1
                    elif stop_criterion == 'before':
                        last_ind = np.argmax(mcum > mtotal)
                    elif stop_criterion == 'after':
                        last_ind = np.argmax(mcum > mtotal) + 1
                masses = masses[:last_ind]
                msamp = masses.sum()
                if verbose:
                    print(
                        "Selected the first %i out of %i masses to get %g total" %
                        (last_ind, len(mcum), msamp))
                # force the break, because some stopping criteria can push msamp < mtotal
                break

        if not silent:
            print("Total mass is %g (limit was %g)" % (msamp, mtotal))

    return masses

def sample_number(N,
                  massfunc='kroupa',
                  sampling='random',
                  tolerance=0.0,
                  mmin=None,
                  mmax=None,
                  silent=False,
                  **kwargs):
    """
    sample N masses from a function
    """
    assert N > 0

    # catch wrong keywords early
    ok_samplings = ['random', 'optimal']
    if not sampling in ok_samplings:
        raise ValueError("Sampling should be either 'random' or 'optimal' (see documentation)")

    mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)

    if sampling == 'optimal':
        if (massfunc, mfc.mmin, mfc.mmax) in expectedmass_cache:
            expected_mass = expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)]
            assert expected_mass > 0
        else:
            expected_mass = mfc.m_integrate(mfc.mmin, mfc.mmax)[0]
            assert expected_mass > 0
            expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)] = expected_mass

        mtot = N * expected_mass
        print(f'expected mass = {expected_mass}, mtot = {mtot}')
        masses, _ = _opt_sample(mtot, mfc, tolerance=tolerance)
        if not silent:
            print(f'Sampled {len(masses)} masses.')
            print(f'Total mass is {np.round(np.sum(masses), 3)}.')
            
    else:
        masses = mfc.distr.rvs(N)
        if not silent:
            print(f'Total mass is {np.round(np.sum(masses), 3)}.')
        
    return masses


### wrappers for specific use cases ###

def multiplicity(m_syst,
                 masses=[0],
                 probs=[0]):
    """
    create multiple systems based on provided
    multiplicity fractions
    (default: Offner+ 2023)
    """
    return 0

def member_masses(m_syst,
                  n_members):
    """
    calculate masses for members of a multiple system
    (default: uniform mass ratio distribution)
    """
    return 0

def multiple_props():
    #figure out multiplicity and member masses then wrap/combine them here
    return 0

def syst_to_stellar(syst_masses,
                    mult_props):
    #convert a cluster of systems to a cluster of stars
    return 0

def make_star_cluster(mtotal=None,
                      nstars=None,
                      return_stellar=False,
                      return_conversion=False,
                      **kwargs):
    """
    make a cluster with either some total mass
    or number of stars
    -sample
    -assign multiples
    -find mass ratios within multiples

    "return stellar" also returns the stellar masses
    "return conversion" also returns the information used to 
    obtain the stellar masses
    """
    return 0

def make_igimf(N_clusters,
               mclust_min=None,
               mclust_max=None,
               mtaper=None,
               imf='kroupa',
               sampling='random',
               stop_criterion='nearest',
               mstar_min=None,
               mstar_max=None,
               **kwargs):
    """
    -sample some number of clusters from a Schechter function
    -run star sampler on mass of each cluster (optionally; allow stellar IMFs)
    """
    return 0

### functions facilitating optimal sampling ###

def _prefactor(max_star, massfunc):
    """
    Returns the multiplier required for an IMF to have at most one star above m_max.
    """
    return 1 / massfunc.integrate(max_star, massfunc.mmax)[0]


def _M_cluster(m, massfunc):
    """
    Returns the mass of a cluster distributed according to some IMF where the 
    largest star has mass m.
    """
    k = _prefactor(m, massfunc)
    return k * massfunc.m_integrate(massfunc.mmin, m)[0] + m


def _max_star(m, M_res, massfunc):
    """
    Returns the most massive star capable of forming in a cluster of mass M_res
    according to the m_max/M_cluster relation. Formatted for use with root finding.
    """
    return M_res - _M_cluster(m, massfunc)


def _max_star_prime(m, M_res, massfunc):
    """
    Returns the derivative of _max_star at mass m. Used for Newton's method in
    the case of an infinite upper bound on the provided mass function.
    """
    term1 = _prefactor(m, massfunc)**2 * massfunc(m) * massfunc.m_integrate(massfunc.mmin, m)[0]
    term2 = m * massfunc(m) * _prefactor(m, massfunc)
    return -term1 - term2 - 1


def _opt_sample(M_res, massfunc, tolerance=0):
    """
    Returns a numpy array containing stellar masses that optimally sample 
    from a provided MassFunction to make a cluster with mass M_res.
    """
    # retrieve mass bounds from provided massfunc
    mmin = massfunc.mmin
    mmax = massfunc.mmax
    finMax = np.isfinite(mmax)

    # finding all the component stars requires a cutoff--ensure there is one
    if not np.logical_or(np.isfinite(np.log(mmin)), np.isfinite(np.log(tolerance))):
        raise ValueError('Optimal sampling requires either mmin or tolerance to be finite and greater than zero.')

    if finMax:
        # bracket from min to ALMOST max (max gives an undefined prefactor)
        sol = root_scalar(_max_star, args=(M_res, massfunc), bracket=[mmin, 0.9999*mmax])
    else:
        # use Newton's method
        sol = root_scalar(_max_star, args=(M_res, massfunc), x0=10*mmin, fprime=_max_star_prime)
    k = _prefactor(sol.root, massfunc)
    M_tot = sol.root
    star_masses = [sol.root]
    m_i = sol.root

    while np.abs(M_res-M_tot) > np.maximum(mmin, tolerance):
        try:
            m_i_plus = root_scalar(lambda x: k * massfunc.integrate(x, m_i)[0]-1,
                                   bracket=[mmin, m_i]).root
        except(ValueError):
            print(f'Reached provided lower mass bound; stopping')
            break
        m = k * massfunc.m_integrate(m_i_plus, m_i)[0]
        star_masses.append(m)
        M_tot += m
        m_i = m_i_plus

    return np.array(star_masses), M_tot
