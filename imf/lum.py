import numpy as np

mass_luminosity_interpolator_cache = {}


def mass_luminosity_interpolator(name):
    if name in mass_luminosity_interpolator_cache:
        return mass_luminosity_interpolator_cache[name]

    elif name == 'VGS':

        # publication values
        vgsMass = [
            51.3, 44.2, 41.0, 38.1, 35.5, 33.1, 30.8, 28.8, 26.9, 25.1, 23.6,
            22.1, 20.8, 19.5, 18.4
        ]
        vgslogL = [
            6.154, 6.046, 5.991, 5.934, 5.876, 5.817, 5.756, 5.695, 5.631,
            5.566, 5.499, 5.431, 5.360, 5.287, 5.211
        ]
        vgslogQ = [
            49.18, 48.99, 48.90, 48.81, 48.72, 48.61, 48.49, 48.34, 48.16,
            47.92, 47.63, 47.25, 46.77, 46.23, 45.69
        ]

        # mass (extrapolated)
        vgsMe = np.concatenate([
            np.linspace(0.03, 0.43, 100),
            np.linspace(0.43, 2, 100),
            np.linspace(2, 20, 100), vgsMass[::-1],
            np.linspace(50, 150, 100)
        ])

        # log luminosity (extrapolated)
        vgslogLe = np.concatenate([
            np.log10(0.23 * np.linspace(0.03, 0.43, 100)**2.3),
            np.log10(np.linspace(0.43, 2, 100)**4),
            np.log10(1.5 * np.linspace(2, 20, 100)**3.5), vgslogL[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3]), vgslogL[:3], 1),
                       np.log10(np.linspace(50, 150, 100)))
        ])

        # log lyman continuum (extrapolated)
        vgslogQe = np.concatenate([
            np.zeros(100),  # 0.03-0.43 solar mass stars produce 0 LyC photons
            np.zeros(100),  # 0.43-2.0 solar mass stars produce 0 LyC photons
            np.polyval(np.polyfit(np.log10(vgsMass[-3:]), vgslogQ[-3:], 1),
                       np.log10(np.linspace(8, 18.4, 100))),
            vgslogQ[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3]), vgslogQ[:3], 1),
                       np.log10(np.linspace(50, 150, 100)))
        ])

        mass_luminosity_interpolator_cache[name] = vgsMe, vgslogLe, vgslogQe

        return mass_luminosity_interpolator_cache[name]

    elif name == 'Ekstrom':
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = 1e7  # effectively infinite

        # this query should cacge
        tbl = Vizier.get_catalogs('J/A+A/537/A146/iso')[0]

        match = tbl['logAge'] == 6.5
        masses = tbl['Mass'][match]
        lums = tbl['logL'][match]
        mass_0 = 0.033
        lum_0 = np.log10((mass_0 / masses[0])**3.5 * 10**lums[0])
        mass_f = 200  # extrapolate to 200 Msun...

        lum_f = np.log10(10**lums[-1] * (mass_f / masses[-1])**1.35)

        masses = np.array([mass_0] + masses.tolist() + [mass_f])
        lums = np.array([lum_0] + lums.tolist() + [lum_f])

        # TODO: come up with a half-decent approximation here?  based on logTe?
        logQ = lums - 0.5

        mass_luminosity_interpolator_cache[name] = masses, lums, logQ

        return mass_luminosity_interpolator_cache[name]

    else:
        raise ValueError("Bad grid name {0}".format(name))


def lum_of_star(mass, grid='Ekstrom'):
    """
    Determines the log of total luminosity of a star given its mass,
    based on a grid of stellar properties.

    Available grids (default = ``'Ekstrom'``):

    * ``'Ekstrom'``: values come from the stellar models of 
      `Ekstrom et al. (2012) <https://doi.org/10.1051/0004-6361/201117751>`_.
      **WARNING** Extrapolates for masses outside of [0.8, 64] :math:`M_\odot`.
    * ``'VGS'``: values come from `Vacca, Garmany & Shull (1996) 
      <https://doi.org/10.1086/177020>`_ Table 5. **WARNING** Extrapolates 
      for masses outside of [18.4, 50] :math:`M_\odot`.

    Extrapolation follows the observed relationship between stellar mass
    and luminosity: https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
    """
    masses, lums, _ = mass_luminosity_interpolator(grid)
    return np.interp(mass, masses, lums)


def lum_of_cluster(masses, grid='Ekstrom'):
    r"""
    Determines the log of the integrated luminosity of a cluster.
    Luminosities are calculated using ``lum_of_star``, with the
    same options and defaults.

    ``masses`` is a list or array of masses.
    """
    logL = lum_of_star(masses, grid=grid)
    logLtot = np.log10((10**logL).sum())
    return logLtot


def lyc_of_star(mass, grid='VGS'):
    r"""
    Determines the log of Lyman continuum luminosity (:math:`Q_1`) 
    of a star given its mass. Grid options are the same as 
    ``lum_of_star`` (default = ``'VGS'``)
    """
    masses, _, logQ = mass_luminosity_interpolator(grid)

    return np.interp(mass, masses, logQ)


def lyc_of_cluster(masses, grid='VGS'):
    r"""
    Determines the log of the integrated Lyman continuum luminosity 
    of a cluster using ``lyc_of_star``, with the same options and 
    defaults. Only stars over 8 :math:`M_\odot` contribute.

    ``masses`` is a list or array of masses.
    """
    if max(masses) < 8:
        return 0

    logq = lyc_of_star(masses[masses >= 8], grid=grid)
    logqtot = np.log10((10**logq).sum())
    return logqtot
